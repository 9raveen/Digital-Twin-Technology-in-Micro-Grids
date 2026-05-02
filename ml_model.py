"""
ml_model.py
===========
ML-based probabilistic blackout risk prediction for the Digital Twin Microgrid.

Trains regression models on the simulation dataset and evaluates them:
  - Random Forest Regressor      (handles non-linearity, feature importance)
  - Gradient Boosting Regressor  (strong sequential learner)
  - XGBoost Regressor            (optimized gradient boosting)
  - Ridge Regression             (linear baseline)

Predicts continuous risk scores [0, 1], then converts to binary blackout at 0.5 threshold.

Input  : datasets/processed/simulation_results.csv  (720 rows)
Output : outputs/model/best_model.pkl
         outputs/model/scaler.pkl
         outputs/model/feature_names.pkl

Target: risk_score (continuous, [0, 1] - smooth blackout probability)

Features used (11 — physical observables + temporal lags):
  Current: p_load_mw, p_solar_mw, soc, line_loading_max, p_grid_mw
  Lags: p_load_lag_1/2/3, p_solar_lag_1/2/3

Removed features (to prevent data leakage):
  v_min — directly determines the label, causes information leakage
  hour_of_day, is_night — removed to prevent temporal pattern memorisation
  The model must learn physical load/solar/battery relationships, not voltage

Train/test split — TEMPORAL (not random):
  Train : days  0–24  (600 hours)
  Test  : days 25–29  (120 hours)
  Rationale: mirrors real deployment — train on history, predict future
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import Ridge
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics         import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve, precision_recall_curve, f1_score,
)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV   = 'datasets/processed/simulation_results.csv'
OUTPUT_DIR  = 'outputs/model'
RANDOM_SEED = 42
TRAIN_DAYS  = 25     # train on first 25 days (days 0–24)
TEST_BUFFER_HOURS = 0  # gap between train and test (prevents temporal leakage)

# Physical observables only — no time features, no voltage leakage
# hour_of_day and is_night removed to prevent temporal memorisation
# v_min removed to prevent data leakage (it directly determines the label)
# Includes lag features for temporal context
FEATURES = [
    'p_load_mw',
    'p_solar_mw',
    'soc',
    'line_loading_max',
    'p_grid_mw',
    'p_load_lag_1', 'p_load_lag_2', 'p_load_lag_3',
    'p_solar_lag_1', 'p_solar_lag_2', 'p_solar_lag_3',
]
TARGET = 'risk_score'


# ── Data Loading & Preparation ────────────────────────────────────────────────

def load_and_prepare(path: str) -> tuple:
    """
    Load simulation results, apply temporal train/test split with buffer.

    Train : days  0 to TRAIN_DAYS-1  (first 25 days)
    Test  : days  TRAIN_DAYS + buffer to end  (last 5+ days, after gap)

    Temporal buffer prevents data leakage by adding a gap between train
    and test periods. Test features cannot causally influence train labels.

    Returns
    -------
    tuple: X_train, X_test, y_train, y_test, df
    """
    df = pd.read_csv(path)

    print(f"  Dataset shape  : {df.shape}")
    print(f"  Risk score range : [{df[TARGET].min():.4f}, {df[TARGET].max():.4f}]")
    print(f"  Risk score mean  : {df[TARGET].mean():.4f}")
    print(f"  Features used  : {FEATURES}")

    buffer_days = TEST_BUFFER_HOURS // 24
    test_start_day = TRAIN_DAYS + buffer_days
    print(f"  Split type     : TEMPORAL  (train days 0–{TRAIN_DAYS-1}, "
          f"buffer {TRAIN_DAYS}–{test_start_day-1}, "
          f"test days {test_start_day}–{df['day'].max()})")

    train_df = df[df['day'] <  TRAIN_DAYS]
    test_df  = df[df['day'] >= test_start_day]

    X_train = train_df[FEATURES].values
    X_test  = test_df[FEATURES].values
    y_train = train_df[TARGET].values
    y_test  = test_df[TARGET].values

    print(f"\n  Train : {len(X_train)} hours | "
          f"Risk mean {y_train.mean():.4f} | Risk range [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  Test  : {len(X_test)} hours | "
          f"Risk mean {y_test.mean():.4f} | Risk range [{y_test.min():.4f}, {y_test.max():.4f}]")

    return X_train, X_test, y_train, y_test, df


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    """Get regression models for risk score prediction."""
    models = {
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators     = 200,
            max_depth        = 8,
            min_samples_leaf = 5,
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        ),
        'GradientBoostingRegressor': GradientBoostingRegressor(
            n_estimators  = 200,
            learning_rate = 0.05,
            max_depth     = 4,
            subsample     = 0.8,
            random_state  = RANDOM_SEED,
        ),
        'Ridge': Ridge(
            alpha = 1.0,
        ),
    }

    if XGBOOST_AVAILABLE:
        models['XGBRegressor'] = XGBRegressor(
            n_estimators  = 200,
            max_depth     = 5,
            learning_rate = 0.05,
            random_state  = RANDOM_SEED,
        )

    return models


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test,
) -> tuple:
    """
    Train all regression models, evaluate on temporal test set.

    Predicts continuous risk scores, then converts to binary blackout predictions.

    Returns
    -------
    tuple: (results dict, scaler, best model name)
    """
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models  = get_models()
    results = {}

    cv = TimeSeriesSplit(n_splits=3)

    for name, model in models.items():
        X_tr = X_train_s if name == 'Ridge' else X_train
        X_te = X_test_s  if name == 'Ridge' else X_test

        # Cross-validation on train set
        cv_data = X_train_s if name == 'Ridge' else X_train
        cv_scores = cross_val_score(
            model, cv_data, y_train,
            cv=cv, scoring='r2', n_jobs=-1,
        )

        model.fit(X_tr, y_train)
        y_pred_risk = model.predict(X_te)

        # Clip to [0, 1] for probabilistic interpretation
        y_pred_risk = np.clip(y_pred_risk, 0, 1)

        # Convert to binary blackout prediction (threshold 0.5)
        y_pred = (y_pred_risk > 0.5).astype(int)

        # Compute regression metrics
        mae = mean_absolute_error(y_test, y_pred_risk)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_risk))
        r2 = r2_score(y_test, y_pred_risk)

        # Also compute classification metrics for blackout threshold
        conf_mat = confusion_matrix(y_test > 0.5, y_pred)

        results[name] = {
            'model':         model,
            'y_pred_risk':   y_pred_risk,
            'y_pred':        y_pred,
            'mae':           mae,
            'rmse':          rmse,
            'r2':            r2,
            'cv_r2_mean':    cv_scores.mean(),
            'cv_r2_std':     cv_scores.std(),
            'conf_matrix':   conf_mat,
            'class_report':  classification_report(
                                 y_test > 0.5, y_pred,
                                 target_names=['Normal', 'Blackout']),
            'f1_blackout':   f1_score(y_test > 0.5, y_pred),
        }

    # Select best by R² score
    best_name = max(results, key=lambda k: results[k]['r2'])
    return results, scaler, best_name


# ── Print Results ─────────────────────────────────────────────────────────────

def print_results(results: dict, best_name: str) -> None:

    print("\n" + "=" * 60)
    print("  Model Evaluation Results  (Risk Score Regression)")
    print("=" * 60)

    print(f"\n  {'Model':<25} | {'R² Score':>8} | {'RMSE':>8} | "
          f"{'MAE':>8} | {'CV-R²':>12}")
    print("  " + "-" * 70)

    for name, r in results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(
            f"  {name:<25} | "
            f"{r['r2']:>8.4f} | "
            f"{r['rmse']:>8.4f} | "
            f"{r['mae']:>8.4f} | "
            f"{r['cv_r2_mean']:.4f}+/-{r['cv_r2_std']:.4f}"
            f"{marker}"
        )

    print(f"\n{'='*60}")
    print(f"  Best Model: {best_name}")
    print(f"{'='*60}")
    r = results[best_name]
    print(f"\n  Risk Score Regression Metrics:")
    print(f"    R² Score (test)  : {r['r2']:.4f}")
    print(f"    RMSE             : {r['rmse']:.4f}")
    print(f"    MAE              : {r['mae']:.4f}")
    print(f"    CV R² (mean±std) : {r['cv_r2_mean']:.4f}±{r['cv_r2_std']:.4f}")

    print(f"\n  Blackout Classification (threshold=0.5):")
    print(r['class_report'])

    cm = r['conf_matrix']
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Precision  : {tp/(tp+fp+1e-9):.4f}  "
          f"(of predicted blackouts, how many were real)")
    print(f"  Recall     : {tp/(tp+fn+1e-9):.4f}  "
          f"(of real blackouts, how many were caught)")
    print(f"  Specificity: {tn/(tn+fp+1e-9):.4f}  "
          f"(of real normals, how many were correct)")
    print(f"  F1 Blackout: {r['f1_blackout']:.4f}")


# ── Feature Importance ────────────────────────────────────────────────────────

def get_feature_importance(results: dict, best_name: str) -> pd.DataFrame:
    model = results[best_name]['model']

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        importance = importance / importance.max()
    else:
        return None

    features = FEATURES
    print(f"\n  Feature Importances ({best_name}):")
    for f, imp in zip(features, importance):
        print(f"  {f:20s} {imp:.4f}")

    return pd.DataFrame({
        'feature': features,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(
    results:   dict,
    best_name: str,
    df_imp:    pd.DataFrame,
    y_test:    np.ndarray,
) -> None:

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    colors = {
        'RandomForestRegressor':       'steelblue',
        'GradientBoostingRegressor':   'darkorange',
        'Ridge': 'seagreen',
        'XGBRegressor': 'purple'
    }

    # ── Risk Score Predictions vs Actuals ─────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for name, r in results.items():
        ax1.scatter(y_test, r['y_pred_risk'], alpha=0.5,
                   color=colors.get(name, 'gray'), s=20,
                   label=f"{name} (R²={r['r2']:.3f})")
    ax1.plot([0,1], [0,1], 'k--', linewidth=1, label='Perfect')
    ax1.set_title('Risk Score: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Actual Risk Score')
    ax1.set_ylabel('Predicted Risk Score')
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # ── Model Comparison ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    model_names = list(results.keys())
    r2_scores = [results[n]['r2'] for n in model_names]
    ax2.barh(model_names, r2_scores, color=[colors.get(n, 'gray') for n in model_names])
    ax2.set_title('Model R² Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('R² Score')
    ax2.grid(True, alpha=0.3, axis='x')

    # ── Confusion Matrix (blackout threshold) ──────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cm = results[best_name]['conf_matrix']
    ax3.imshow(cm, cmap='Blues')
    ax3.set_xticks([0,1]); ax3.set_xticklabels(['Normal','Blackout'])
    ax3.set_yticks([0,1]); ax3.set_yticklabels(['Normal','Blackout'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Blackout Classification\n({best_name})',
                  fontsize=12, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i,j]),
                     ha='center', va='center', fontsize=16, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')

    # ── Risk Score over Time ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    y_risk_full = results[best_name]['y_pred_risk']
    ax4.plot(y_risk_full, color='crimson', linewidth=0.8,
             alpha=0.8, label='Predicted Risk')
    ax4.plot(y_test, color='steelblue', linewidth=0.8,
             alpha=0.5, label='Actual Risk')
    ax4.axhline(0.5, color='grey', linestyle='--',
                linewidth=1, label='Threshold 0.5')
    blackout_idx = np.where(y_test > 0.5)[0]
    ax4.scatter(blackout_idx,
                np.ones(len(blackout_idx)) * 0.02,
                c='red', s=8, alpha=0.5, label='Actual Blackout')
    ax4.set_title(f'Risk Score Prediction — {best_name} (Test: Days 25–29)',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Test Timestep (hours)')
    ax4.set_ylabel('Risk Score')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── Feature Importance ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if df_imp is not None:
        ax5.barh(df_imp['feature'][::-1],
                 df_imp['importance'][::-1],
                 color='steelblue', alpha=0.8)
        ax5.set_title('Feature Importances', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Importance')
        ax5.grid(True, alpha=0.3, axis='x')

    plt.suptitle(
        'Digital Twin — Blackout Risk Prediction (Regression)  '
        '(Physical Features Only · Temporal Split)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/ml_results.png', bbox_inches='tight', dpi=150)
    print(f"\n  Plot saved → outputs/ml_results.png")
    plt.show()


# ── Reliability Metrics ───────────────────────────────────────────────────────

def compute_reliability_metrics(df: pd.DataFrame) -> None:
    """Compute LOLP and EENS from risk scores (threshold > 0.5 for blackout)."""
    # Convert risk scores to binary blackout using threshold
    blackout_mask = df['risk_score'] > 0.5
    lolp = blackout_mask.mean()
    eens = df.loc[blackout_mask, 'p_load_mw'].sum()

    print(f"\n{'='*60}")
    print("  Reliability Metrics  (from risk scores, threshold=0.5)")
    print(f"{'='*60}")
    print(f"  LOLP     : {lolp:.4f}  ({lolp*100:.2f}% of hours)")
    print(f"  EENS     : {eens:.3f} MWh over 30 days")
    print(f"  EENS/day : {eens/30:.3f} MWh/day")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_artifacts(results: dict, best_name: str, scaler: StandardScaler) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path   = os.path.join(OUTPUT_DIR, 'best_model.pkl')
    scaler_path  = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    feature_path = os.path.join(OUTPUT_DIR, 'feature_names.pkl')

    joblib.dump(results[best_name]['model'], model_path)
    joblib.dump(scaler,                      scaler_path)
    joblib.dump(FEATURES,                    feature_path)

    print(f"\n  Saved best model ({best_name}) → {model_path}")
    print(f"  Saved scaler                  → {scaler_path}")
    print(f"  Saved feature names           → {feature_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 60)
    print("  ML Model — Blackout Prediction")
    print("=" * 60)

    print("\n[1/5] Loading and preparing data...")
    X_train, X_test, y_train, y_test, df = load_and_prepare(INPUT_CSV)

    print("\n[2/5] Training and evaluating models...")
    results, scaler, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    print("\n[3/5] Results...")
    print_results(results, best_name)
    df_imp = get_feature_importance(results, best_name)

    compute_reliability_metrics(df)

    print("\n[4/5] Generating plots...")
    plot_results(results, best_name, df_imp, y_test)

    print("\n[5/5] Saving artifacts...")
    save_artifacts(results, best_name, scaler)

    print("\n" + "=" * 60)
    print(f"  Complete. Best model: {best_name}")
    print("  Ready for dashboard/app.py")
    print("=" * 60)