"""
ml_model.py
===========
ML-based probabilistic blackout prediction for the Digital Twin Microgrid.

Trains three classifiers on the simulation dataset and evaluates them:
  - Random Forest      (handles non-linearity, feature importance)
  - Gradient Boosting  (strong sequential learner)
  - Logistic Regression (linear baseline)

Input  : datasets/processed/simulation_results.csv  (720 rows x 18 cols)
Output : outputs/model/best_model.pkl
         outputs/model/scaler.pkl
         outputs/model/feature_names.pkl

Features used (3 — physical observables only):
  p_load_mw, p_solar_mw, soc

Dropped time features:
  hour_of_day, is_night → removed to prevent temporal pattern memorisation
  The model must learn physical load/solar/battery relationships, not clock patterns

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

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve, precision_recall_curve, f1_score,
)


# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV   = 'datasets/processed/simulation_results.csv'
OUTPUT_DIR  = 'outputs/model'
RANDOM_SEED = 42
TRAIN_DAYS  = 25     # train on first 25 days (days 0–24)
TEST_BUFFER_HOURS = 0  # gap between train and test (prevents temporal leakage)

# Physical observables only — no time features
# hour_of_day and is_night removed to prevent temporal memorisation
FEATURES = [
    'p_load_mw',
    'p_solar_mw',
    'soc',
]
TARGET = 'blackout'


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
    print(f"  Blackout rate  : {df[TARGET].mean()*100:.1f}%  "
          f"({df[TARGET].sum()} blackout / {len(df)} total)")
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
          f"{y_train.sum()} blackout / {len(y_train)-y_train.sum()} normal "
          f"({y_train.mean()*100:.1f}%)")
    print(f"  Test  : {len(X_test)} hours | "
          f"{y_test.sum()} blackout / {len(y_test)-y_test.sum()} normal "
          f"({y_test.mean()*100:.1f}%)")

    return X_train, X_test, y_train, y_test, df


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators     = 200,
            max_depth        = 8,
            min_samples_leaf = 5,
            class_weight     = 'balanced',
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators  = 200,
            learning_rate = 0.05,
            max_depth     = 4,
            subsample     = 0.8,
            random_state  = RANDOM_SEED,
        ),
        'LogisticRegression': LogisticRegression(
            C            = 1.0,
            class_weight = 'balanced',
            max_iter     = 1000,
            random_state = RANDOM_SEED,
        ),
    }


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test,
) -> tuple:
    """
    Train all models, evaluate on temporal test set.

    Returns
    -------
    tuple: (results dict, scaler, best model name)
    """
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models  = get_models()
    results = {}
    
    # Use TimeSeriesSplit for temporal data (not stratified shuffle)
    # This respects temporal ordering and prevents information leakage
    cv = TimeSeriesSplit(n_splits=3)

    for name, model in models.items():
        X_tr = X_train_s if name == 'LogisticRegression' else X_train
        X_te = X_test_s  if name == 'LogisticRegression' else X_test

        # Cross-validation on train set with temporal split
        cv_data   = X_train_s if name == 'LogisticRegression' else X_train
        cv_scores = cross_val_score(
            model, cv_data, y_train,
            cv=cv, scoring='roc_auc', n_jobs=-1,
        )

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        results[name] = {
            'model':         model,
            'y_pred':        y_pred,
            'y_prob':        y_prob,
            'roc_auc':       roc_auc_score(y_test, y_prob),
            'avg_precision': average_precision_score(y_test, y_prob),
            'brier_score':   brier_score_loss(y_test, y_prob),
            'cv_auc_mean':   cv_scores.mean(),
            'cv_auc_std':    cv_scores.std(),
            'conf_matrix':   confusion_matrix(y_test, y_pred),
            'class_report':  classification_report(
                                 y_test, y_pred,
                                 target_names=['Normal', 'Blackout']),
            'f1_blackout':   f1_score(y_test, y_pred),
        }

    # Select best by F1-score of blackout class
    # F1 balances precision and recall — better than ROC-AUC for imbalanced data
    best_name = max(results, key=lambda k: results[k]['f1_blackout'])
    return results, scaler, best_name


# ── Print Results ─────────────────────────────────────────────────────────────

def print_results(results: dict, best_name: str) -> None:

    print("\n" + "=" * 60)
    print("  Model Evaluation Results  (Temporal Split)")
    print("=" * 60)

    print(f"\n  {'Model':<22} | {'ROC-AUC':>7} | {'F1-BO':>6} | "
          f"{'Brier':>6} | {'CV-AUC':>12}")
    print("  " + "-" * 65)

    for name, r in results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(
            f"  {name:<22} | "
            f"{r['roc_auc']:>7.4f} | "
            f"{r['f1_blackout']:>6.4f} | "
            f"{r['brier_score']:>6.4f} | "
            f"{r['cv_auc_mean']:.4f}+/-{r['cv_auc_std']:.4f}"
            f"{marker}"
        )

    print(f"\n{'='*60}")
    print(f"  Best Model: {best_name}")
    print(f"{'='*60}")
    r = results[best_name]
    print(f"\n  Classification Report:")
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
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        importances = importances / importances.sum()
    else:
        return None

    df_imp = pd.DataFrame({
        'feature':    FEATURES,
        'importance': importances,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"\n  Feature Importances ({best_name}):")
    for _, row in df_imp.iterrows():
        bar = '*' * int(row['importance'] * 50)
        print(f"  {row['feature']:<20} {row['importance']:.4f}  {bar}")

    return df_imp


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
        'RandomForest':       'steelblue',
        'GradientBoosting':   'darkorange',
        'LogisticRegression': 'seagreen',
    }

    # ── ROC Curves ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax1.plot(fpr, tpr, color=colors[name], linewidth=2,
                 label=f"{name[:8]}  AUC={r['roc_auc']:.3f}")
    ax1.plot([0,1],[0,1], 'k--', linewidth=1, label='Random')
    ax1.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Precision-Recall Curves ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r['y_prob'])
        ax2.plot(rec, prec, color=colors[name], linewidth=2,
                 label=f"{name[:8]}  AP={r['avg_precision']:.3f}")
    ax2.axhline(y_test.mean(), color='k', linestyle='--',
                linewidth=1, label=f'Baseline={y_test.mean():.3f}')
    ax2.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Confusion Matrix ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cm = results[best_name]['conf_matrix']
    ax3.imshow(cm, cmap='Blues')
    ax3.set_xticks([0,1]); ax3.set_xticklabels(['Normal','Blackout'])
    ax3.set_yticks([0,1]); ax3.set_yticklabels(['Normal','Blackout'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Confusion Matrix\n({best_name})',
                  fontsize=12, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i,j]),
                     ha='center', va='center', fontsize=16, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')

    # ── Blackout Probability over Time ────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    y_prob_full = results[best_name]['y_prob']
    ax4.plot(y_prob_full, color='crimson', linewidth=0.8,
             alpha=0.8, label='P(Blackout)')
    ax4.axhline(0.5, color='grey', linestyle='--',
                linewidth=1, label='Threshold 0.5')
    blackout_idx = np.where(y_test == 1)[0]
    ax4.scatter(blackout_idx,
                np.ones(len(blackout_idx)) * 0.02,
                c='red', s=8, alpha=0.5, label='Actual Blackout')
    ax4.set_title(f'Predicted Blackout Probability — {best_name} (Test: Days 25–29)',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Test Timestep (hours)')
    ax4.set_ylabel('P(Blackout)')
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
        'Digital Twin — Blackout Prediction  '
        '(Physical Features Only · Temporal Split)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/ml_results.png', bbox_inches='tight', dpi=150)
    print(f"\n  Plot saved → outputs/ml_results.png")
    plt.show()


# ── Reliability Metrics ───────────────────────────────────────────────────────

def compute_reliability_metrics(df: pd.DataFrame) -> None:
    lolp = df['blackout'].mean()
    eens = df.loc[df['blackout'] == 1, 'p_load_mw'].sum()

    print(f"\n{'='*60}")
    print("  Reliability Metrics  (full 30-day simulation)")
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