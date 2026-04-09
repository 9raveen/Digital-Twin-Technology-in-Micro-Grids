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
         outputs/ml_report.txt

Features used (13):
  hour_of_day, is_night,
  p_load_mw, p_solar_mw, p_battery_mw, p_grid_mw,
  soc, v_min, v_max, v_mean, line_loading_max, p_loss_mw, v_margin

Dropped:
  timestep  → identifier, not a physical feature
  day       → identifier
  converged → constant (all 1.0 in this simulation)
  severity  → string, redundant with v_margin
  blackout  → target label
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve, precision_recall_curve,f1_score,
)


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV   = 'datasets/processed/simulation_results.csv'
OUTPUT_DIR  = 'outputs/model'
REPORT_PATH = 'outputs/ml_report.txt'
RANDOM_SEED = 42
TEST_SIZE   = 0.2    # 80/20 train/test split

FEATURES = [
    'hour_of_day',
    'is_night',
    'p_load_mw',
    'p_solar_mw',
    'soc',
]
TARGET = 'blackout'


# ── Data Loading & Preparation ────────────────────────────────────────────────

def load_and_prepare(path: str) -> tuple:
    """
    Load simulation results, select features, split train/test.

    Returns
    -------
    tuple: X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(path)

    print(f"  Dataset shape  : {df.shape}")
    print(f"  Blackout rate  : {df[TARGET].mean()*100:.1f}%  "
          f"({df[TARGET].sum()} blackout / {len(df)} total)")
    print(f"  Features used  : {len(FEATURES)}")

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,     # preserve class ratio in both splits
    )

    print(f"\n  Train set : {len(X_train)} rows  "
          f"({y_train.sum()} blackout / {len(y_train)-y_train.sum()} normal)")
    print(f"  Test set  : {len(X_test)} rows   "
          f"({y_test.sum()} blackout / {len(y_test)-y_test.sum()} normal)")

    return X_train, X_test, y_train, y_test


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return dict of model name → unfitted classifier.
    All models configured for class imbalance with class_weight='balanced'.
    """
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators  = 200,
            max_depth     = 8,
            min_samples_leaf = 5,
            class_weight  = 'balanced',
            random_state  = RANDOM_SEED,
            n_jobs        = -1,
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators  = 200,
            learning_rate = 0.05,
            max_depth     = 4,
            subsample     = 0.8,
            random_state  = RANDOM_SEED,
        ),
        'LogisticRegression': LogisticRegression(
            C             = 1.0,
            class_weight  = 'balanced',
            max_iter      = 1000,
            random_state  = RANDOM_SEED,
        ),
    }


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test,
) -> tuple:
    """
    Train all models, evaluate on test set, return results dict.

    Logistic Regression uses StandardScaler.
    Tree-based models use raw features.

    Returns
    -------
    tuple: (results dict, scaler, best model name)
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models  = get_models()
    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for name, model in models.items():

        # Use scaled data for Logistic Regression, raw for tree models
        X_tr = X_train_s if name == 'LogisticRegression' else X_train
        X_te = X_test_s  if name == 'LogisticRegression' else X_test

        # ── Cross-validation ───────────────────────────────────────
        cv_data = X_train_s if name == 'LogisticRegression' else X_train
        cv_scores = cross_val_score(
            model, cv_data, y_train,
            cv=cv, scoring='roc_auc', n_jobs=-1,
        )

        # ── Fit on full train set ──────────────────────────────────
        model.fit(X_tr, y_train)

        # ── Predict ───────────────────────────────────────────────
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        # ── Metrics ───────────────────────────────────────────────
        results[name] = {
            'model':          model,
            'y_pred':         y_pred,
            'y_prob':         y_prob,
            'roc_auc':        roc_auc_score(y_test, y_prob),
            'avg_precision':  average_precision_score(y_test, y_prob),
            'brier_score':    brier_score_loss(y_test, y_prob),
            'cv_auc_mean':    cv_scores.mean(),
            'cv_auc_std':     cv_scores.std(),
            'conf_matrix':    confusion_matrix(y_test, y_pred),
            'class_report':   classification_report(y_test, y_pred,
                                  target_names=['Normal', 'Blackout']),
        }
    for name, r in results.items():
        r['f1_blackout'] = f1_score(y_test, r['y_pred'])

    best_name = max(results, key=lambda k: results[k]['f1_blackout'])
    return results, scaler, best_name


# ── Print Results ─────────────────────────────────────────────────────────────

def print_results(results: dict, best_name: str) -> None:
    """Print evaluation results for all models."""

    print("\n" + "=" * 60)
    print("  Model Evaluation Results")
    print("=" * 60)

    # ── Comparison table ───────────────────────────────────────────
    print(f"\n  {'Model':<22} | {'ROC-AUC':>7} | {'Avg-Prec':>8} | "
          f"{'Brier':>6} | {'CV-AUC':>10}")
    print("  " + "-" * 62)

    for name, r in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(
            f"  {name:<22} | "
            f"{r['roc_auc']:>7.4f} | "
            f"{r['avg_precision']:>8.4f} | "
            f"{r['brier_score']:>6.4f} | "
            f"{r['cv_auc_mean']:.4f}±{r['cv_auc_std']:.4f}"
            f"{marker}"
        )

    # ── Detailed report for best model ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Best Model: {best_name}")
    print(f"{'='*60}")
    r = results[best_name]
    print(f"\n  Classification Report:")
    print(r['class_report'])
    print(f"  Confusion Matrix:")
    cm = r['conf_matrix']
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Precision  : {tp/(tp+fp):.4f}  (of predicted blackouts, how many were real)")
    print(f"  Recall     : {tp/(tp+fn):.4f}  (of real blackouts, how many were caught)")
    print(f"  Specificity: {tn/(tn+fp):.4f}  (of real normals, how many were correct)")


# ── Feature Importance ────────────────────────────────────────────────────────

def get_feature_importance(results: dict, best_name: str) -> pd.DataFrame:
    """Extract feature importances from best model if available."""
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
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:<20} {row['importance']:.4f}  {bar}")

    return df_imp


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(
    results: dict,
    best_name: str,
    df_imp: pd.DataFrame,
    y_test: np.ndarray,
) -> None:
    """Plot ROC curves, PR curves, confusion matrix, feature importance."""

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    colors = {
        'RandomForest':      'steelblue',
        'GradientBoosting':  'darkorange',
        'LogisticRegression':'seagreen',
    }

    # ── Plot 1: ROC Curves ────────────────────────────────────────
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

    # ── Plot 2: Precision-Recall Curves ──────────────────────────
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

    # ── Plot 3: Confusion Matrix (best model) ─────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cm = results[best_name]['conf_matrix']
    im = ax3.imshow(cm, cmap='Blues')
    ax3.set_xticks([0,1]); ax3.set_xticklabels(['Normal','Blackout'])
    ax3.set_yticks([0,1]); ax3.set_yticklabels(['Normal','Blackout'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Confusion Matrix\n({best_name})',
                  fontsize=12, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i,j]),
                     ha='center', va='center',
                     fontsize=16, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')

    # ── Plot 4: Blackout Probability over Time ────────────────────
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
    ax4.set_title(f'Predicted Blackout Probability — {best_name}',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Test Timestep')
    ax4.set_ylabel('P(Blackout)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Feature Importance ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if df_imp is not None:
        top10 = df_imp.head(10)
        ax5.barh(top10['feature'][::-1],
                 top10['importance'][::-1],
                 color='steelblue', alpha=0.8)
        ax5.set_title('Top 10 Feature Importances',
                      fontsize=12, fontweight='bold')
        ax5.set_xlabel('Importance')
        ax5.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Digital Twin — Blackout Prediction Results',
                 fontsize=14, fontweight='bold', y=1.01)

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/ml_results.png', bbox_inches='tight', dpi=150)
    print(f"\n  Plot saved → outputs/ml_results.png")
    plt.show()


# ── Reliability Metrics ───────────────────────────────────────────────────────

def compute_reliability_metrics(df: pd.DataFrame, best_name: str,
                                 results: dict,
                                 X_test: np.ndarray,
                                 scaler: StandardScaler) -> None:
    """
    Compute LOLP and EENS from full simulation dataset.

    LOLP (Loss of Load Probability) = fraction of hours with blackout
    EENS (Expected Energy Not Served) = load during blackout hours (MWh)
    """
    lolp = df['blackout'].mean()
    eens = df.loc[df['blackout'] == 1, 'p_load_mw'].sum()   # 1 hr timestep → MWh

    print(f"\n{'='*60}")
    print("  Reliability Metrics")
    print(f"{'='*60}")
    print(f"  LOLP  : {lolp:.4f}  ({lolp*100:.2f}% of hours)")
    print(f"  EENS  : {eens:.3f} MWh over 30 days")
    print(f"  EENS/day : {eens/30:.3f} MWh/day")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_artifacts(results: dict, best_name: str,
                   scaler: StandardScaler) -> None:
    """Save best model, scaler, and feature names."""
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

    # ── Load data ──────────────────────────────────────────────────
    print("\n[1/5] Loading and preparing data...")
    df = pd.read_csv(INPUT_CSV)
    X_train, X_test, y_train, y_test = load_and_prepare(INPUT_CSV)

    # ── Train & evaluate ───────────────────────────────────────────
    print("\n[2/5] Training and evaluating models...")
    results, scaler, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # ── Print results ──────────────────────────────────────────────
    print("\n[3/5] Results...")
    print_results(results, best_name)

    # ── Feature importance ─────────────────────────────────────────
    df_imp = get_feature_importance(results, best_name)

    # ── Reliability metrics ────────────────────────────────────────
    compute_reliability_metrics(df, best_name, results, X_test, scaler)

    # ── Plot ───────────────────────────────────────────────────────
    print("\n[4/5] Generating plots...")
    plot_results(results, best_name, df_imp, y_test)

    # ── Save ───────────────────────────────────────────────────────
    print("\n[5/5] Saving artifacts...")
    save_artifacts(results, best_name, scaler)

    print("\n" + "=" * 60)
    print(f"  Complete. Best model: {best_name}")
    print("  Ready for dashboard/app.py")
    print("=" * 60)