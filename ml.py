"""
ML Model for Digital Twin Microgrid.

Trains classifiers to predict blackout probability from simulation features.

Models
------
  1. RandomForestClassifier        — tree-based, handles tabular well
  2. GradientBoostingClassifier    — often best accuracy on structured data
  3. LogisticRegression            — simple interpretable baseline
  4. LSTM (PyTorch)                — captures sequential/temporal patterns  ← NEW
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import joblib

# Optional PyTorch import — LSTM is skipped gracefully if not available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] PyTorch not found. LSTM model will be skipped.")
    print("       Install with: pip install torch")


# ── Feature definitions ──────────────────────────────────────────────────────

BASE_FEATURES = [
    'p_load_mw', 'p_solar_mw', 'p_battery_mw', 'p_grid_mw',
    'soc', 'v_min', 'v_max', 'v_mean', 'line_loading_max',
    'hour_of_day', 'converged',
]

ENGINEERED_FEATURES = [
    'solar_load_ratio', 'soc_deficit', 'v_violation_margin',
    'net_power_balance', 'is_night',
]

# Temporal features added by digital_twin.add_temporal_features()
TEMPORAL_FEATURES = [
    'p_load_lag_1h', 'p_load_lag_3h', 'p_solar_lag_1h', 'soc_lag_1h',
    'load_rolling_avg_6h', 'solar_rolling_avg_6h', 'v_min_rolling_min_6h',
    'soc_rate_of_change', 'load_rate_of_change',
    'load_solar_diff', 'grid_stress_index',
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES + TEMPORAL_FEATURES


# ── Feature engineering ───────────────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features (domain knowledge) on top of raw simulation columns.

    solar_load_ratio    — how much of the load PV covers
    soc_deficit         — how depleted the battery is
    v_violation_margin  — distance from voltage limit (negative = violation)
    net_power_balance   — MW surplus (+) or deficit (−)
    is_night            — 1 if hour < 6 or hour > 20
    """
    df = df.copy()
    df['solar_load_ratio']   = df['p_solar_mw'] / (df['p_load_mw'] + 1e-6)
    df['soc_deficit']        = 1.0 - df['soc']
    df['v_violation_margin'] = df['v_min'] - 0.95
    df['net_power_balance']  = df['p_solar_mw'] + df['p_battery_mw'] - df['p_load_mw']
    df['is_night']           = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 20)).astype(int)
    return df


def _get_available_features(df: pd.DataFrame) -> list:
    """Return ALL_FEATURES that are actually present in df."""
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing   = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"[INFO] {len(missing)} features not in DataFrame (skipped): {missing}")
    return available


# ── LSTM model (PyTorch) ──────────────────────────────────────────────────────

class LSTMBlackoutPredictor(nn.Module):
    """
    Sequence-to-one LSTM that reads a window of W consecutive timesteps
    and predicts the blackout probability at the final step.

    Architecture
    ────────────
        Input  : (batch, seq_len, n_features)
        LSTM   : 2 layers, hidden_size=64, dropout=0.3 between layers
        Linear : hidden_size → 1
        Sigmoid: output ∈ (0, 1)  ← blackout probability
    """
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)         # out: (batch, seq_len, hidden)
        last    = out[:, -1, :]       # take final timestep: (batch, hidden)
        return self.sigmoid(self.fc(last)).squeeze(-1)  # (batch,)


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Slide a window of length seq_len over X/y to build sequence samples.

    Returns
    -------
    X_seq : np.ndarray  shape (n_samples, seq_len, n_features)
    y_seq : np.ndarray  shape (n_samples,)  — label at the LAST step
    """
    n = len(X)
    X_seq, y_seq = [], []
    for i in range(seq_len, n):
        X_seq.append(X[i - seq_len: i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seq_len: int = 24,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> dict:
    """
    Train an LSTM blackout predictor on sequential data.

    Parameters
    ----------
    X_train, y_train : training arrays (already scaled)
    X_test,  y_test  : test arrays (already scaled)
    seq_len          : look-back window in hours (default 24 = 1 day)
    epochs           : training epochs
    batch_size       : mini-batch size
    lr               : Adam learning rate
    device           : 'cpu' or 'cuda'

    Returns
    -------
    dict with keys: model, roc_auc, avg_precision, brier_score, report
    """
    if not TORCH_AVAILABLE:
        return {}

    print(f"\n{'=' * 40}")
    print(f"Training: LSTM (seq_len={seq_len}, epochs={epochs})")

    # Build sliding-window sequences
    X_tr_seq, y_tr_seq = _build_sequences(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = _build_sequences(X_test,  y_test,  seq_len)

    if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
        print("[WARNING] Not enough data to build LSTM sequences. Skipping.")
        return {}

    n_features = X_tr_seq.shape[2]

    # Class imbalance weight
    n_pos = y_tr_seq.sum()
    n_neg = len(y_tr_seq) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], dtype=torch.float32).to(device)

    # DataLoaders
    tr_ds = TensorDataset(
        torch.from_numpy(X_tr_seq),
        torch.from_numpy(y_tr_seq),
    )
    te_ds = TensorDataset(
        torch.from_numpy(X_te_seq),
        torch.from_numpy(y_te_seq),
    )
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    # Model, loss, optimiser
    model     = LSTMBlackoutPredictor(n_features=n_features).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Re-define model to output raw logits for BCEWithLogitsLoss
    class LSTMLogits(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.lstm = base.lstm
            self.fc   = base.fc
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model_logits = LSTMLogits(model).to(device)
    optimiser    = torch.optim.Adam(model_logits.parameters(), lr=lr)
    scheduler    = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

    # Training loop
    for epoch in range(1, epochs + 1):
        model_logits.train()
        epoch_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model_logits(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model_logits.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={epoch_loss/len(tr_dl):.4f}")

    # Evaluation
    model_logits.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            logits = model_logits(xb)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(yb.numpy().astype(int))

    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = 0.0
    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = 0.0
    brier = brier_score_loss(y_true, y_prob)

    print(f"  ROC-AUC        : {roc:.4f}")
    print(f"  Avg Precision  : {ap:.4f}")
    print(f"  Brier Score    : {brier:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Wrap: expose a sklearn-like predict_proba for consistent saving
    # We store the raw PyTorch module plus seq_len for inference
    return {
        'model':         model_logits,
        'seq_len':       seq_len,
        'roc_auc':       roc,
        'avg_precision': ap,
        'brier_score':   brier,
        'report':        classification_report(y_true, y_pred, zero_division=0),
        'is_lstm':       True,
    }


# ── sklearn model training ────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> tuple:
    """
    Train all classifiers (sklearn + LSTM) and evaluate.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results from digital_twin.py — must already contain
        temporal features if add_temporal_features() was called there.

    Returns
    -------
    tuple(dict, StandardScaler, pd.DataFrame)
        (results dict, scaler, df with engineered features)
    """
    # Add domain-knowledge features on top of whatever's in df
    df = add_engineered_features(df)

    feature_cols = _get_available_features(df)
    print(f"\nFeature set: {len(feature_cols)} features")
    print(f"  Base      : {len(BASE_FEATURES)}")
    print(f"  Engineered: {len(ENGINEERED_FEATURES)}")
    temporal_present = [f for f in TEMPORAL_FEATURES if f in feature_cols]
    print(f"  Temporal  : {len(temporal_present)} "
          f"({'present' if temporal_present else 'NOT PRESENT — run updated digital_twin.py'})")

    X = df[feature_cols].values
    y = df['blackout'].values

    n_normal   = (y == 0).sum()
    n_blackout = (y == 1).sum()
    print(f"\nClass distribution — Normal: {n_normal} | Blackout: {n_blackout}")

    if n_blackout == 0:
        print("[WARNING] No blackout events in data. Cannot train ML model.")
        return {}, None, df
    if n_blackout < 5:
        print("[WARNING] Very few blackout events. Results may be unreliable.")

    # Train/test split (no shuffle — preserve temporal order)
    split_idx  = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Feature scaling (needed for LogReg; harmless for trees)
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    # ── sklearn models ────────────────────────────────────────────────────
    sklearn_models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
    }

    results = {}

    for name, model in sklearn_models.items():
        print(f"\n{'=' * 40}")
        print(f"Training: {name}")

        if name == 'LogisticRegression':
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        try:
            roc = roc_auc_score(y_test, y_prob)
        except ValueError:
            roc = 0.0
        try:
            ap = average_precision_score(y_test, y_prob)
        except ValueError:
            ap = 0.0
        brier = brier_score_loss(y_test, y_prob)

        results[name] = {
            'model':         model,
            'roc_auc':       roc,
            'avg_precision': ap,
            'brier_score':   brier,
            'report':        classification_report(y_test, y_pred, zero_division=0),
        }

        print(f"  ROC-AUC        : {roc:.4f}")
        print(f"  Avg Precision  : {ap:.4f}")
        print(f"  Brier Score    : {brier:.4f}")
        print(results[name]['report'])

    # ── LSTM model ────────────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        device      = 'cuda' if torch.cuda.is_available() else 'cpu'
        lstm_result = train_lstm(
            X_train_s, y_train,
            X_test_s,  y_test,
            seq_len=24,   # 24-hour look-back window
            epochs=30,
            batch_size=32,
            lr=1e-3,
            device=device,
        )
        if lstm_result:
            results['LSTM'] = lstm_result
    else:
        print("\n[SKIP] LSTM — PyTorch not installed.")

    # ── Save best sklearn model (by ROC-AUC) ─────────────────────────────
    sklearn_results = {k: v for k, v in results.items() if 'is_lstm' not in v}
    if sklearn_results:
        best_name  = max(sklearn_results, key=lambda k: sklearn_results[k]['roc_auc'])
        best_model = sklearn_results[best_name]['model']

        os.makedirs('outputs/model', exist_ok=True)
        joblib.dump(best_model, 'outputs/model/best_model.pkl')
        joblib.dump(scaler,     'outputs/model/scaler.pkl')
        joblib.dump(feature_cols, 'outputs/model/feature_cols.pkl')
        print(f"\nBest sklearn model : {best_name} "
              f"(ROC-AUC={sklearn_results[best_name]['roc_auc']:.4f})")
        print(f"Saved to           : outputs/model/")

    # Save LSTM separately if available
    if TORCH_AVAILABLE and 'LSTM' in results and results['LSTM']:
        lstm_path = 'outputs/model/lstm_model.pt'
        torch.save(results['LSTM']['model'].state_dict(), lstm_path)
        print(f"LSTM model saved   : {lstm_path} "
              f"(ROC-AUC={results['LSTM']['roc_auc']:.4f})")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"{'Model':<22} {'ROC-AUC':>9} {'Avg-Prec':>10} {'Brier':>8}")
    print(f"{'─' * 55}")
    for name, r in results.items():
        print(f"{name:<22} {r['roc_auc']:>9.4f} {r['avg_precision']:>10.4f} "
              f"{r['brier_score']:>8.4f}")
    print(f"{'─' * 55}")

    return results, scaler, df


# ── Reliability metrics ───────────────────────────────────────────────────────

def compute_reliability_metrics(df: pd.DataFrame) -> dict:
    """
    Compute power system reliability metrics.

    LOLP : Loss of Load Probability = fraction of hours with blackout
    EENS : Expected Energy Not Served = total unserved energy during blackouts
    """
    T              = len(df)
    blackout_hours = df['blackout'].sum()
    lolp           = blackout_hours / T if T > 0 else 0
    eens           = df.loc[df['blackout'] == 1, 'p_load_mw'].sum()

    print(f"\nReliability Metrics:")
    print(f"  LOLP : {lolp:.4f} ({lolp * 100:.2f}%)")
    print(f"  EENS : {eens:.3f} MWh")

    return {'LOLP': lolp, 'EENS': eens}


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    csv_path = 'datasets/processed/simulation_results.csv'
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found. Run digital_twin.py first.")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        print(f"Columns: {list(df.columns)}")

        results, scaler, df_feat = train_and_evaluate(df)
        compute_reliability_metrics(df_feat)