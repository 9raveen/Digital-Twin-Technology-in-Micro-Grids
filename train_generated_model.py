"""
train_generated_model.py
========================
Training pipeline for datasets/generated.

This script trains a blackout classifier on the generated scenario dataset:
  - Uses datasets/generated/train.csv for fitting
  - Uses datasets/generated/val.csv for model selection
  - Uses datasets/generated/test.csv for final evaluation
  - Saves the best model as a probability classifier compatible with app.py

Target:
  blackout (binary, 0/1)

Features used:
  p_load_mw, p_solar_mw, soc, line_loading_max, p_grid_mw,
  p_load_lag_1/2/3, p_solar_lag_1/2/3

Excluded from training:
  timestep, day, hour_of_day, is_night, v_min, v_min_measured, v_max,
  v_mean, p_loss_mw, converged, v_margin, severity, risk_score,
  scenario_id, scenario_type, blackout

Artifacts saved to output_dir:
  - best_model.pkl
  - feature_names.pkl
  - metrics.json
  - confusion_matrix.png
  - test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DEFAULT_INPUT_DIR = Path("datasets/generated")
DEFAULT_OUTPUT_DIR = Path("outputs/model")
RANDOM_SEED = 42
TARGET = "blackout"
DEFAULT_TRAIN_FRAC = 0.5
DEFAULT_VAL_FRAC = 0.175

FEATURES: List[str] = [
    "p_load_mw",
    "p_solar_mw",
    "soc",
    "line_loading_max",
    "p_grid_mw",
    "p_load_lag_1",
    "p_load_lag_2",
    "p_load_lag_3",
    "p_solar_lag_1",
    "p_solar_lag_2",
    "p_solar_lag_3",
]


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def stratified_train_val_test_split(
    df: pd.DataFrame,
    random_seed: int,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> DatasetSplits:
    """Create train/val/test splits stratified by blackout labels."""
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column '{TARGET}' in generated dataset")

    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")

    blackout = df[TARGET].astype(int)
    class_counts = blackout.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Cannot stratify split: blackout target has only one class")
    if class_counts.min() < 3:
        raise ValueError(
            "Cannot stratify split: each blackout class needs at least 3 samples "
            "for train/val/test"
        )

    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=random_seed,
        shuffle=True,
        stratify=blackout,
    )

    temp_blackout = temp_df[TARGET].astype(int)
    test_frac = 1.0 - train_frac - val_frac
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_frac,
        random_state=random_seed,
        shuffle=True,
        stratify=temp_blackout,
    )

    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def load_splits(input_dir: Path, random_seed: int, stratify_by_blackout: bool = True) -> DatasetSplits:
    """Load generated data and optionally rebuild splits stratified by blackout."""
    train_path = input_dir / "train.csv"
    val_path = input_dir / "val.csv"
    test_path = input_dir / "test.csv"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required split file: {path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if stratify_by_blackout:
        full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
        return stratified_train_val_test_split(full_df, random_seed=random_seed)

    return DatasetSplits(train=train_df, val=val_df, test=test_df)


def resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the training feature columns present in the dataset."""
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(
            "Generated dataset is missing required feature columns: "
            + ", ".join(missing)
        )
    return FEATURES.copy()


def prepare_xy(df: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target vector."""
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column '{TARGET}' in generated dataset")

    X = df[feature_names].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def build_pipeline(estimator, use_scaler: bool) -> Pipeline:
    """Build a preprocessing + model pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def build_candidate_models(random_seed: int) -> Dict[str, Pipeline]:
    """Create candidate probability classifiers."""
    models: Dict[str, Pipeline] = {
        "RandomForestClassifier": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=random_seed,
                n_jobs=-1,
            ),
            use_scaler=False,
        ),
        "GradientBoostingClassifier": build_pipeline(
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.85,
                random_state=random_seed,
            ),
            use_scaler=False,
        ),
        "LogisticRegression": build_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_seed,
            ),
            use_scaler=True,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBClassifier"] = build_pipeline(
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_seed,
                n_jobs=-1,
            ),
            use_scaler=False,
        )

    return models


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute classification metrics from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate a fitted model on a split."""
    y_prob = model.predict_proba(X)[:, 1]
    metrics = probability_metrics(y.to_numpy(), y_prob)
    metrics["confusion_matrix"] = confusion_matrix(y, (y_prob >= 0.5).astype(int)).tolist()
    metrics["classification_report"] = classification_report(
        y,
        (y_prob >= 0.5).astype(int),
        target_names=["Normal", "Blackout"],
        zero_division=0,
    )
    return metrics


def fit_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_seed: int,
) -> Tuple[str, Pipeline, Dict[str, Dict[str, float]]]:
    """Train all candidates and select the best by validation ROC AUC."""
    models = build_candidate_models(random_seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    results: Dict[str, Dict[str, float]] = {}
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, model in models.items():
        fit_kwargs = {"classifier__sample_weight": sample_weight}
        model.fit(X_train, y_train, **fit_kwargs)
        val_metrics = evaluate_model(model, X_val, y_val)
        results[name] = val_metrics

        score = val_metrics["roc_auc"]
        if np.isnan(score):
            score = val_metrics["f1"]

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    if best_name is None or best_model is None:
        raise RuntimeError("No model could be trained successfully")

    return best_name, best_model, results


def refit_best_model(
    best_name: str,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    random_seed: int,
) -> Pipeline:
    """Refit the selected model on train + validation data."""
    candidate_models = build_candidate_models(random_seed)
    model = candidate_models[best_name]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_val)
    model.fit(X_train_val, y_train_val, classifier__sample_weight=sample_weight)
    return model


def plot_confusion_matrix(confusion_matrix_values: List[List[int]], output_path: Path) -> None:
    """Save a labeled confusion matrix plot."""
    cm = np.array(confusion_matrix_values)
    labels = ["Normal", "Blackout"]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    threshold = cm.max() / 2.0 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                f"{cm[row, col]:,}",
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
                fontsize=12,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_artifacts(
    model: Pipeline,
    feature_names: List[str],
    metrics: Dict[str, object],
    output_dir: Path,
    test_df: pd.DataFrame,
) -> None:
    """Save model and diagnostic artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "best_model.pkl")
    joblib.dump(feature_names, output_dir / "feature_names.pkl")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    test_metrics = metrics.get("test", {})
    if isinstance(test_metrics, dict) and "confusion_matrix" in test_metrics:
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            output_dir / "confusion_matrix.png",
        )

    if TARGET in test_df.columns:
        test_predictions = test_df[["scenario_id", "scenario_type", TARGET]].copy()
    else:
        test_predictions = test_df.copy()

    test_predictions["predicted_probability"] = model.predict_proba(test_df[feature_names])[:, 1]
    test_predictions["predicted_blackout"] = (test_predictions["predicted_probability"] >= 0.5).astype(int)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def print_summary(
    split_sizes: Dict[str, int],
    feature_names: List[str],
    validation_results: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, float],
    best_name: str,
) -> None:
    """Print a concise training summary."""
    print("\n" + "=" * 80)
    print("GENERATED DATA ML TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nSplits:")
    print(f"  Train: {split_sizes['train']:,}")
    print(f"  Val  : {split_sizes['val']:,}")
    print(f"  Test : {split_sizes['test']:,}")
    print(f"\nFeatures ({len(feature_names)}): {', '.join(feature_names)}")

    print("\nValidation metrics:")
    for name, metrics in validation_results.items():
        print(
            f"  {name:<25} | ROC AUC={metrics['roc_auc']:.4f} | "
            f"F1={metrics['f1']:.4f} | Brier={metrics['brier']:.4f}"
        )

    print(f"\nBest model: {best_name}")
    print("\nTest metrics:")
    print(f"  ROC AUC          : {test_metrics['roc_auc']:.4f}")
    print(f"  PR AUC           : {test_metrics['pr_auc']:.4f}")
    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision        : {test_metrics['precision']:.4f}")
    print(f"  Recall           : {test_metrics['recall']:.4f}")
    print(f"  F1               : {test_metrics['f1']:.4f}")
    print(f"  Brier            : {test_metrics['brier']:.4f}")
    print("\nConfusion matrix (test):")
    cm = np.array(test_metrics["confusion_matrix"])
    print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}")
    print("\nClassification report (test):")
    print(test_metrics["classification_report"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a blackout classifier on datasets/generated")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing train.csv, val.csv, and test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save the trained model artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for model reproducibility",
    )
    parser.add_argument(
        "--use-existing-splits",
        action="store_true",
        help="Use train.csv/val.csv/test.csv as-is instead of rebuilding stratified splits",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    splits = load_splits(
        input_dir,
        random_seed=args.seed,
        stratify_by_blackout=not args.use_existing_splits,
    )
    feature_names = resolve_feature_columns(splits.train)

    X_train, y_train = prepare_xy(splits.train, feature_names)
    X_val, y_val = prepare_xy(splits.val, feature_names)
    X_test, y_test = prepare_xy(splits.test, feature_names)

    best_name, _best_val_model, validation_results = fit_and_select(
        X_train, y_train, X_val, y_val, args.seed
    )

    X_train_val = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    final_model = refit_best_model(best_name, X_train_val, y_train_val, args.seed)

    test_metrics = evaluate_model(final_model, X_test, y_test)

    metrics = {
        "best_model": best_name,
        "feature_names": feature_names,
        "validation": validation_results,
        "test": test_metrics,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target": TARGET,
    }

    save_artifacts(final_model, feature_names, metrics, output_dir, splits.test)

    print_summary(
        {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        feature_names,
        validation_results,
        test_metrics,
        best_name,
    )

    print(f"\nArtifacts saved to: {output_dir}")
    print("  - best_model.pkl")
    print("  - feature_names.pkl")
    print("  - metrics.json")
    print("  - confusion_matrix.png")
    print("  - test_predictions.csv")


if __name__ == "__main__":
    main()
