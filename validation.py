"""
validation.py
=============
Comprehensive validation framework for data quality, bounds checking,
and leakage prevention.
"""

import numpy as np
import pandas as pd
from config import (
    MIN_LOAD_MW, MAX_LOAD_MW,
    MIN_SOLAR_MW, MAX_SOLAR_MW,
    MIN_SOC, MAX_SOC,
    MIN_VOLTAGE_PU, MAX_VOLTAGE_PU,
    BLACKOUT_THRESHOLD, V_MIN_LIMIT, V_MAX_LIMIT
)


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_load_series(series: pd.Series, name: str = "load_series") -> None:
    """
    Validate hourly load time series.

    Parameters
    ----------
    series : pd.Series
        Load values in MW
    name : str
        Series name for error messages

    Raises
    ------
    ValidationError
        If series fails any check
    """
    # ── Type check ─────────────────────────────────────────────────────────
    if not isinstance(series, pd.Series):
        raise ValidationError(f"{name}: Must be pd.Series, got {type(series)}")

    # ── NaN check ──────────────────────────────────────────────────────────
    if series.isna().sum() > 0:
        raise ValidationError(f"{name}: Contains {series.isna().sum()} NaN values")

    # ── Length check ───────────────────────────────────────────────────────
    if len(series) < 720:
        raise ValidationError(f"{name}: Length {len(series)} < 720 hours minimum")

    # ── Bounds check ───────────────────────────────────────────────────────
    if series.min() < MIN_LOAD_MW:
        raise ValidationError(f"{name}: Min {series.min():.3f} < {MIN_LOAD_MW} MW")
    if series.max() > MAX_LOAD_MW:
        raise ValidationError(f"{name}: Max {series.max():.3f} > {MAX_LOAD_MW} MW")

    # ── Monotonicity check (should not have huge jumps) ────────────────────
    max_jump = series.diff().abs().max()
    if max_jump > 0.5:  # More than 500 kW jump in 1 hour
        raise ValidationError(f"{name}: Unrealistic jump of {max_jump:.3f} MW detected")

    print(f"✓ {name} validation PASSED | min={series.min():.3f} MW, max={series.max():.3f} MW")


def validate_solar_series(series: pd.Series, name: str = "solar_series") -> None:
    """Validate solar generation time series."""
    if not isinstance(series, pd.Series):
        raise ValidationError(f"{name}: Must be pd.Series")
    if series.isna().sum() > 0:
        raise ValidationError(f"{name}: Contains NaN values")
    if len(series) < 720:
        raise ValidationError(f"{name}: Length {len(series)} < 720")
    if series.min() < MIN_SOLAR_MW:
        raise ValidationError(f"{name}: Min {series.min():.3f} < {MIN_SOLAR_MW} MW")
    if series.max() > MAX_SOLAR_MW:
        raise ValidationError(f"{name}: Max {series.max():.3f} > {MAX_SOLAR_MW} MW")

    print(f"✓ {name} validation PASSED | min={series.min():.3f} MW, max={series.max():.3f} MW")


def validate_simulation_results(df: pd.DataFrame) -> None:
    """
    Validate simulation results dataframe for data quality and leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results with features and labels

    Raises
    ------
    ValidationError
        If data fails validation
    """
    # ── Required columns check ────────────────────────────────────────────
    required_cols = ['p_load_mw', 'p_solar_mw', 'soc', 'v_min', 'v_min_measured',
                     'line_loading_max', 'p_grid_mw', 'blackout', 'risk_score']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValidationError(f"Missing columns: {missing}")

    # ── NaN check ──────────────────────────────────────────────────────────
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"⚠ Warning: {nan_counts.sum()} NaN values found:")
        print(nan_counts[nan_counts > 0])

    # ── Load bounds ────────────────────────────────────────────────────────
    if df['p_load_mw'].min() < MIN_LOAD_MW or df['p_load_mw'].max() > MAX_LOAD_MW:
        raise ValidationError(f"Load out of bounds: [{df['p_load_mw'].min():.3f}, {df['p_load_mw'].max():.3f}]")

    # ── Solar bounds ───────────────────────────────────────────────────────
    if df['p_solar_mw'].min() < MIN_SOLAR_MW or df['p_solar_mw'].max() > MAX_SOLAR_MW:
        raise ValidationError(f"Solar out of bounds: [{df['p_solar_mw'].min():.3f}, {df['p_solar_mw'].max():.3f}]")

    # ── SOC bounds ────────────────────────────────────────────────────────
    if df['soc'].min() < MIN_SOC or df['soc'].max() > MAX_SOC:
        raise ValidationError(f"SOC out of bounds: [{df['soc'].min():.3f}, {df['soc'].max():.3f}]")

    # ── Voltage bounds ────────────────────────────────────────────────────
    if df['v_min'].min() < MIN_VOLTAGE_PU:
        raise ValidationError(f"v_min {df['v_min'].min():.4f} < {MIN_VOLTAGE_PU}")
    if df['v_min'].max() > MAX_VOLTAGE_PU:
        raise ValidationError(f"v_min {df['v_min'].max():.4f} > {MAX_VOLTAGE_PU}")

    # ── Risk score bounds ──────────────────────────────────────────────────
    if df['risk_score'].min() < 0.0 or df['risk_score'].max() > 1.0:
        raise ValidationError(f"risk_score out of [0,1]: [{df['risk_score'].min():.3f}, {df['risk_score'].max():.3f}]")

    # ── Blackout label check ───────────────────────────────────────────────
    blackout_count = (df['blackout'] == 1).sum()
    blackout_rate = blackout_count / len(df) * 100
    if blackout_rate < 5 or blackout_rate > 50:
        print(f"⚠ Warning: Unusual blackout rate {blackout_rate:.1f}% (expected 10-35%)")

    # ── Measurement noise check (v_min vs v_min_measured) ──────────────────
    noise = (df['v_min_measured'] - df['v_min']).abs()
    max_noise = noise.max()
    if max_noise > 0.05:  # More than 5% error seems wrong
        print(f"⚠ Warning: Max measurement noise {max_noise:.4f} pu (expected < 0.02)")

    # ── Data leakage check: correlation of v_min with label ───────────────
    corr_v_min_label = df['v_min'].corr(df['blackout'])
    if corr_v_min_label > 0.95:
        print(f"⚠ WARNING: v_min correlation with blackout {corr_v_min_label:.3f} too high!")
        print(f"   This indicates DIRECT LEAKAGE — label depends entirely on v_min")
        print(f"   This is expected only if v_min is a feature (it shouldn't be)")

    print(f"✓ Simulation results validation PASSED")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Blackout rate: {blackout_rate:.1f}%")
    print(f"  v_min range: [{df['v_min'].min():.4f}, {df['v_min'].max():.4f}] pu")


def validate_ml_features(df: pd.DataFrame, feature_names: list) -> None:
    """
    Validate ML feature matrix for consistency and leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe
    feature_names : list
        Expected feature names

    Raises
    ------
    ValidationError
        If features don't match or contain leakage
    """
    # ── Feature count check ────────────────────────────────────────────────
    if len(df.columns) != len(feature_names):
        raise ValidationError(f"Feature count mismatch: {len(df.columns)} != {len(feature_names)}")

    # ── Feature names check ────────────────────────────────────────────────
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        raise ValidationError(f"Missing features: {missing_features}")

    extra_features = set(df.columns) - set(feature_names)
    if extra_features:
        raise ValidationError(f"Extra features (leakage risk): {extra_features}")

    # ── Leakage check: No v_min, hour_of_day, etc. ────────────────────────
    leakage_features = ['v_min', 'v_max', 'v_mean', 'hour_of_day', 'day', 'is_night']
    found_leakage = set(df.columns) & set(leakage_features)
    if found_leakage:
        raise ValidationError(f"LEAKAGE FEATURES DETECTED: {found_leakage}")

    # ── NaN check ──────────────────────────────────────────────────────────
    if df.isna().sum().sum() > 0:
        raise ValidationError(f"Features contain {df.isna().sum().sum()} NaN values")

    # ── Scale check: Features should be roughly [-2, 2] after StandardScaler
    feature_max = df.abs().max().max()
    if feature_max > 10:
        print(f"⚠ Warning: Feature scaling appears off (max abs value {feature_max:.1f})")

    print(f"✓ ML features validation PASSED | {len(feature_names)} features, {len(df)} samples")


def validate_train_test_split(X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Validate train/test split for temporal ordering and no data leakage.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test labels

    Raises
    ------
    ValidationError
        If split is invalid
    """
    # ── Size check ────────────────────────────────────────────────────────
    train_size = len(X_train)
    test_size = len(X_test)
    train_pct = train_size / (train_size + test_size) * 100

    if train_pct < 70 or train_pct > 90:
        raise ValidationError(f"Train/test split {train_pct:.1f}% unusual (expect 70-90%)")

    # ── Shape matching check ───────────────────────────────────────────────
    if X_train.shape[0] != len(y_train):
        raise ValidationError(f"X_train/y_train size mismatch: {X_train.shape[0]} != {len(y_train)}")
    if X_test.shape[0] != len(y_test):
        raise ValidationError(f"X_test/y_test size mismatch: {X_test.shape[0]} != {len(y_test)}")

    # ── Class balance check (for classification) ───────────────────────────
    if hasattr(y_train, 'unique') and len(y_train.unique()) <= 2:
        train_pos_rate = (y_train == 1).sum() / len(y_train) * 100
        test_pos_rate = (y_test == 1).sum() / len(y_test) * 100
        print(f"✓ Train/test split validation PASSED")
        print(f"  Train size: {train_size} ({train_pct:.1f}%), positive rate: {train_pos_rate:.1f}%")
        print(f"  Test size: {test_size} ({100-train_pct:.1f}%), positive rate: {test_pos_rate:.1f}%")
    else:
        print(f"✓ Train/test split validation PASSED | Train: {train_size}, Test: {test_size}")


if __name__ == '__main__':
    print("Validation module ready. Import and use validators in your pipeline.")
