"""
feature_engineering.py
======================
Advanced temporal feature engineering beyond simple lags.
"""

import pandas as pd
import numpy as np
from config import LAG_PERIODS


def add_rolling_statistics(df: pd.DataFrame, windows: list = [3, 6, 12],
                          columns: list = ['p_load_mw', 'p_solar_mw']) -> pd.DataFrame:
    """
    Add rolling mean/std/min/max features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series
    windows : list
        Window sizes (hours) for rolling statistics
    columns : list
        Columns to compute statistics on

    Returns
    -------
    pd.DataFrame
        Dataframe with added rolling features
    """
    df_copy = df.copy()

    for col in columns:
        for window in windows:
            df_copy[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
            df_copy[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
            df_copy[f'{col}_rolling_min_{window}h'] = df[col].rolling(window=window, min_periods=1).min()
            df_copy[f'{col}_rolling_max_{window}h'] = df[col].rolling(window=window, min_periods=1).max()

    # Fill NaN from rolling at start
    df_copy = df_copy.fillna(method='bfill')

    return df_copy


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week one-hot encoding."""
    df_copy = df.copy()

    # Assuming 30-day simulation starting on day 0
    # Day 0 = Monday (0), Day 1 = Tuesday (1), ... Day 6 = Sunday (6)
    df_copy['day_of_week'] = df_copy['day'] % 7
    df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)

    # One-hot encode day of week
    for day in range(7):
        df_copy[f'day_of_week_{day}'] = (df_copy['day_of_week'] == day).astype(int)

    return df_copy


def add_hour_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour-of-day as cyclical features (sin/cos).

    Prevents artificial distance between hour 23 and hour 0.
    """
    df_copy = df.copy()

    # Convert hour to angle (0-24 hours → 0-2π radians)
    hour_angle = 2 * np.pi * df_copy['hour_of_day'] / 24

    df_copy['hour_sin'] = np.sin(hour_angle)
    df_copy['hour_cos'] = np.cos(hour_angle)

    return df_copy


def add_power_balance_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived statistics from power balance features.

    Examples:
    - surplus: p_solar - p_load
    - grid_stress: |p_grid| / grid_limit
    - battery_stress: p_battery / max_rate
    """
    df_copy = df.copy()

    # Power surplus (negative = deficit)
    df_copy['power_surplus'] = df_copy['p_solar_mw'] - df_copy['p_load_mw']

    # Grid stress (normalized by 5 MW limit)
    df_copy['grid_stress'] = np.abs(df_copy['p_grid_mw']) / 5.0
    df_copy['grid_stress'] = np.clip(df_copy['grid_stress'], 0, 1)

    # Battery stress (normalized by 0.25 MW max rate)
    df_copy['battery_stress'] = np.abs(df_copy['p_battery_mw']) / 0.25
    df_copy['battery_stress'] = np.clip(df_copy['battery_stress'], 0, 1)

    # SOC deviation from target (0.6)
    df_copy['soc_deviation'] = np.abs(df_copy['soc'] - 0.6)

    return df_copy


def add_change_rates(df: pd.DataFrame, columns: list = ['p_load_mw', 'p_solar_mw', 'soc'],
                    periods: list = [1, 3, 6]) -> pd.DataFrame:
    """
    Add rate-of-change features (first differences).

    Captures dynamics and trend.
    """
    df_copy = df.copy()

    for col in columns:
        for period in periods:
            df_copy[f'{col}_change_{period}h'] = df_copy[col].diff(periods=period)

    # Fill NaN
    df_copy = df_copy.fillna(method='bfill')

    return df_copy


def create_enhanced_features(df: pd.DataFrame, keep_original_time_features: bool = True) -> pd.DataFrame:
    """
    Create comprehensive feature set from simulation results.

    Parameters
    ----------
    df : pd.DataFrame
        Raw simulation results
    keep_original_time_features : bool
        If True, keep hour_of_day, day, is_night (not recommended for ML)

    Returns
    -------
    pd.DataFrame
        Enhanced feature matrix
    """
    print("\n" + "="*60)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*60)

    df_features = df.copy()

    # 1. Rolling statistics (3h, 6h, 12h windows)
    print("\n1. Adding rolling statistics...")
    df_features = add_rolling_statistics(df_features)
    print(f"   ✓ Added {len(df_features.columns) - len(df.columns)} rolling features")

    # 2. Day-of-week encoding
    print("2. Adding day-of-week features...")
    df_features = add_day_of_week(df_features)
    print(f"   ✓ Added day_of_week, is_weekend, 7 one-hot encoded days")

    # 3. Cyclical hour encoding
    print("3. Adding cyclical hour encoding...")
    df_features = add_hour_cyclical_encoding(df_features)
    print(f"   ✓ Added hour_sin, hour_cos (cyclical representation)")

    # 4. Power balance statistics
    print("4. Adding power balance statistics...")
    df_features = add_power_balance_statistics(df_features)
    print(f"   ✓ Added surplus, grid_stress, battery_stress, soc_deviation")

    # 5. Change rates
    print("5. Adding rate-of-change features...")
    df_features = add_change_rates(df_features)
    print(f"   ✓ Added change rates for load, solar, SOC")

    # Remove problematic temporal features if not needed
    if not keep_original_time_features:
        cols_to_drop = ['hour_of_day', 'day', 'is_night']
        df_features = df_features.drop(columns=cols_to_drop, errors='ignore')
        print(f"\n6. Removed direct temporal features: {cols_to_drop}")

    print(f"\n✓ Feature engineering complete")
    print(f"  Original shape: {df.shape}")
    print(f"  Enhanced shape: {df_features.shape}")
    print(f"  New features added: {df_features.shape[1] - df.shape[1]}")

    return df_features


if __name__ == '__main__':
    print("Feature engineering module ready.")
    print("\nUsage:")
    print("  df_enhanced = create_enhanced_features(df_raw)")
    print("\nOr individual functions:")
    print("  df = add_rolling_statistics(df)")
    print("  df = add_day_of_week(df)")
    print("  df = add_hour_cyclical_encoding(df)")
