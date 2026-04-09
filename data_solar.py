"""
data_solar.py
=============
Loads and preprocesses Kaggle Solar Power Generation dataset (Plant 1).

Dataset specifics:
  - 22 inverters (SOURCE_KEY), each with its own DC_POWER row per timestamp
  - 15-minute resolution, DATE_TIME format: DD-MM-YYYY HH:MM
  - DC_POWER values are in Watts (W)

Processing steps:
  - Sum DC_POWER across all 22 inverters per timestamp → total plant output
  - Resample 15-min → hourly (mean)
  - Fill NaN with 0.0 (night hours have no generation)
  - W → MW  (÷ 1,000,000)
  - Normalize to [0, 1]
  - Scale to PV capacity

Output:
  - Hourly p_solar(t) in MW, simulation-ready
"""

import pandas as pd
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

PV_CAPACITY_MW = 3.0    # Installed PV capacity assumed for the microgrid // before 1.5 now changed to 3.0 daytime surplus becomes meaningful


# ── Solar Generation ──────────────────────────────────────────────────────────

def load_solar_generation(path: str) -> pd.Series:
    """
    Load Kaggle Solar Power Generation dataset (Plant 1).

    Steps:
      - Parse DATE_TIME (DD-MM-YYYY HH:MM format)
      - Sum DC_POWER across all 22 inverters per timestamp
      - Resample 15-min → hourly (mean)
      - Fill NaN with 0.0 (night hours)
      - W → MW

    Parameters
    ----------
    path : str
        Path to Plant_1_Generation_Data.csv

    Returns
    -------
    pd.Series
        Hourly total plant solar generation in MW, with DatetimeIndex
    """
    df = pd.read_csv(path)

    # Parse DATE_TIME — format is DD-MM-YYYY HH:MM
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True)
    df = df.sort_values('DATE_TIME').reset_index(drop=True)

    # Sum DC_POWER across all inverters per timestamp
    total_w = df.groupby('DATE_TIME')['DC_POWER'].sum()

    # Resample 15-min → hourly (mean over 4 readings)
    hourly_w = total_w.resample('h').mean()

    # Night hours produce NaN after resample boundary → fill with 0
    hourly_w = hourly_w.fillna(0.0)

    # W → MW
    hourly_mw = hourly_w / 1_000_000

    return hourly_mw


# ── Normalization & Scaling ───────────────────────────────────────────────────

def normalize_by_max(series: pd.Series) -> pd.Series:
    """
    Max-scale a series to [0, 1].
    Preserves the shape of the generation profile.

    Parameters
    ----------
    series : pd.Series
        Raw solar generation in MW

    Returns
    -------
    pd.Series
        Normalized solar generation in range [0, 1]
    """
    max_val = series.max()
    if max_val == 0:
        raise ValueError("Series maximum is 0 — cannot normalize.")
    return series / max_val


def scale_to_pv_capacity(solar_norm: pd.Series,
                          pv_capacity_mw: float = PV_CAPACITY_MW) -> pd.Series:
    """
    Scale a normalized [0, 1] series to installed PV capacity.

    Parameters
    ----------
    solar_norm     : pd.Series
        Normalized solar generation in [0, 1]
    pv_capacity_mw : float
        Installed PV capacity in MW (default 1.5)

    Returns
    -------
    pd.Series
        Solar generation scaled to MW range [0, pv_capacity_mw]
    """
    return solar_norm * pv_capacity_mw


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def prepare_solar_series(
    path: str,
    n_days: int = 30,
    pv_capacity_mw: float = PV_CAPACITY_MW,
) -> pd.Series:
    """
    Full pipeline: raw CSV → simulation-ready hourly solar generation in MW.

    Steps:
        1. Load and aggregate raw plant data
        2. Trim to simulation horizon
        3. Normalize to [0, 1]
        4. Scale to PV capacity

    Parameters
    ----------
    path           : str
        Path to Plant_1_Generation_Data.csv
    n_days         : int
        Simulation horizon in days (default 30 — dataset covers ~34 days)
    pv_capacity_mw : float
        Installed PV capacity in MW (default 1.5)

    Returns
    -------
    pd.Series
        Integer-indexed hourly solar generation in MW, length = n_days * 24
    """
    raw    = load_solar_generation(path)
    trimmed = raw.iloc[: n_days * 24]
    norm   = normalize_by_max(trimmed)
    scaled = scale_to_pv_capacity(norm, pv_capacity_mw)

    series = scaled.reset_index(drop=True)
    series.name = 'p_solar_mw'
    return series


# ── Validation ────────────────────────────────────────────────────────────────

def validate_solar_series(series: pd.Series,
                           pv_capacity_mw: float = PV_CAPACITY_MW) -> None:
    """
    Sanity checks on the prepared solar series.
    Prints a summary if all checks pass.
    Raises AssertionError with a clear message if any check fails.

    Parameters
    ----------
    series         : pd.Series
        Output of prepare_solar_series()
    pv_capacity_mw : float
        Installed PV capacity in MW
    """
    assert series.isna().sum() == 0, \
        f"Solar series contains {series.isna().sum()} NaN values."
    assert series.min() >= 0.0, \
        f"Solar series contains negative values (min={series.min():.6f})."
    assert series.max() <= pv_capacity_mw + 1e-6, \
        f"Solar exceeds PV capacity: max={series.max():.4f} > {pv_capacity_mw} MW."

    night_hours  = (series == 0.0).sum()
    active_hours = (series > 0.0).sum()

    print(
        f"[VALID] Solar series OK\n"
        f"        Length       : {len(series)} hours\n"
        f"        Min          : {series.min():.6f} MW\n"
        f"        Max          : {series.max():.4f} MW\n"
        f"        Mean (all)   : {series.mean():.4f} MW\n"
        f"        Mean (active): {series[series > 0].mean():.4f} MW\n"
        f"        Night hours  : {night_hours}  ({night_hours/len(series)*100:.1f}%)\n"
        f"        Active hours : {active_hours}  ({active_hours/len(series)*100:.1f}%)"
    )


# ── Save ──────────────────────────────────────────────────────────────────────

def save_solar(series: pd.Series, out_path: str) -> None:
    """
    Save solar series to CSV.
    Column: p_solar_mw
    """
    series.to_csv(out_path, index=False, header=['p_solar_mw'])
    print(f"[SAVED] Solar data → {out_path}  (shape: {len(series)} rows x 1 column)")


# ── Entry point (quick test) ──────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    path   = sys.argv[1] if len(sys.argv) > 1 else \
             'datasets/Solar Power Generation Data/Plant_1_Generation_Data.csv'
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"Loading solar data from: {path}")
    solar = prepare_solar_series(path, n_days=n_days)

    validate_solar_series(solar)
    save_solar(solar, out_path='datasets/processed/solar_scaled.csv')

    print(f"\nFirst 5 values:\n{solar.head()}")
