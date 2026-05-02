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


# ── Window Extraction & Perturbation Functions ─────────────────────────────────

def extract_solar_windows(
    raw_solar_series: pd.Series,
    n_windows: int = 50,
    window_size_hours: int = 720,
    strategy: str = 'uniform'
) -> list:
    """
    Extract N independent 720-hour windows from raw solar data.

    Uses systematic sampling to ensure non-overlapping windows and maximum
    temporal coverage. Wraps around at end of data.

    Parameters
    ----------
    raw_solar_series : pd.Series
        Full hourly solar series (e.g., 8760 hours = 1 year)
    n_windows : int
        Number of windows to extract (default 50)
    window_size_hours : int
        Length of each window in hours (default 720 = 30 days)
    strategy : str
        'uniform': evenly spaced, 'random': random offsets

    Returns
    -------
    list of pd.Series
        N independent solar windows, each of length window_size_hours
    """
    available_hours = len(raw_solar_series)

    if window_size_hours > available_hours:
        raise ValueError(f"Window size {window_size_hours}h exceeds data length {available_hours}h")

    windows = []

    if strategy == 'uniform':
        # Evenly spaced windows
        step_size = (available_hours - window_size_hours) / max(n_windows - 1, 1)

        for i in range(n_windows):
            start_idx = int(i * step_size)
            end_idx = start_idx + window_size_hours
            window = raw_solar_series.iloc[start_idx:end_idx].reset_index(drop=True)
            windows.append(window)

    else:  # random
        # Randomly offset windows (reproducible with seed)
        for i in range(n_windows):
            start_idx = np.random.randint(0, max(available_hours - window_size_hours + 1, 1))
            end_idx = start_idx + window_size_hours
            window = raw_solar_series.iloc[start_idx:end_idx].reset_index(drop=True)
            windows.append(window)

    return windows


def add_cloud_intermittency(
    series: pd.Series,
    intermittency_factor: float = 0.1,
    seed: int = None
) -> pd.Series:
    """
    Simulate cloud transients with rapid solar fluctuations.

    Creates realistic cloud-induced variability by adding fast oscillations
    during daylight hours only.

    Parameters
    ----------
    series : pd.Series
        Solar series in MW
    intermittency_factor : float
        Strength of intermittency (0 = none, 0.3 = high clouds)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.Series
        Solar series with cloud-induced variability
    """
    if seed is not None:
        np.random.seed(seed)

    if intermittency_factor == 0:
        return series.copy()

    cloudy = series.copy()

    # Only add clouds during active solar hours (> 0.01 MW)
    is_daylight = series > 0.01

    # Add high-frequency noise (cloud transients)
    noise = np.random.normal(0, 1, len(series)) * series * intermittency_factor * 0.5

    cloudy[is_daylight] = cloudy[is_daylight] + noise[is_daylight]

    # Clip to valid range
    cloudy = np.clip(cloudy, 0, PV_CAPACITY_MW)

    return pd.Series(cloudy, index=series.index)


def scale_solar_capacity(
    series: pd.Series,
    scaling_factor: float,
    max_capacity_mw: float = PV_CAPACITY_MW
) -> pd.Series:
    """
    Scale solar generation by constant factor (simulate different PV capacity).

    Parameters
    ----------
    series : pd.Series
        Solar series in MW
    scaling_factor : float
        Multiplier (e.g., 0.5 = half capacity, 1.2 = 20% oversized)
    max_capacity_mw : float
        Maximum allowed capacity after scaling

    Returns
    -------
    pd.Series
        Scaled solar series, clipped to valid range
    """
    scaled = series * scaling_factor
    scaled = np.clip(scaled, 0, max_capacity_mw)
    return pd.Series(scaled, index=series.index)


def inject_solar_dips(
    series: pd.Series,
    dip_count: int = 2,
    dip_magnitude: float = 0.3,
    dip_duration_hours: int = 2,
    seed: int = None
) -> pd.Series:
    """
    Inject artificial solar drops (cloud shadows) into solar series.

    Parameters
    ----------
    series : pd.Series
        Solar series in MW
    dip_count : int
        Number of solar dips to inject
    dip_magnitude : float
        Reduction factor during dip (e.g., 0.3 = drop to 30% of normal)
    dip_duration_hours : int
        Duration of each dip in hours
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.Series
        Solar series with injected dips
    """
    if seed is not None:
        np.random.seed(seed)

    if dip_count == 0:
        return series.copy()

    dipped = series.copy()
    n_hours = len(series)

    for _ in range(dip_count):
        # Random dip start time (prefer daylight hours)
        start_idx = np.random.randint(0, max(n_hours - dip_duration_hours, 1))
        end_idx = min(start_idx + dip_duration_hours, n_hours)

        # Only apply dip if there's actual generation
        if dipped.iloc[start_idx:end_idx].max() > 0.01:
            dipped.iloc[start_idx:end_idx] *= dip_magnitude

    # Clip to valid range
    dipped = np.clip(dipped, 0, PV_CAPACITY_MW)

    return pd.Series(dipped, index=series.index)


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
