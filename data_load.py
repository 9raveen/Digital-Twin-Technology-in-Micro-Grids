"""
Loads and preprocesses UCI Electricity Load Diagrams (370 clients).
Outputs hourly load in MW, normalized and scaled to IEEE 33-bus base load.
"""

import numpy as np
import pandas as pd


# IEEE 33-bus total feeder base load
IEEE33_BASE_LOAD_MW = 3.715


def load_electricity_370(path: str) -> pd.Series:
    """
    Read UCI Electricity Load Diagrams dataset.

    - Fixes '?' missing value markers
    - Forward/backward fills gaps
    - Aggregates 370 clients from kW to MW
    - Resamples from 15-min to hourly resolution

    Parameters
    ----------
    path : str
        Path to LD2011_2014.txt

    Returns
    -------
    pd.Series
        Hourly total load in MW with DatetimeIndex
    """
    df = pd.read_csv(
        path,
        sep=';',
        decimal=',',
        index_col=0,
        parse_dates=True
    )

    # Chronological order (required for resampling)
    df = df.sort_index()

    # UCI uses '?' for missing — replace before any numeric operation
    df = df.replace('?', np.nan)
    df = df.astype(float)

    # Fill gaps: forward-fill first, then backward-fill for leading NaNs
    df = df.ffill().bfill()

    # Sum across all 370 clients (axis=1) → total load in kW
    total_kw = df.sum(axis=1)

    # kW → MW
    total_mw = total_kw / 1000.0

    # Downsample 15-min → hourly by averaging 4 readings per hour
    hourly_mw = total_mw.resample('h').mean()

    # Drop any boundary NaNs introduced by resampling
    hourly_mw = hourly_mw.dropna()

    return hourly_mw


def normalize_by_max(series: pd.Series) -> pd.Series:
    """
    Max-scale series to [0, 1].
    Preserves the shape of the load profile.

    Parameters
    ----------
    series : pd.Series
        Raw load in MW

    Returns
    -------
    pd.Series
        Normalized load in range [0, 1]
    """
    return series / series.max()


def scale_to_ieee_base(load_norm: pd.Series, base_mw: float = IEEE33_BASE_LOAD_MW) -> pd.Series:
    """
    Scale normalized [0, 1] load to IEEE 33-bus feeder capacity.

    Parameters
    ----------
    load_norm : pd.Series
        Normalized load in [0, 1]
    base_mw : float
        IEEE 33-bus base load in MW (default 3.715)

    Returns
    -------
    pd.Series
        Load scaled to MW range [0, base_mw]
    """
    return load_norm * base_mw


def prepare_load_series(path: str, n_days: int = 30) -> pd.Series:
    """
    Full pipeline: raw CSV → simulation-ready hourly load in MW.

    Steps:
        1. Load and clean raw UCI data
        2. Trim to simulation horizon
        3. Normalize to [0, 1]
        4. Scale to IEEE 33-bus base load

    Parameters
    ----------
    path   : str
        Path to LD2011_2014.txt
    n_days : int
        Simulation horizon in days (default 60)

    Returns
    -------
    pd.Series
        Integer-indexed hourly load in MW, length = n_days * 24
    """
    raw    = load_electricity_370(path)
    trimmed = raw.iloc[: n_days * 24]
    norm   = normalize_by_max(trimmed)
    scaled = scale_to_ieee_base(norm)

    return scaled.reset_index(drop=True)


def validate_load_series(series: pd.Series) -> None:
    """
    Sanity checks on the prepared load series.
    Prints a summary if all checks pass.
    Raises AssertionError with a clear message if any check fails.

    Parameters
    ----------
    series : pd.Series
        Output of prepare_load_series()
    """
    assert series.isna().sum() == 0, \
        f"Load series contains {series.isna().sum()} NaN values."
    assert series.min() >= 0.0, \
        f"Load series contains negative values (min={series.min():.4f})."
    assert series.max() <= IEEE33_BASE_LOAD_MW + 1e-6, \
        f"Load exceeds IEEE 33-bus capacity: max={series.max():.4f} MW."

    print(
        f"[VALID] Load series OK\n"
        f"        Length : {len(series)} hours\n"
        f"        Min    : {series.min():.4f} MW\n"
        f"        Max    : {series.max():.4f} MW\n"
        f"        Mean   : {series.mean():.4f} MW"
    )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    path   = sys.argv[1] if len(sys.argv) > 1 else r'datasets\electricityloaddiagrams20112014\LD2011_2014.txt'
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"Loading data from: {path}")
    load = prepare_load_series(path, n_days=60)

    validate_load_series(load)

    print(f"\nFirst 5 values:\n{load.head()}")

