"""
digital_twin.py
===============
Main simulation loop for the Digital Twin Microgrid project.

Orchestrates all modules across 720 hourly timesteps (30 days):
  1. Load p_load(t) and p_solar(t) from preprocessed CSVs
  2. Battery dispatch — charge/discharge decision each hour
  3. AC load flow — pandapower IEEE 33-bus network
  4. Label generation — physics-based blackout label
  5. Record all features and labels to simulation_results.csv

Output: datasets/processed/simulation_results.csv
  - 720 rows (one per hour)
  - 16 columns (features + label)
  - Ready for ml_model.py

Imports:
  data_load.py       → load series (not used here — reads from CSV)
  data_solar.py      → solar series (not used here — reads from CSV)
  battery_model.py   → BatteryModel
  grid_simulator.py  → create_network, run_load_flow
  label_generator.py → generate_label, get_severity, get_blackout_margin
"""

import os
import time
import numpy as np
import pandas as pd

from battery_model   import BatteryModel
from grid_simulator  import create_network, run_load_flow
from label_generator import generate_label, get_severity, get_blackout_margin


# ── Config ────────────────────────────────────────────────────────────────────

LOAD_CSV   = 'datasets/processed/load_scaled.csv'
SOLAR_CSV  = 'datasets/processed/solar_scaled.csv'
OUTPUT_CSV = 'datasets/processed/simulation_results.csv'

GRID_IMPORT_LIMIT_MW = 5.0   # max MW the main grid connection can supply


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_inputs() -> tuple:
    """
    Load preprocessed load and solar series from CSV.

    Returns
    -------
    tuple (p_load, p_solar)
        Both are pd.Series of length 720
    """
    load_df  = pd.read_csv(LOAD_CSV)
    solar_df = pd.read_csv(SOLAR_CSV)

    p_load  = load_df['load_mw'].reset_index(drop=True)
    p_solar = solar_df['p_solar_mw'].reset_index(drop=True)

    assert len(p_load) == len(p_solar), \
        f"Length mismatch: load={len(p_load)}, solar={len(p_solar)}"

    print(f"  Load  series : {len(p_load)} hours | "
          f"min={p_load.min():.3f} MW | max={p_load.max():.3f} MW")
    print(f"  Solar series : {len(p_solar)} hours | "
          f"min={p_solar.min():.3f} MW | max={p_solar.max():.3f} MW")

    return p_load, p_solar


def print_progress(t: int, total: int, record: dict, interval: int = 72) -> None:
    """Print simulation progress every `interval` timesteps."""
    if t % interval == 0 or t == total - 1:
        pct = (t + 1) / total * 100
        print(
            f"  t={t+1:>4}/{total} ({pct:5.1f}%) | "
            f"Load={record['p_load_mw']:.3f} MW | "
            f"Solar={record['p_solar_mw']:.3f} MW | "
            f"SOC={record['soc']:.3f} | "
            f"Vmin={record['v_min']:.4f} | "
            f"{'⚡ BLACKOUT' if record['blackout'] else '  normal  '}"
        )


# ── Main Simulation Loop ──────────────────────────────────────────────────────

def run_simulation() -> pd.DataFrame:
    """
    Run the full 30-day Digital Twin simulation.

    For each hourly timestep t:
      1. Read p_load(t) and p_solar(t)
      2. Battery step → p_battery(t), soc(t)
      3. Compute p_grid(t) = p_load - p_solar - p_battery
      4. Run AC load flow → v_min, v_max, v_mean, line_loading, p_loss
      5. Generate blackout label and severity
      6. Record all values

    Returns
    -------
    pd.DataFrame
        720-row simulation results with all features and labels
    """

    print("\n" + "=" * 60)
    print("  Digital Twin — Microgrid Simulation")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────
    print("\n[1/4] Loading input series...")
    p_load_series, p_solar_series = load_inputs()
    T = len(p_load_series)

    # ── Initialise components ─────────────────────────────────────
    print("\n[2/4] Initialising battery and IEEE 33-bus network...")
    battery = BatteryModel(
        capacity_mwh   = 0.5,
        soc_init       = 0.5,
        soc_min        = 0.1,
        soc_max        = 0.9,
        charge_rate    = 0.25,
        discharge_rate = 0.25,
        efficiency     = 0.95,
    )
    battery.summary()

    net = create_network()
    print(f"\n  Network : {len(net.bus)} buses | "
          f"{len(net.load)} loads | "
          f"{len(net.line)} lines")

    # ── Simulation loop ───────────────────────────────────────────
    print(f"\n[3/4] Running simulation ({T} timesteps)...\n")
    t_start = time.time()

    records = []

    for t in range(T):
        p_load  = float(p_load_series.iloc[t])
        p_solar = float(p_solar_series.iloc[t])

        # Battery dispatch
        p_battery, soc = battery.step(p_load, p_solar)

        # Grid residual — what the main grid must supply
        p_grid = p_load - p_solar - p_battery

        # AC load flow
        lf = run_load_flow(net, p_load, p_solar)

        # Add realistic voltage measurement noise to break deterministic relationships
        v_min_measured = lf['v_min'] + np.random.normal(0, 0.007)  # ±0.7% sensor error
        
        # Label generation (uses noisy measurement, not ideal voltage)
        label    = generate_label(v_min_measured, lf['converged'])
        severity = get_severity(lf['v_min'], lf['converged'])  # severity uses ideal voltage for reference
        margin   = get_blackout_margin(lf['v_min'])

        # Derived time features
        hour_of_day = t % 24
        day         = t // 24
        is_night    = int(hour_of_day < 6 or hour_of_day >= 20)

        record = {
            # ── Time ──────────────────────────────────────────────
            'timestep':         t,
            'day':              day,
            'hour_of_day':      hour_of_day,
            'is_night':         is_night,
            # ── Power balance ──────────────────────────────────────
            'p_load_mw':        p_load,
            'p_solar_mw':       p_solar,
            'p_battery_mw':     p_battery,
            'p_grid_mw':        p_grid,
            # ── Battery ────────────────────────────────────────────
            'soc':              soc,
            # ── Grid (from load flow) ──────────────────────────────
            'v_min':            lf['v_min'],
            'v_min_measured':   v_min_measured,  # with measurement noise
            'v_max':            lf['v_max'],
            'v_mean':           lf['v_mean'],
            'line_loading_max': lf['line_loading_max'],
            'p_loss_mw':        lf['p_loss_mw'],
            'converged':        int(lf['converged']),
            # ── Labels ─────────────────────────────────────────────
            'v_margin':         margin,
            'severity':         severity,
            'blackout':         label,
        }

        records.append(record)
        print_progress(t, T, record, interval=72)
        
    # ── Build DataFrame ───────────────────────────────────────────
    df = pd.DataFrame(records)

    elapsed = time.time() - t_start
    print(f"\n  Simulation complete in {elapsed:.1f}s")

    return df


# ── Summary & Save ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print simulation result summary."""

    blackout_hours = df['blackout'].sum()
    normal_hours   = len(df) - blackout_hours
    blackout_rate  = blackout_hours / len(df) * 100

    print("\n" + "=" * 60)
    print("  Simulation Summary")
    print("=" * 60)
    print(f"\n  Total hours    : {len(df)}")
    print(f"  Blackout hours : {blackout_hours}  ({blackout_rate:.1f}%)")
    print(f"  Normal hours   : {normal_hours}  ({100-blackout_rate:.1f}%)")

    print(f"\n  Power balance:")
    print(f"    Load  mean   : {df['p_load_mw'].mean():.4f} MW")
    print(f"    Solar mean   : {df['p_solar_mw'].mean():.4f} MW")
    print(f"    Battery mean : {df['p_battery_mw'].mean():.4f} MW")
    print(f"    Grid  mean   : {df['p_grid_mw'].mean():.4f} MW")

    print(f"\n  Voltage profile:")
    print(f"    Vmin overall : {df['v_min'].min():.4f} pu")
    print(f"    Vmax overall : {df['v_max'].max():.4f} pu")
    print(f"    Vmean avg    : {df['v_mean'].mean():.4f} pu")

    print(f"\n  Severity breakdown:")
    for level in ['CRITICAL', 'HIGH', 'MODERATE', 'NORMAL']:
        count = (df['severity'] == level).sum()
        pct   = count / len(df) * 100
        print(f"    {level:<10} : {count:>4} hrs  ({pct:5.1f}%)")

    print(f"\n  Battery:")
    print(f"    SOC mean     : {df['soc'].mean():.4f}")
    print(f"    SOC min      : {df['soc'].min():.4f}")
    print(f"    SOC max      : {df['soc'].max():.4f}")

    print(f"\n  ML dataset:")
    print(f"    Rows         : {len(df)}")
    print(f"    Columns      : {len(df.columns)}")
    print(f"    Features     : {len(df.columns) - 3}  (excl. timestep, severity, blackout)")
    print(f"    Label        : blackout  (0/1)")


def save_results(df: pd.DataFrame) -> None:
    """Save simulation results to CSV."""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[4/4] Saved → {OUTPUT_CSV}")
    print(f"      Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string(index=False))


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    df = run_simulation()
    print_summary(df)
    save_results(df)

    print("\n" + "=" * 60)
    print("  Simulation complete. Ready for ml_model.py")
    print("=" * 60)