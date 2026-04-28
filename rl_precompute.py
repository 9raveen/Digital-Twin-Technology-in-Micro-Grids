"""
rl_precompute.py
================
Pre-computes pandapower AC load flow results for a discretised grid of
(net_load, p_solar) combinations and saves them to a lookup table CSV.

Why this exists
---------------
Running pandapower inside the RL training loop (~30ms per call) makes
training take hours. By pre-computing all possible operating points once
and looking them up during training (~0.001ms per lookup), training
completes in minutes.

Approach
--------
  - Discretise net_load into N_LOAD bins  spanning [MIN_LOAD, MAX_LOAD]
  - Discretise p_solar  into N_SOLAR bins spanning [0, MAX_SOLAR]
  - Run one pandapower load flow per (load, solar) combination
  - Save results to datasets/processed/lf_lookup.csv

Lookup table columns:
  net_load_mw, p_solar_mw,
  v_min, v_max, v_mean, line_loading_max, p_loss_mw, converged

Usage
-----
    python rl_precompute.py          # runs ~400 load flows, saves CSV
"""

import os
import time
import numpy as np
import pandas as pd

from grid_simulator import create_network, run_load_flow


# ── Config ────────────────────────────────────────────────────────────────────

N_LOAD       = 25      # load bins  (25 × 17 = 425 total load flows)
N_SOLAR      = 17      # solar bins
MIN_LOAD     = 0.5     # MW  — minimum net load
MAX_LOAD     = 4.0     # MW  — slightly above IEEE 33-bus base (3.715)
MIN_SOLAR    = 0.0     # MW
MAX_SOLAR    = 3.0     # MW  — project PV capacity
OUTPUT_PATH  = 'datasets/processed/lf_lookup.csv'


# ── Main ──────────────────────────────────────────────────────────────────────

def precompute_lookup_table() -> pd.DataFrame:
    """
    Run pandapower load flows for all (net_load, p_solar) combinations
    and return a DataFrame lookup table.
    """
    print("=" * 55)
    print("  RL Pre-computation — Load Flow Lookup Table")
    print("=" * 55)

    load_bins  = np.linspace(MIN_LOAD,  MAX_LOAD,  N_LOAD)
    solar_bins = np.linspace(MIN_SOLAR, MAX_SOLAR, N_SOLAR)
    total      = N_LOAD * N_SOLAR

    print(f"\n  Load  bins : {N_LOAD}  ({MIN_LOAD:.1f} → {MAX_LOAD:.1f} MW)")
    print(f"  Solar bins : {N_SOLAR}  ({MIN_SOLAR:.1f} → {MAX_SOLAR:.1f} MW)")
    print(f"  Total runs : {total} load flows\n")

    net = create_network()
    records = []
    t_start = time.time()

    for i, net_load in enumerate(load_bins):
        for j, p_solar in enumerate(solar_bins):
            lf = run_load_flow(net, float(net_load), float(p_solar))
            records.append({
                'net_load_mw':      round(float(net_load), 6),
                'p_solar_mw':       round(float(p_solar),  6),
                'v_min':            round(lf['v_min'],            6),
                'v_max':            round(lf['v_max'],            6),
                'v_mean':           round(lf['v_mean'],           6),
                'line_loading_max': round(lf['line_loading_max'], 6),
                'p_loss_mw':        round(lf['p_loss_mw'],        6),
                'converged':        int(lf['converged']),
            })

        # Progress every 5 load levels
        if (i + 1) % 5 == 0 or i == 0:
            done = (i + 1) * N_SOLAR
            pct  = done / total * 100
            elapsed = time.time() - t_start
            eta     = elapsed / done * (total - done) if done > 0 else 0
            print(f"  {done:>4}/{total} ({pct:5.1f}%)  "
                  f"elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    elapsed = time.time() - t_start
    df = pd.DataFrame(records)

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Table shape : {df.shape}")
    print(f"\n  Sample rows:")
    print(df.head(5).to_string(index=False))

    return df


def save_lookup(df: pd.DataFrame) -> None:
    """Save lookup table to CSV."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")
    print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = precompute_lookup_table()
    save_lookup(df)
    print("\n  Ready for rl_environment.py")