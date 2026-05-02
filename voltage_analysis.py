import pandas as pd
import numpy as np
import sys

sys.path.append(".")

from data_load import prepare_load_series
from data_solar import prepare_solar_series
from battery_model import BatteryModel
from grid_simulator import create_network, run_load_flow

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
load = prepare_load_series(
    'datasets/LD2011_2014.txt',
    n_days=30
)

solar = prepare_solar_series(
    'datasets/Solar Power Generation Data/Plant_1_Generation_Data.csv',
    n_days=30
)

# ─────────────────────────────────────────────────────────────
# Initialize system
# ─────────────────────────────────────────────────────────────
net = create_network()
battery = BatteryModel()

total = len(load)

# Storage
violations = 0
v_mins = []

# ─────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────
for t in range(total):
    p_load  = load.iloc[t]
    p_solar = solar.iloc[t]

    # Battery step
    p_batt, soc = battery.step(p_load, p_solar)

    # Power flow
    result = run_load_flow(net, p_load, p_solar)

    # Store minimum voltage
    v_mins.append(result['v_min'])

    # Count violations (threshold = 0.95)
    if result['v_min'] < 0.95:
        violations += 1

# Convert to numpy array
v_mins = np.array(v_mins)

# ─────────────────────────────────────────────────────────────
# Basic Results
# ─────────────────────────────────────────────────────────────
print("=" * 50)
print("Voltage Violation Analysis")
print("=" * 50)

print(f"Total hours     : {total}")
print(f"Violations      : {violations}")
print(f"Blackout rate   : {violations / total * 100:.1f}%")
print(f"Normal hours    : {total - violations}")

# ─────────────────────────────────────────────────────────────
# Threshold Analysis
# ─────────────────────────────────────────────────────────────
print("\nVoltage Threshold Sensitivity:")

for threshold in [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]:
    rate = (v_mins < threshold).mean() * 100
    print(f"V_min < {threshold:.2f} → {rate:.1f}%")

# ─────────────────────────────────────────────────────────────
# Optional: Quick stats
# ─────────────────────────────────────────────────────────────
print("\nVoltage Statistics:")
print(f"Min voltage : {v_mins.min():.4f}")
print(f"Max voltage : {v_mins.max():.4f}")
print(f"Mean voltage: {v_mins.mean():.4f}")