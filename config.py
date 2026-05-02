"""
config.py
=========
Centralized configuration for Digital Twin Microgrid project.
Replaces hardcoded paths and parameters scattered across modules.
"""

import os
from pathlib import Path

# ── PROJECT ROOT ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"

# ── Create directories if missing ──────────────────────────────────────────
for directory in [PROCESSED_DIR, OUTPUT_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ── DATA PATHS ─────────────────────────────────────────────────────────────
RAW_LOAD_CSV = DATA_DIR / "LD2011_2014.txt"
RAW_SOLAR_CSV = DATA_DIR / "Solar Power Generation Data" / "Plant_1_Generation_Data.csv"

LOAD_SCALED_CSV = PROCESSED_DIR / "load_scaled.csv"
SOLAR_SCALED_CSV = PROCESSED_DIR / "solar_scaled.csv"
SIMULATION_RESULTS_CSV = PROCESSED_DIR / "simulation_results.csv"

# ── MODEL PATHS ───────────────────────────────────────────────────────────
BEST_MODEL_PKL = MODEL_DIR / "best_model.pkl"
SCALER_PKL = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PKL = MODEL_DIR / "feature_names.pkl"

# ── SIMULATION PARAMETERS ─────────────────────────────────────────────────
SIMULATION_DAYS = 30
SIMULATION_HOURS = SIMULATION_DAYS * 24  # 720 hours
TIMESTEP_HOURS = 1.0

# ── GRID PARAMETERS ───────────────────────────────────────────────────────
IEEE33_BASE_LOAD_MW = 3.715
IEEE33_BASE_LOAD_MVAR = 2.300
SOLAR_BUS = 17
BATTERY_BUS = 17

# Voltage limits (IEEE standard)
V_MIN_LIMIT = 0.93  # Lower bound (pu)
V_MAX_LIMIT = 1.05  # Upper bound (pu)
BLACKOUT_THRESHOLD = 0.93  # Stricter than IEEE 0.95

# ── ZIP LOAD MODEL PARAMETERS ─────────────────────────────────────────────
ZIP_Z_PERCENT = 30  # Impedance (proportional to V²)
ZIP_I_PERCENT = 30  # Current (proportional to V)
ZIP_P_PERCENT = 40  # Constant Power

# ── BATTERY PARAMETERS ────────────────────────────────────────────────────
BATTERY_CAPACITY_MWH = 0.5
BATTERY_SOC_INIT = 0.5
BATTERY_SOC_MIN = 0.1
BATTERY_SOC_MAX = 0.9
BATTERY_CHARGE_RATE_MW = 0.25
BATTERY_DISCHARGE_RATE_MW = 0.25
BATTERY_EFFICIENCY = 0.95
BATTERY_RESPONSE_FACTOR = 0.5
BATTERY_TARGET_SOC = 0.6
BATTERY_GRID_IMPORT_LIMIT_MW = 5.0

# ── TIME/NIGHT BOUNDARIES ─────────────────────────────────────────────────
# Day = 6 AM to 8 PM (6 to 20), Night = 8 PM to 6 AM
DAY_START_HOUR = 6
DAY_END_HOUR = 20

# ── MEASUREMENT & NOISE ───────────────────────────────────────────────────
V_MEASUREMENT_NOISE_STD = 0.007  # ±0.7% sensor error
LOAD_MEASUREMENT_NOISE_STD = 0.02  # ±2% meter uncertainty

# ── ML MODEL PARAMETERS ───────────────────────────────────────────────────
TRAIN_TEST_SPLIT_DAY = 24  # Train: days 0-23, Test: days 24-29
RANDOM_STATE = 42

# Lag features
LAG_PERIODS = [1, 2, 3]  # 1-hour, 2-hour, 3-hour lags

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2

# Gradient Boosting
GB_N_ESTIMATORS = 200
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH = 5
GB_MIN_SAMPLES_SPLIT = 5

# XGBoost (if available)
XGB_N_ESTIMATORS = 200
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 5

# Ridge
RIDGE_ALPHA = 1.0

# ── VALIDATION PARAMETERS ────────────────────────────────────────────────
MIN_LOAD_MW = 0.5
MAX_LOAD_MW = 4.5
MIN_SOLAR_MW = 0.0
MAX_SOLAR_MW = 3.5
MIN_SOC = 0.0
MAX_SOC = 1.0
MIN_VOLTAGE_PU = 0.8
MAX_VOLTAGE_PU = 1.1

# ── PERFORMANCE PROFILING ────────────────────────────────────────────────
ENABLE_PROFILING = False
PROFILING_OUTPUT_DIR = OUTPUT_DIR / "profiling"

# ── LOGGING ──────────────────────────────────────────────────────────────
ENABLE_DEBUG_LOGGING = False
LOG_FILE = OUTPUT_DIR / "digital_twin.log"

# ── DISPLAY PARAMETERS ─────────────────────────────────────────────────────
DISPLAY_PROGRESS_INTERVAL = 72  # Print progress every N timesteps
DISPLAY_DECIMALS = 4
