# Quick Start Execution Guide

## How to Run Everything (Step by Step)

### **Option A: Quick Test (15 minutes)**

```bash
cd c:/Users/bmani/OneDrive/Desktop/Digital-Twin-Technology-in-Micro-Grids

# [1] PREPARE DATA (Solar FIRST, Load SECOND)
echo "=== STEP 1: Loading Solar Data ==="
python3 data_solar.py
echo "✓ Solar data ready (solar_scaled.csv)"

# [2] Load scaled data
echo -e "\n=== STEP 2: Loading Load Data ==="
python3 data_load.py
echo "✓ Load data ready (load_scaled.csv)"

# [3] Validate both
echo -e "\n=== STEP 3: Validating Data ==="
python3 validation.py
echo "✓ Data validation complete"

# [4] Single simulation
echo -e "\n=== STEP 4: Running Simulation ==="
python3 digital_twin.py
echo "✓ Simulation complete (simulation_results.csv)"

# [5] ML training
echo -e "\n=== STEP 5: Training ML Model ==="
python3 ml_model.py
echo "✓ ML model ready (best_model.pkl)"

# [6] Launch dashboard
echo -e "\n=== STEP 6: Launching Dashboard ==="
streamlit run app.py
```

---

### **Option B: Single Command (All-in-One)**

```bash
cd c:/Users/bmani/OneDrive/Desktop/Digital-Twin-Technology-in-Micro-Grids

python3 run_sim.py

# Then in another terminal:
streamlit run app.py
```

---

### **Option C: Generate Large Dataset (1 hour)**

```bash
cd c:/Users/bmani/OneDrive/Desktop/Digital-Twin-Technology-in-Micro-Grids

# [1] Prepare data (same as above)
python3 data_solar.py
python3 data_load.py
python3 validation.py

# [2] Generate 50 scenarios (36,000 samples)
echo -e "\n=== GENERATING DATASET ==="
python3 generate_dataset.py \
  --n-scenarios 50 \
  --output-dir datasets/generated \
  --seed 42

echo "✓ Dataset ready: datasets/generated/"
echo "  - train.csv (18,000 samples)"
echo "  - val.csv (6,300 samples)"
echo "  - test.csv (11,700 samples)"
```

---

## What Each File Does

### **Data Preparation (Must run in this order)**

#### 1️⃣ **data_solar.py** (Solar FIRST)
```bash
python3 data_solar.py
```
- Loads Kaggle solar generation data
- Resamples 15-min → hourly
- Normalizes & scales to MW
- **Output:** `datasets/processed/solar_scaled.csv`
- **Time:** ~2 minutes

#### 2️⃣ **data_load.py** (Load SECOND)
```bash
python3 data_load.py
```
- Loads UCI electricity load data
- Resamples 15-min → hourly
- Normalizes & scales to MW
- **Output:** `datasets/processed/load_scaled.csv`
- **Time:** ~2 minutes
- **Requires:** Solar data must be prepared first

#### 3️⃣ **validation.py** (Validate both)
```bash
python3 validation.py
```
- Checks solar series for NaNs, bounds, realism
- Checks load series for NaNs, bounds, realism
- Prints summary statistics
- **Output:** Console reports
- **Time:** <1 minute

---

### **Simulation & Modeling**

#### 4️⃣ **digital_twin.py** (Main simulation)
```bash
python3 digital_twin.py
```
- Runs 720-hour (30-day) simulation
- Battery model with 5 control strategies
- IEEE 33-bus AC load flow
- Generates blackout labels
- **Output:** `datasets/processed/simulation_results.csv` (720 samples)
- **Time:** ~5-10 minutes

#### 5️⃣ **ml_model.py** (Train ML models)
```bash
python3 ml_model.py
```
- Trains Random Forest, Gradient Boosting, XGBoost, Ridge
- Predicts risk scores (0-1)
- Evaluates on test set
- Saves best model
- **Output:** 
  - `outputs/model/best_model.pkl`
  - `outputs/model/scaler.pkl`
  - `outputs/ml_results.png`
- **Time:** ~3-5 minutes

---

### **Dashboard**

#### 6️⃣ **app.py** (Interactive UI)
```bash
streamlit run app.py
```
- Web interface for predictions
- Interactive scenario selection
- Real-time risk scoring
- **Output:** Web dashboard (http://localhost:8501)
- **Time:** Instant (server mode)

---

### **Large-Scale Dataset Generation**

#### 7️⃣ **generate_dataset.py** (Multi-scenario orchestrator)
```bash
python3 generate_dataset.py --n-scenarios 50 --output-dir datasets/generated --seed 42
```
- Generates 50 independent scenarios
- Creates 36,000 samples total
- Validates quality & diversity
- Splits into train/val/test
- **Output:**
  - `datasets/generated/train.csv` (18,000)
  - `datasets/generated/val.csv` (6,300)
  - `datasets/generated/test.csv` (11,700)
  - `datasets/generated/dataset_info.txt`
  - `datasets/generated/analysis_report.html`
- **Time:** ~45-60 minutes

---

## File Execution Dependency Graph

```
START
  │
  ├─→ [1] data_solar.py ✓
  │   └─→ solar_scaled.csv
  │       │
  │       └─→ [2] data_load.py ✓
  │           └─→ load_scaled.csv
  │               │
  │               ├─→ [3] validation.py ✓
  │               │   └─→ Reports
  │               │
  │               └─→ [4] digital_twin.py ✓
  │                   └─→ simulation_results.csv
  │                       │
  │                       ├─→ [5] ml_model.py ✓
  │                       │   └─→ best_model.pkl
  │                       │       │
  │                       │       └─→ [6] app.py ✓
  │                       │           └─→ Dashboard
  │                       │
  │                       └─→ [7] generate_dataset.py ✓
  │                           └─→ train/val/test CSVs
  │
  └─→ OTHER TOOLS (optional)
      ├─ voltage_analysis.py
      ├─ validate_battery_injection.py
      ├─ profiling.py
      ├─ compare_zip_models.py
      └─ hyperparameter_optimization.py
```

---

## Verification Checklist

After each step, verify success:

```bash
# After data_solar.py
ls -lh datasets/processed/solar_scaled.csv
# Expected: File exists, ~1-2 MB

# After data_load.py  
ls -lh datasets/processed/load_scaled.csv
# Expected: File exists, ~1-2 MB

# After digital_twin.py
ls -lh datasets/processed/simulation_results.csv
# Expected: File exists, 720 rows (+ header)

# After ml_model.py
ls -lh outputs/model/best_model.pkl
# Expected: File exists, ~5-10 MB

# After generate_dataset.py
ls -lh datasets/generated/
# Expected: train.csv, val.csv, test.csv exist
```

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `solar_scaled.csv not found` | data_solar.py not run | Run `python3 data_solar.py` FIRST |
| `load_scaled.csv not found` | data_load.py not run | Run `python3 data_load.py` SECOND |
| `simulation_results.csv not found` | digital_twin.py failed | Run `python3 digital_twin.py` again |
| `best_model.pkl not found` | ml_model.py failed | Check error output, rerun |
| `ModuleNotFoundError` | Missing dependency | `pip install pandas numpy scikit-learn` |
| Long generation time | Normal (1-2h for 100 scenarios) | Use smaller n_scenarios (10-20) for testing |
| Low model accuracy | Small dataset (720 samples) | Generate larger dataset with generate_dataset.py |

---

## Next Steps After Execution

### ✅ After Step 4 (digital_twin.py)
- Dataset: 720 samples
- Baseline model ready
- Can train and test models

### ✅ After Step 5 (ml_model.py)
- Model accuracy: ~75-85% on test set
- Feature importance available
- Model can make predictions

### ✅ After Step 6 (app.py)
- Interactive dashboard running
- Can input custom scenarios
- Real-time predictions

### ✅ After Step 7 (generate_dataset.py)
- Large dataset: 36,000 samples
- Better model training
- Improved generalization (~85-90% AUC)

---

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| data_solar.py | 2 min | First-time parsing large CSV |
| data_load.py | 2 min | Depends on data_solar completion |
| validation.py | <1 min | Quick checks |
| digital_twin.py | 5-10 min | 720-hour simulation loop |
| ml_model.py | 3-5 min | Model training & evaluation |
| app.py | Instant | Web server startup |
| generate_dataset.py (10 scenarios) | 10 min | Quick test run |
| generate_dataset.py (50 scenarios) | 50 min | Production dataset |
| generate_dataset.py (100 scenarios) | 2 hours | Maximum diversity |

---

## File Connections at a Glance

```
┌──────────────────────────────────────────────────────┐
│ TIER 1: DATA LOADING (Order matters!)               │
├──────────────────────────────────────────────────────┤
│ data_solar.py  ──→  solar_scaled.csv                │
│        ↓                                              │
│ data_load.py   ──→  load_scaled.csv                 │
│        ↓                                              │
│ validation.py  ──→  Validation Reports              │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ TIER 2: SIMULATION                                   │
├──────────────────────────────────────────────────────┤
│ digital_twin.py (with battery_model + grid_sim)    │
│        ↓                                              │
│ simulation_results.csv (720 samples)                 │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ TIER 3: ML TRAINING                                  │
├──────────────────────────────────────────────────────┤
│ ml_model.py                                          │
│        ↓                                              │
│ best_model.pkl + scaler.pkl + ml_results.png        │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ TIER 4: DASHBOARD                                    │
├──────────────────────────────────────────────────────┤
│ app.py (Streamlit)                                   │
│        ↓                                              │
│ http://localhost:8501                                │
└──────────────────────────────────────────────────────┘

ALTERNATIVE: LARGE-SCALE DATASET
┌──────────────────────────────────────────────────────┐
│ generate_dataset.py (uses all above + scenarios)    │
│        ↓                                              │
│ train.csv (18k) + val.csv (6.3k) + test.csv (11.7k) │
│        ↓                                              │
│ Better ML models with 85-90% AUC                     │
└──────────────────────────────────────────────────────┘
```

---

## Summary

**Golden Rule:** 
```
SOLAR FIRST → LOAD SECOND → Validate → Simulate → Train ML → Dashboard
```

**All files are connected.** No changes to structure needed. Just follow the execution order above.

**Time to completion:**
- Quick test: 15 minutes
- Full single run: 25 minutes  
- Production dataset: 1 hour

Good luck! 🚀
