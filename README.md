# ⚡ Hybrid Physics–ML Digital Twin for Stability Analysis and Blackout Prediction in Microgrids

> **PHYSIC-DT-RISK** · IEEE 33-Bus · AC Newton-Raphson · Machine Learning · Streamlit Dashboard

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![pandapower](https://img.shields.io/badge/pandapower-3.4.0-orange)](https://pandapower.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IIIT Nagpur](https://img.shields.io/badge/IIIT-Nagpur-blue)](https://iiitn.ac.in/)

A physics-informed offline Digital Twin of an IEEE 33-bus microgrid that simulates 30 days of operation under diverse load, solar, and battery conditions — and trains machine learning models to predict the probability of a blackout at each hourly timestep using only observable pre-physics inputs.

Developed as **Mini Project II** at the **Department of Computer Science & Engineering, IIIT Nagpur** (Jan–May 2026), under the supervision of **Dr. Khushboo A. Jain**.

LIVE APP - https://digital-twin-microgrid-pr4.streamlit.app/
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [Datasets](#-datasets)
- [Installation](#-installation)
- [Usage](#-usage)
- [ML Pipelines](#-ml-pipelines)
- [Results](#-results)
- [Dependencies](#-dependencies)
- [Hardware Requirements](#-hardware-requirements)
- [Team](#-team)

---

## 🔍 Overview

Modern microgrids — integrating renewable solar generation, battery storage, and variable loads — create complex, time-varying behaviour that traditional power system analysis tools cannot reliably model. Real-world blackout events are rare, meaning no labeled training dataset exists.

This project addresses both problems through a **Digital Twin** approach:

1. A physics-accurate simulation of the IEEE 33-bus distribution network generates thousands of labeled failure scenarios synthetically.
2. Machine learning models trained on this data predict blackout risk from three simple, pre-physics observable inputs: load demand, solar generation, and battery state-of-charge.

The power balance equation driving the system:

```
P_load(t) = P_solar(t) + P_battery(t) + P_grid(t)
```

A blackout occurs when this balance cannot be maintained above the voltage threshold of **v_min < 0.93 pu**.

---

## ✨ Key Features

- **Physics-Informed Simulation** — IEEE 33-bus AC Newton-Raphson load flow via `pandapower 3.4.0` with ZIP voltage-dependent load models (30% Z + 30% I + 40% P)
- **Realistic BESS Modeling** — SOC tracking, split charge/discharge efficiency (95%), C-rate dependent losses, ramp rate limits (0.1 MW/step), grid-aware discharge control
- **35,850 Labeled Training Samples** — 50 independent scenarios × 720 hours covering NORMAL, STRESSED, EXTREME, and SEASONAL operating conditions
- **Zero Data Leakage** — voltage features (which encode the label) are fully excluded; ML trains on only 3 pre-physics observables
- **Dual ML Pipeline** — temporal regression (Pipeline 1) for dynamics + scenario-based classification (Pipeline 2) for rare event generalization
- **Streamlit Dashboard** — real-time blackout risk monitoring, voltage profiles, SOC tracking, and live prediction

---

## 📁 Project Structure

```
microgrid-digital-twin/
│
├── data/
│   ├── raw/
│   │   ├── LD2011_2014.txt               # UCI Electricity Load Diagrams
│   │   └── Plant_1_Generation_Data.csv   # Kaggle Solar Power Generation
│   └── processed/
│       ├── load_scaled.csv               # 720-hour load profile (3.715 MW)
│       ├── solar_scaled.csv              # 720-hour solar profile (3.0 MW)
│       ├── simulation_results.csv        # 720 × 18 features (baseline run)
│       └── full_dataset.csv              # 35,850 × 18 features (all scenarios)
│
├── src/
│   ├── data_load.py                      # UCI load preprocessing pipeline
│   ├── data_solar.py                     # Kaggle solar preprocessing pipeline
│   ├── battery_model.py                  # BESS simulation with SOC control
│   ├── grid_simulator.py                 # pandapower IEEE 33-bus AC load flow
│   ├── label_generator.py                # Blackout labeling + risk scoring
│   ├── digital_twin.py                   # Main simulation loop (720 timesteps)
│   ├── scenario_generator.py             # 50-scenario parameterized runner
│   ├── dataset_generator.py              # Combines all scenarios → full_dataset.csv
│   └── ml_model.py                       # Training, evaluation, model export
│
├── dashboard/
│   └── app.py                            # Streamlit real-time monitoring UI
│
├── outputs/
│   ├── model/
│   │   ├── best_model.joblib             # Saved XGBoost classifier
│   │   ├── scaler.joblib                 # StandardScaler artifact
│   │   └── feature_names.json           # Feature list for inference
│   ├── plots/                            # Confusion matrices, feature importance
│   └── reports/                          # Evaluation metrics, classification reports
│
├── notebooks/
│   └── analysis.ipynb                    # Exploratory data analysis
│
├── requirements.txt
└── README.md
```

---

## 🏗 System Architecture

```
UCI Load Data ──────┐
                    ▼
              data_load.py ──► load_scaled.csv
                                      │
Kaggle Solar Data ──┐                 ▼
                    ▼         ┌───────────────────┐
              data_solar.py ──►  digital_twin.py  │
                              │  (t = 0..719)     │
                              │  ┌─────────────┐  │
                              │  │battery_model│  │
                              │  └──────┬──────┘  │
                              │  ┌──────▼──────┐  │
                              │  │grid_simulato│  │  ← pandapower AC NR
                              │  └──────┬──────┘  │
                              │  ┌──────▼──────┐  │
                              │  │label_generat│  │  ← v_min < 0.93 pu
                              │  └─────────────┘  │
                              └────────┬──────────┘
                                       │
                              simulation_results.csv (720 × 18)
                                       │
                         scenario_generator.py (× 50)
                                       │
                              full_dataset.csv (35,850 × 18)
                                       │
                         ┌─────────────┴────────────┐
                         ▼                          ▼
                   Pipeline 1                 Pipeline 2
                (Temporal Regression)    (Scenario Classification)
                  Random Forest ★              XGBoost ★
                   R² = 0.8646             ROC-AUC = 0.9711
                         │                          │
                         └─────────────┬────────────┘
                                       ▼
                              Streamlit Dashboard
                           (dashboard/app.py)
```

---

## 📊 Datasets

| Dataset                                 | Source                                                                                       | Description                                                                                       |
| --------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| UCI Electricity Load Diagrams 2011–2014 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) | 370 residential clients (Portugal), 15-min resolution → aggregated, resampled, scaled to 3.715 MW |
| Kaggle Solar Power Generation — Plant 1 | [Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)              | 22 inverters, India plant, May–Jun 2020, DC power → grouped, converted, scaled to 3.0 MW          |
| IEEE 33-Bus Distribution Network        | Built-in via `pandapower.networks.case33bw()`                                                | 33 buses, 37 lines, 12.66 kV, base load 3.715 MW                                                  |

**Blackout threshold:** `v_min < 0.93 pu` (tuned from 0.95 pu which gave unusable 68% imbalance → 0.93 pu gives balanced 21.4% blackout rate)

---

## 🚀 Installation

### Prerequisites

- Python 3.12
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/microgrid-digital-twin.git
cd microgrid-digital-twin
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the datasets

Place the raw data files in `data/raw/`:

- **UCI Load Dataset:** Download `LD2011_2014.txt` from [UCI ML Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
- **Kaggle Solar Dataset:** Download `Plant_1_Generation_Data.csv` from [Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)

---

## ▶️ Usage

### Step 1 — Preprocess data

```bash
python src/data_load.py      # Generates data/processed/load_scaled.csv
python src/data_solar.py     # Generates data/processed/solar_scaled.csv
```

### Step 2 — Run the Digital Twin (baseline, 720 hours)

```bash
python src/digital_twin.py   # Generates data/processed/simulation_results.csv
```

### Step 3 — Generate scenario-based dataset (50 scenarios)

```bash
python src/dataset_generator.py  # Generates data/processed/full_dataset.csv
```

### Step 4 — Train ML models

```bash
python src/ml_model.py       # Trains both pipelines, saves model to outputs/model/
```

### Step 5 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`.

---

## 🤖 ML Pipelines

### Pipeline 1 — Temporal Regression

| Setting    | Value                                              |
| ---------- | -------------------------------------------------- |
| Dataset    | 720 hours (baseline simulation)                    |
| Features   | `p_load_mw`, `p_solar_mw`, `soc`                   |
| Target     | `risk_score` (continuous, 0–1)                     |
| Split      | Temporal — train: hours 0–599, test: hours 600–719 |
| Best Model | Random Forest (R² = 0.8646)                        |

### Pipeline 2 — Scenario-Based Classification

| Setting    | Value                                              |
| ---------- | -------------------------------------------------- |
| Dataset    | 35,850 samples (50 scenarios)                      |
| Features   | `p_load_mw`, `p_solar_mw`, `soc`                   |
| Target     | `blackout` (binary 0/1)                            |
| Split      | By scenario ID — no temporal leakage               |
| Best Model | XGBoost (ROC-AUC = 0.9711, Blackout Recall = 0.95) |

**Why only 3 features?** Features derived after running the load flow (e.g., `v_min`, `v_mean`, `p_grid_mw`) directly encode the blackout label — using them is data leakage. Only pre-physics observables are valid inputs.

---

## 📈 Results

### Pipeline 1 — Regression

| Model               | R²         | RMSE       | MAE        | CV-R²      |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| **Random Forest ★** | **0.8646** | **0.0718** | **0.0567** | **0.8516** |
| Gradient Boosting   | 0.8475     | 0.0762     | 0.0602     | 0.8364     |
| Ridge Regression    | 0.8464     | —          | —          | —          |
| XGBoost             | —          | —          | —          | —          |

### Pipeline 2 — Classification

| Model               | ROC-AUC    | Blackout F1 | Blackout Recall |
| ------------------- | ---------- | ----------- | --------------- |
| **XGBoost ★**       | **0.9711** | —           | **0.95**        |
| Random Forest       | —          | —           | —               |
| Gradient Boosting   | —          | —           | —               |
| Logistic Regression | —          | —           | —               |

**Key findings:**

- `net_load` (= p_load − p_solar − p_battery) is the strongest predictor of blackout risk
- Critical threshold: `net_load > 2.979 MW` → voltage crosses blackout boundary
- 154 of 720 baseline hours cross the blackout threshold; 39 are preventable by optimal battery dispatch
- Scenario-based training significantly improves generalization to rare extreme events

---

## 📦 Dependencies

```txt
# Core
python==3.12

# Power Systems
pandapower==3.4.0

# Machine Learning
scikit-learn>=1.0
xgboost>=2.0
joblib

# Data Processing
numpy
pandas
scipy

# Visualization & Dashboard
matplotlib
streamlit>=1.0

# RL Environment (optional)
gymnasium>=0.29
```

Install all at once:

```bash
pip install pandapower==3.4.0 scikit-learn xgboost joblib numpy pandas scipy matplotlib streamlit gymnasium
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

---

## 💻 Hardware Requirements

| Component | Minimum                                | Recommended                 |
| --------- | -------------------------------------- | --------------------------- |
| Processor | Intel Core i5 / AMD Ryzen 5 (4+ cores) | Intel Core i7 / AMD Ryzen 7 |
| RAM       | 8 GB                                   | 16 GB                       |
| Storage   | 5 GB free                              | SSD recommended             |
| OS        | Windows 10/11 or Ubuntu 20.04+         | Windows 11 / Ubuntu 22.04   |

> **Note:** The 50-scenario simulation runs ~36,000 AC load flow solves. A precomputed 425-point lookup table reduces per-step time to < 1 ms. Full dataset generation takes approximately 10–15 minutes on recommended hardware.

---

## 👥 Team

| Name                        | 
| --------------------------- | 
| Saidu Venkata Revanth Varma | 
| Vankadoth Praveen           | 
| Bojja Manikanta             | 

**Guide:** Dr. Khushboo A. Jain, Assistant Professor, CSE, IIIT Nagpur

**Institution:** Indian Institute of Information Technology, Nagpur
**Department:** Computer Science and Engineering (CSE-AIML)
**Semester:** 6th | January – May 2026

---

## 📚 References

- Grieves, M. (2014). Digital Twin: Manufacturing Excellence through Virtual Factory Replication.
- Tao, F. et al. (2018). Digital Twin-Driven Product Design, Manufacturing and Service.
- Thurner, L. et al. (2018). pandapower — An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems. _IEEE Transactions on Power Systems._
- Zhao, J. et al. (2020). Machine Learning for Voltage Instability Classification.
- UCI Machine Learning Repository — Electricity Load Diagrams 2011–2014.
- Kaggle — Solar Power Generation Data.

---

## 📄 License

This project is submitted as an academic Mini Project at IIIT Nagpur. Please cite appropriately if referencing this work.
