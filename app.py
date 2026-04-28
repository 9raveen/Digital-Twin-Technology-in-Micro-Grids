import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# LOAD MODEL + FEATURES
# ─────────────────────────────────────────────
model = joblib.load("outputs/model/best_model.pkl")
feature_names = joblib.load("outputs/model/feature_names.pkl")

# Load dataset for realistic defaults
df_ref = pd.read_csv("datasets/processed/simulation_results.csv")
feature_means = df_ref.mean(numeric_only=True)

# Ensure all features exist in means
for col in feature_names:
    if col not in feature_means:
        feature_means[col] = 0.0

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Digital Twin Microgrid", layout="wide")

st.title("⚡ Digital Twin Microgrid Dashboard")
st.markdown("### Real-time Blackout Risk Prediction")

# ─────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────
st.sidebar.header("🔧 System Inputs")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Custom", "High Load Stress", "Solar Rich", "Night Peak"]
)

p_load = st.sidebar.slider("Load (MW)", 1.0, 4.0, 2.5)
p_solar = st.sidebar.slider("Solar (MW)", 0.0, 3.5, 1.0)
hour = st.sidebar.slider("Hour", 0, 23, 12)

# Apply scenarios
if scenario == "High Load Stress":
    p_load, p_solar = 3.8, 0.5
elif scenario == "Solar Rich":
    p_load, p_solar = 2.0, 3.0
elif scenario == "Night Peak":
    p_load, p_solar = 3.5, 0.0

is_night = 1 if hour < 6 or hour > 18 else 0
net_load = p_load - p_solar

# ─────────────────────────────────────────────
# BUILD FEATURE VECTOR (FINAL FIX)
# ─────────────────────────────────────────────

# Start with ALL features initialized to 0
X = pd.DataFrame([dict.fromkeys(feature_names, 0.0)])

# Fill known inputs
if 'hour_of_day' in feature_names:
    X.loc[0, 'hour_of_day'] = hour

if 'is_night' in feature_names:
    X.loc[0, 'is_night'] = is_night

if 'p_load_mw' in feature_names:
    X.loc[0, 'p_load_mw'] = p_load

if 'p_solar_mw' in feature_names:
    X.loc[0, 'p_solar_mw'] = p_solar

if 'soc' in feature_names:
    X.loc[0, 'soc'] = 0.2

# Reconstruct useful features
if 'p_grid_mw' in feature_names:
    X.loc[0, 'p_grid_mw'] = net_load

# Fill remaining features with mean values
for col in feature_names:
    if X.loc[0, col] == 0 and col in feature_means:
        X.loc[0, col] = feature_means[col]

# Final strict alignment
X = X[feature_names]

# Safety check
if X.shape[1] != len(feature_names):
    st.error(f"Feature mismatch: {X.shape[1]} vs {len(feature_names)}")
    st.stop()

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
try:
    prob = model.predict_proba(X)[0][1]
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()

# ─────────────────────────────────────────────
# MAIN METRICS
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("⚠️ Blackout Risk", f"{prob*100:.1f}%")

with col2:
    st.metric("⚡ Net Load", f"{net_load:.2f} MW")

with col3:
    if prob < 0.3:
        st.success("LOW RISK")
    elif prob < 0.7:
        st.warning("MEDIUM RISK")
    else:
        st.error("HIGH RISK")

# ─────────────────────────────────────────────
# LOAD VS SOLAR
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Load vs Solar")

df_plot = pd.DataFrame({
    "Type": ["Load", "Solar"],
    "MW": [p_load, p_solar]
})

st.bar_chart(df_plot.set_index("Type"))

# ─────────────────────────────────────────────
# 24-HOUR DIGITAL TWIN SIMULATION
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🔁 24-Hour Simulation")

hours = np.arange(24)
probs = []
soc = 0.5
soc_values = []

for h in hours:
    is_night_h = 1 if h < 6 or h > 18 else 0

    load_h = p_load + 0.3 * np.sin(h / 24 * 2 * np.pi)
    solar_h = max(0, p_solar * np.sin(h / 24 * np.pi))
    net_h = load_h - solar_h

    X_sim = pd.DataFrame([dict.fromkeys(feature_names, 0.0)])

    if 'hour_of_day' in feature_names:
        X_sim.loc[0, 'hour_of_day'] = h

    if 'is_night' in feature_names:
        X_sim.loc[0, 'is_night'] = is_night_h

    if 'p_load_mw' in feature_names:
        X_sim.loc[0, 'p_load_mw'] = load_h

    if 'p_solar_mw' in feature_names:
        X_sim.loc[0, 'p_solar_mw'] = solar_h

    if 'p_grid_mw' in feature_names:
        X_sim.loc[0, 'p_grid_mw'] = net_h

    # Fill missing with means
    for col in feature_names:
        if X_sim.loc[0, col] == 0 and col in feature_means:
            X_sim.loc[0, col] = feature_means[col]

    X_sim = X_sim[feature_names]

    prob_h = model.predict_proba(X_sim)[0][1]
    probs.append(prob_h)

    # Battery SOC simulation
    surplus = solar_h - load_h
    soc += 0.05 * surplus
    soc = np.clip(soc, 0.1, 0.9)
    soc_values.append(soc)

df_sim = pd.DataFrame({
    "Hour": hours,
    "Blackout Probability": probs
})

df_soc = pd.DataFrame({
    "Hour": hours,
    "SOC": soc_values
})

st.line_chart(df_sim.set_index("Hour"))
st.line_chart(df_soc.set_index("Hour"))

# ─────────────────────────────────────────────
# VOLTAGE ESTIMATION
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("⚡ Voltage Estimate")

v_min_est = 1.0 - (0.08 * (p_load / 3.7)) + (0.05 * (p_solar / 3.0))
v_min_est = np.clip(v_min_est, 0.9, 1.05)

st.metric("Estimated Vmin (pu)", f"{v_min_est:.3f}")

if v_min_est < 0.93:
    st.error("Voltage Critical ⚠️")
elif v_min_est < 0.95:
    st.warning("Voltage Low")
else:
    st.success("Voltage Stable")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Digital Twin Microgrid | ML-based Blackout Prediction")