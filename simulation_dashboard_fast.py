"""
Digital Twin Microgrid - FAST Simulation Dashboard (Lightweight)
================================================================
Instant comparison without expensive 24-hour profiles.
24-hour analysis available as optional button-triggered calculation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from grid_simulator import run_load_flow, create_network
import joblib

# PAGE CONFIG
st.set_page_config(page_title="Digital Twin Microgrid - FAST", layout="wide")

# Load model
model = joblib.load("outputs/model/best_model.pkl")
feature_names = joblib.load("outputs/model/feature_names.pkl")
df_ref = pd.read_csv("datasets/processed/simulation_results.csv")
feature_means = df_ref.mean(numeric_only=True)

for col in feature_names:
    if col not in feature_means:
        feature_means[col] = 0.0

# Initialize session state
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {
        "Scenario A": {"p_load": 2.5, "p_solar": 1.0, "p_battery": 0.0, "hour": 12, "use_zip": True},
        "Scenario B": {"p_load": 3.5, "p_solar": 0.5, "p_battery": 0.3, "hour": 18, "use_zip": True},
    }

# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

@st.cache_data
def predict_blackout_risk(p_load: float, p_solar: float, line_loading: float, soc: float = 0.5) -> float:
    """Predict blackout risk (cached)"""
    X = pd.DataFrame([dict.fromkeys(feature_names, 0.0)])
    p_grid = p_load - p_solar

    if "p_load_mw" in feature_names:
        X.loc[0, "p_load_mw"] = p_load
    if "p_solar_mw" in feature_names:
        X.loc[0, "p_solar_mw"] = p_solar
    if "soc" in feature_names:
        X.loc[0, "soc"] = soc
    if "line_loading_max" in feature_names:
        X.loc[0, "line_loading_max"] = line_loading
    if "p_grid_mw" in feature_names:
        X.loc[0, "p_grid_mw"] = p_grid

    for lag_feat in ["p_load_lag_1", "p_load_lag_2", "p_load_lag_3"]:
        if lag_feat in feature_names:
            X.loc[0, lag_feat] = p_load
    for lag_feat in ["p_solar_lag_1", "p_solar_lag_2", "p_solar_lag_3"]:
        if lag_feat in feature_names:
            X.loc[0, lag_feat] = p_solar

    for col in feature_names:
        if X.loc[0, col] == 0 and col in feature_means:
            X.loc[0, col] = feature_means[col]

    X = X[feature_names]
    prob = model.predict_proba(X)[0][1]
    return prob


@st.cache_data
def run_scenario_simulation(scenario_name: str, p_load: float, p_solar: float, p_battery: float, hour: int, use_zip: bool) -> dict:
    """Run simulation for a scenario (cached)"""
    net = create_network(use_zip=use_zip)
    results = run_load_flow(net, p_load, p_solar, p_battery)

    line_loading = results["line_loading_max"]
    blackout_risk = predict_blackout_risk(p_load, p_solar, line_loading)

    net_load = p_load - p_solar
    is_night = 1 if hour < 6 or hour > 18 else 0

    v_min = results["v_min"]
    if v_min < 0.93:
        v_status = "🔴 Critical"
    elif v_min < 0.95:
        v_status = "🟡 Low"
    else:
        v_status = "🟢 Stable"

    line_loading_val = results["line_loading_max"]
    if line_loading_val > 100:
        loading_status = "🔴 Overloaded"
    elif line_loading_val > 80:
        loading_status = "🟡 High"
    else:
        loading_status = "🟢 Normal"

    return {
        "scenario_name": scenario_name,
        "p_load": p_load,
        "p_solar": p_solar,
        "p_battery": p_battery,
        "hour": hour,
        "net_load": net_load,
        "is_night": is_night,
        "converged": results["converged"],
        "v_min": v_min,
        "v_max": results["v_max"],
        "v_mean": results["v_mean"],
        "v_status": v_status,
        "v_violation": results["v_violation"],
        "line_loading": line_loading_val,
        "loading_status": loading_status,
        "p_loss": results["p_loss_mw"],
        "blackout_risk": blackout_risk,
        "zip_mode": results["zip_mode"],
    }


def get_risk_color(risk: float) -> str:
    """Get color based on risk level"""
    if risk < 0.3:
        return "#90EE90"
    elif risk < 0.7:
        return "#FFD700"
    else:
        return "#FF6B6B"


# ─────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────
st.title("⚡ Digital Twin Microgrid - Fast Simulator")
st.markdown("**Real-time scenario comparison** (optimized for speed)")

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎮 Scenario Controls")

    scenario_selection = st.selectbox("Edit Scenario:", ["Scenario A", "Scenario B"])

    st.markdown(f"### {scenario_selection}")
    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")

    # Preset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 High Stress", use_container_width=True):
            st.session_state.scenarios[scenario_selection] = {"p_load": 3.8, "p_solar": 0.2, "p_battery": 0.0, "hour": 18, "use_zip": True}
            st.rerun()
        if st.button("☀️ Solar Peak", use_container_width=True):
            st.session_state.scenarios[scenario_selection] = {"p_load": 2.0, "p_solar": 3.0, "p_battery": 0.0, "hour": 12, "use_zip": True}
            st.rerun()
    with col2:
        if st.button("🌙 Night Peak", use_container_width=True):
            st.session_state.scenarios[scenario_selection] = {"p_load": 3.5, "p_solar": 0.0, "p_battery": 0.5, "hour": 21, "use_zip": True}
            st.rerun()
        if st.button("⚖️ Balanced", use_container_width=True):
            st.session_state.scenarios[scenario_selection] = {"p_load": 2.5, "p_solar": 1.5, "p_battery": 0.3, "hour": 14, "use_zip": True}
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Manual Controls")

    current_scenario = st.session_state.scenarios[scenario_selection]

    p_load = st.slider("💡 Load (MW)", 0.5, 4.0, current_scenario["p_load"], 0.1, key=f"{scenario_selection}_load")
    p_solar = st.slider("☀️ Solar (MW)", 0.0, 3.5, current_scenario["p_solar"], 0.1, key=f"{scenario_selection}_solar")
    p_battery = st.slider("🔋 Battery (MW)", 0.0, 1.0, current_scenario["p_battery"], 0.1, key=f"{scenario_selection}_battery")
    hour = st.slider("🕐 Hour", 0, 23, current_scenario["hour"], 1, key=f"{scenario_selection}_hour")
    use_zip = st.checkbox("ZIP Load Model", current_scenario["use_zip"], key=f"{scenario_selection}_zip")

    st.session_state.scenarios[scenario_selection] = {
        "p_load": p_load,
        "p_solar": p_solar,
        "p_battery": p_battery,
        "hour": hour,
        "use_zip": use_zip,
    }

# ─────────────────────────────────────────────────────────────────────
# RUN SIMULATIONS (FAST)
# ─────────────────────────────────────────────────────────────────────
sim_a = run_scenario_simulation(
    "Scenario A",
    st.session_state.scenarios["Scenario A"]["p_load"],
    st.session_state.scenarios["Scenario A"]["p_solar"],
    st.session_state.scenarios["Scenario A"]["p_battery"],
    st.session_state.scenarios["Scenario A"]["hour"],
    st.session_state.scenarios["Scenario A"]["use_zip"],
)
sim_b = run_scenario_simulation(
    "Scenario B",
    st.session_state.scenarios["Scenario B"]["p_load"],
    st.session_state.scenarios["Scenario B"]["p_solar"],
    st.session_state.scenarios["Scenario B"]["p_battery"],
    st.session_state.scenarios["Scenario B"]["hour"],
    st.session_state.scenarios["Scenario B"]["use_zip"],
)

# ─────────────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Key Metrics Comparison")

cols = st.columns(5)

with cols[0]:
    st.metric("A - Load (MW)", f"{sim_a['p_load']:.2f}", delta=f"Solar: {sim_a['p_solar']:.2f}")
    st.metric("B - Load (MW)", f"{sim_b['p_load']:.2f}", delta=f"Solar: {sim_b['p_solar']:.2f}")

with cols[1]:
    st.metric("A - Voltage (pu)", f"{sim_a['v_min']:.3f}", delta=sim_a["v_status"])
    st.metric("B - Voltage (pu)", f"{sim_b['v_min']:.3f}", delta=sim_b["v_status"])

with cols[2]:
    st.metric("A - Line Load (%)", f"{sim_a['line_loading']:.1f}", delta=sim_a["loading_status"])
    st.metric("B - Line Load (%)", f"{sim_b['line_loading']:.1f}", delta=sim_b["loading_status"])

with cols[3]:
    st.metric("A - Loss (MW)", f"{sim_a['p_loss']:.4f}", delta=f"Batt: {sim_a['p_battery']:.2f}")
    st.metric("B - Loss (MW)", f"{sim_b['p_loss']:.4f}", delta=f"Batt: {sim_b['p_battery']:.2f}")

with cols[4]:
    st.metric("A - Risk (%)", f"{sim_a['blackout_risk']*100:.1f}%", delta=f"{'🔴 HIGH' if sim_a['blackout_risk']>0.7 else '🟡 MED' if sim_a['blackout_risk']>0.3 else '🟢 LOW'}")
    st.metric("B - Risk (%)", f"{sim_b['blackout_risk']*100:.1f}%", delta=f"{'🔴 HIGH' if sim_b['blackout_risk']>0.7 else '🟡 MED' if sim_b['blackout_risk']>0.3 else '🟢 LOW'}")

# ─────────────────────────────────────────────────────────────────────
# ENERGY FLOW
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col_e1, col_e2 = st.columns(2)

with col_e1:
    st.subheader("📍 Scenario A - Energy Flow")
    values_a = [sim_a["p_load"], sim_a["p_solar"], sim_a["p_battery"], max(0, sim_a["p_load"] - sim_a["p_solar"] - sim_a["p_battery"])]
    fig_a = go.Figure(data=[go.Bar(x=["Load", "Solar", "Battery", "Grid"], y=values_a, marker_color=["#FF6B6B", "#FFD700", "#87CEEB", "#90EE90"], text=[f"{v:.2f}" for v in values_a], textposition="outside")])
    fig_a.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Power (MW)")
    st.plotly_chart(fig_a, width='stretch')

with col_e2:
    st.subheader("📍 Scenario B - Energy Flow")
    values_b = [sim_b["p_load"], sim_b["p_solar"], sim_b["p_battery"], max(0, sim_b["p_load"] - sim_b["p_solar"] - sim_b["p_battery"])]
    fig_b = go.Figure(data=[go.Bar(x=["Load", "Solar", "Battery", "Grid"], y=values_b, marker_color=["#FF6B6B", "#FFD700", "#87CEEB", "#90EE90"], text=[f"{v:.2f}" for v in values_b], textposition="outside")])
    fig_b.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Power (MW)")
    st.plotly_chart(fig_b, width='stretch')

# ─────────────────────────────────────────────────────────────────────
# VOLTAGE COMPARISON
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.subheader("⚡ Scenario A - Voltage Profile")
    fig_va = go.Figure()
    fig_va.add_hline(y=1.05, line_dash="dash", line_color="red")
    fig_va.add_hline(y=0.93, line_dash="dash", line_color="red")
    fig_va.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.3)
    fig_va.add_trace(go.Scatter(x=["Vmin", "Vmean", "Vmax"], y=[sim_a["v_min"], sim_a["v_mean"], sim_a["v_max"]], mode="lines+markers", line=dict(color="#4472C4", width=3), marker=dict(size=10)))
    fig_va.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Voltage (pu)")
    st.plotly_chart(fig_va, width='stretch')

with col_v2:
    st.subheader("⚡ Scenario B - Voltage Profile")
    fig_vb = go.Figure()
    fig_vb.add_hline(y=1.05, line_dash="dash", line_color="red")
    fig_vb.add_hline(y=0.93, line_dash="dash", line_color="red")
    fig_vb.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.3)
    fig_vb.add_trace(go.Scatter(x=["Vmin", "Vmean", "Vmax"], y=[sim_b["v_min"], sim_b["v_mean"], sim_b["v_max"]], mode="lines+markers", line=dict(color="#ED7D31", width=3), marker=dict(size=10)))
    fig_vb.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Voltage (pu)")
    st.plotly_chart(fig_vb, width='stretch')

# ─────────────────────────────────────────────────────────────────────
# RISK COMPARISON
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    st.subheader("🔴 Blackout Risk")
    fig_risk = go.Figure(data=[go.Bar(x=["Scenario A", "Scenario B"], y=[sim_a["blackout_risk"]*100, sim_b["blackout_risk"]*100], marker_color=[get_risk_color(sim_a["blackout_risk"]), get_risk_color(sim_b["blackout_risk"])], text=[f"{sim_a['blackout_risk']*100:.1f}%", f"{sim_b['blackout_risk']*100:.1f}%"], textposition="outside")])
    fig_risk.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Risk (%)")
    st.plotly_chart(fig_risk, width='stretch')

with col_r2:
    st.subheader("📊 Line Loading")
    fig_load = go.Figure(data=[go.Bar(x=["Scenario A", "Scenario B"], y=[sim_a["line_loading"], sim_b["line_loading"]], marker_color=["#90EE90" if sim_a["line_loading"]<80 else "#FFD700" if sim_a["line_loading"]<100 else "#FF6B6B", "#90EE90" if sim_b["line_loading"]<80 else "#FFD700" if sim_b["line_loading"]<100 else "#FF6B6B"], text=[f"{sim_a['line_loading']:.1f}%", f"{sim_b['line_loading']:.1f}%"], textposition="outside")])
    fig_load.add_hline(y=100, line_dash="dash", line_color="red")
    fig_load.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Loading (%)")
    st.plotly_chart(fig_load, width='stretch')

with col_r3:
    st.subheader("💨 Power Loss")
    fig_loss = go.Figure(data=[go.Bar(x=["Scenario A", "Scenario B"], y=[sim_a["p_loss"], sim_b["p_loss"]], marker_color=["#4472C4", "#ED7D31"], text=[f"{sim_a['p_loss']:.4f}", f"{sim_b['p_loss']:.4f}"], textposition="outside")])
    fig_loss.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Loss (MW)")
    st.plotly_chart(fig_loss, width='stretch')

# ─────────────────────────────────────────────────────────────────────
# OPTIONAL 24-HOUR PROFILE (LOAD ON DEMAND)
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("📈 Load 24-Hour Profile Analysis (slow - click to analyze)"):
    st.info("⏳ Calculating 24-hour profiles... please wait (~10-20 seconds)")

    @st.cache_data
    def simulate_24h_profile(p_load_base: float, p_solar_base: float, p_batt_base: float, use_zip: bool):
        probs_24 = []
        voltages_24 = []
        for h in np.arange(24):
            load_h = p_load_base + 0.3 * np.sin(h / 24 * 2 * np.pi)
            solar_h = max(0, p_solar_base * np.sin(h / 24 * np.pi))
            net = create_network(use_zip=use_zip)
            res = run_load_flow(net, load_h, solar_h, p_batt_base)
            line_loading = res["line_loading_max"]
            prob = predict_blackout_risk(load_h, solar_h, line_loading)
            probs_24.append(prob)
            voltages_24.append(res["v_min"])
        return probs_24, voltages_24

    probs_a_24, voltages_a_24 = simulate_24h_profile(sim_a["p_load"], sim_a["p_solar"], sim_a["p_battery"], sim_a["use_zip"])
    probs_b_24, voltages_b_24 = simulate_24h_profile(sim_b["p_load"], sim_b["p_solar"], sim_b["p_battery"], sim_b["use_zip"])

    hours = np.arange(24)
    col_24_1, col_24_2 = st.columns(2)

    with col_24_1:
        fig_24_risk = go.Figure()
        fig_24_risk.add_trace(go.Scatter(x=hours, y=[p*100 for p in probs_a_24], mode="lines", name="Scenario A", line=dict(color="#4472C4", width=3), fill="tozeroy"))
        fig_24_risk.add_trace(go.Scatter(x=hours, y=[p*100 for p in probs_b_24], mode="lines", name="Scenario B", line=dict(color="#ED7D31", width=3), fill="tozeroy"))
        fig_24_risk.update_layout(title="24-Hour Risk Profile", xaxis_title="Hour", yaxis_title="Risk (%)", height=350, hovermode="x unified")
        st.plotly_chart(fig_24_risk, width='stretch')

    with col_24_2:
        fig_24_volt = go.Figure()
        fig_24_volt.add_trace(go.Scatter(x=hours, y=voltages_a_24, mode="lines", name="Scenario A", line=dict(color="#4472C4", width=3)))
        fig_24_volt.add_trace(go.Scatter(x=hours, y=voltages_b_24, mode="lines", name="Scenario B", line=dict(color="#ED7D31", width=3)))
        fig_24_volt.add_hline(y=0.93, line_dash="dash", line_color="red")
        fig_24_volt.add_hline(y=1.05, line_dash="dash", line_color="red")
        fig_24_volt.update_layout(title="24-Hour Voltage Profile", xaxis_title="Hour", yaxis_title="Vmin (pu)", height=350, hovermode="x unified")
        st.plotly_chart(fig_24_volt, width='stretch')

# ─────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Detailed Comparison Table")

results_df = pd.DataFrame({
    "Metric": [
        "Load (MW)", "Solar (MW)", "Battery (MW)", "Net Load (MW)",
        "Vmin (pu)", "Vmean (pu)", "Vmax (pu)", "Voltage Violation",
        "Line Loading (%)", "Power Loss (MW)", "Blackout Risk (%)", "Converged"
    ],
    "Scenario A": [
        f"{sim_a['p_load']:.2f}", f"{sim_a['p_solar']:.2f}", f"{sim_a['p_battery']:.2f}", f"{sim_a['net_load']:.2f}",
        f"{sim_a['v_min']:.4f}", f"{sim_a['v_mean']:.4f}", f"{sim_a['v_max']:.4f}", "✗ No" if not sim_a["v_violation"] else "✓ Yes",
        f"{sim_a['line_loading']:.2f}", f"{sim_a['p_loss']:.4f}", f"{sim_a['blackout_risk']*100:.2f}", "✓" if sim_a["converged"] else "✗"
    ],
    "Scenario B": [
        f"{sim_b['p_load']:.2f}", f"{sim_b['p_solar']:.2f}", f"{sim_b['p_battery']:.2f}", f"{sim_b['net_load']:.2f}",
        f"{sim_b['v_min']:.4f}", f"{sim_b['v_mean']:.4f}", f"{sim_b['v_max']:.4f}", "✗ No" if not sim_b["v_violation"] else "✓ Yes",
        f"{sim_b['line_loading']:.2f}", f"{sim_b['p_loss']:.4f}", f"{sim_b['blackout_risk']*100:.2f}", "✓" if sim_b["converged"] else "✗"
    ],
})

st.dataframe(results_df, use_container_width=True)

st.markdown("---")
st.caption("⚡ Digital Twin Microgrid - Fast Comparison Dashboard | IEEE 33-bus Network")
