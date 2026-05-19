"""
Digital Twin Microgrid - Simulation & Comparison Dashboard
===========================================================
Compare different microgrid scenarios with interactive parameters and real-time analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from grid_simulator import run_load_flow, create_network
import joblib

# Add caching decorator
st.set_page_config(page_title="Digital Twin Microgrid - Simulator", layout="wide")

# Load ML model
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
        "Scenario A": {
            "p_load": 2.5,
            "p_solar": 1.0,
            "p_battery": 0.0,
            "hour": 12,
            "use_zip": True,
        },
        "Scenario B": {
            "p_load": 3.5,
            "p_solar": 0.5,
            "p_battery": 0.3,
            "hour": 18,
            "use_zip": True,
        },
    }

# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def predict_blackout_risk(p_load: float, p_solar: float, line_loading: float, soc: float = 0.5) -> float:
    """Predict blackout risk probability using ML model"""
    X = pd.DataFrame([dict.fromkeys(feature_names, 0.0)])

    # Calculate derived features
    p_grid = p_load - p_solar  # Grid power = demand - generation

    # Set all available features
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

    # Fill lag features with current values (simplified - use current as proxy for lag)
    if "p_load_lag_1" in feature_names:
        X.loc[0, "p_load_lag_1"] = p_load
    if "p_load_lag_2" in feature_names:
        X.loc[0, "p_load_lag_2"] = p_load
    if "p_load_lag_3" in feature_names:
        X.loc[0, "p_load_lag_3"] = p_load
    if "p_solar_lag_1" in feature_names:
        X.loc[0, "p_solar_lag_1"] = p_solar
    if "p_solar_lag_2" in feature_names:
        X.loc[0, "p_solar_lag_2"] = p_solar
    if "p_solar_lag_3" in feature_names:
        X.loc[0, "p_solar_lag_3"] = p_solar

    # Fill any remaining with means
    for col in feature_names:
        if X.loc[0, col] == 0 and col in feature_means:
            X.loc[0, col] = feature_means[col]

    X = X[feature_names]
    prob = model.predict_proba(X)[0][1]
    return prob


@st.cache_data
def run_scenario_simulation(scenario_name: str, p_load: float, p_solar: float, p_battery: float, hour: int, use_zip: bool) -> dict:
    """Run full simulation for a scenario (cached)"""
    # Run load flow
    net = create_network(use_zip=use_zip)
    results = run_load_flow(net, p_load, p_solar, p_battery)

    # Predict blackout risk with line loading
    line_loading = results["line_loading_max"]
    blackout_risk = predict_blackout_risk(p_load, p_solar, line_loading)

    # Calculate metrics
    net_load = p_load - p_solar
    is_night = 1 if hour < 6 or hour > 18 else 0

    # Voltage stability indicator
    v_min = results["v_min"]
    if v_min < 0.93:
        v_status = "🔴 Critical"
    elif v_min < 0.95:
        v_status = "🟡 Low"
    else:
        v_status = "🟢 Stable"

    # Line loading indicator
    line_loading = results["line_loading_max"]
    if line_loading > 100:
        loading_status = "🔴 Overloaded"
    elif line_loading > 80:
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
        "line_loading": line_loading,
        "loading_status": loading_status,
        "p_loss": results["p_loss_mw"],
        "blackout_risk": blackout_risk,
        "zip_mode": results["zip_mode"],
    }


def get_risk_color(risk: float) -> str:
    """Get color based on risk level"""
    if risk < 0.3:
        return "#90EE90"  # Green
    elif risk < 0.7:
        return "#FFD700"  # Yellow
    else:
        return "#FF6B6B"  # Red


# ─────────────────────────────────────────────────────────────────────
# TITLE & DESCRIPTION
# ─────────────────────────────────────────────────────────────────────
st.title("⚡ Digital Twin Microgrid - Simulator")
st.markdown(
    "**Compare different microgrid operating scenarios with real-time physics simulation**"
)

# ─────────────────────────────────────────────────────────────────────
# SCENARIO CONFIGURATION SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎮 Scenario Controls")

    scenario_selection = st.selectbox(
        "Edit Scenario:",
        ["Scenario A", "Scenario B"],
        key="scenario_selector",
    )

    st.markdown(f"### {scenario_selection}")

    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 High Stress", use_container_width=True):
            st.session_state.scenarios[scenario_selection] = {
                "p_load": 3.8,
                "p_solar": 0.2,
                "p_battery": 0.0,
                "hour": 18,
                "use_zip": True,
            }
            st.rerun()
        if st.button("☀️ Solar Peak", use_container_width=True):
            st.session_state.scenarios[scenario_selection] = {
                "p_load": 2.0,
                "p_solar": 3.0,
                "p_battery": 0.0,
                "hour": 12,
                "use_zip": True,
            }
            st.rerun()
    with col2:
        if st.button("🌙 Night Peak", use_container_width=True):
            st.session_state.scenarios[scenario_selection] = {
                "p_load": 3.5,
                "p_solar": 0.0,
                "p_battery": 0.5,
                "hour": 21,
                "use_zip": True,
            }
            st.rerun()
        if st.button("⚖️ Balanced", use_container_width=True):
            st.session_state.scenarios[scenario_selection] = {
                "p_load": 2.5,
                "p_solar": 1.5,
                "p_battery": 0.3,
                "hour": 14,
                "use_zip": True,
            }
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Manual Controls")

    current_scenario = st.session_state.scenarios[scenario_selection]

    # Load slider
    p_load = st.slider(
        "💡 Load (MW)",
        min_value=0.5,
        max_value=4.0,
        value=current_scenario["p_load"],
        step=0.1,
        key=f"{scenario_selection}_load",
    )

    # Solar slider
    p_solar = st.slider(
        "☀️ Solar (MW)",
        min_value=0.0,
        max_value=3.5,
        value=current_scenario["p_solar"],
        step=0.1,
        key=f"{scenario_selection}_solar",
    )

    # Battery slider
    p_battery = st.slider(
        "🔋 Battery (MW)",
        min_value=0.0,
        max_value=1.0,
        value=current_scenario["p_battery"],
        step=0.1,
        key=f"{scenario_selection}_battery",
    )

    # Hour slider
    hour = st.slider(
        "🕐 Hour of Day",
        min_value=0,
        max_value=23,
        value=current_scenario["hour"],
        step=1,
        key=f"{scenario_selection}_hour",
    )

    # ZIP model toggle
    use_zip = st.checkbox(
        "Enable ZIP Load Model",
        value=current_scenario["use_zip"],
        key=f"{scenario_selection}_zip",
        help="Voltage-dependent load model (realistic)",
    )

    # Update session state
    st.session_state.scenarios[scenario_selection] = {
        "p_load": p_load,
        "p_solar": p_solar,
        "p_battery": p_battery,
        "hour": hour,
        "use_zip": use_zip,
    }

# ─────────────────────────────────────────────────────────────────────
# RUN SIMULATIONS
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
# TOP ROW: KEY METRICS COMPARISON
# ─────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Key Metrics Comparison")

metric_cols = st.columns(5)

with metric_cols[0]:
    st.metric(
        "Scenario A - Load",
        f"{sim_a['p_load']:.2f} MW",
        f"Net: {sim_a['net_load']:.2f} MW",
    )
    st.metric(
        "Scenario B - Load",
        f"{sim_b['p_load']:.2f} MW",
        f"Net: {sim_b['net_load']:.2f} MW",
    )

with metric_cols[1]:
    st.metric(
        "Scenario A - Solar",
        f"{sim_a['p_solar']:.2f} MW",
        f"Battery: {sim_a['p_battery']:.2f} MW",
    )
    st.metric(
        "Scenario B - Solar",
        f"{sim_b['p_solar']:.2f} MW",
        f"Battery: {sim_b['p_battery']:.2f} MW",
    )

with metric_cols[2]:
    st.metric(
        "Scenario A - Voltage (pu)",
        f"{sim_a['v_min']:.3f}",
        sim_a["v_status"],
    )
    st.metric(
        "Scenario B - Voltage (pu)",
        f"{sim_b['v_min']:.3f}",
        sim_b["v_status"],
    )

with metric_cols[3]:
    st.metric(
        "Scenario A - Line Loading",
        f"{sim_a['line_loading']:.1f}%",
        sim_a["loading_status"],
    )
    st.metric(
        "Scenario B - Line Loading",
        f"{sim_b['line_loading']:.1f}%",
        sim_b["loading_status"],
    )

with metric_cols[4]:
    st.metric(
        "Scenario A - Blackout Risk",
        f"{sim_a['blackout_risk']:.1%}",
        delta=f"{(sim_a['blackout_risk']-0.5)*100:+.1f}%",
    )
    st.metric(
        "Scenario B - Blackout Risk",
        f"{sim_b['blackout_risk']:.1%}",
        delta=f"{(sim_b['blackout_risk']-0.5)*100:+.1f}%",
    )

# ─────────────────────────────────────────────────────────────────────
# VISUALIZATION: Energy Flow Comparison
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col_sankey1, col_sankey2 = st.columns(2)

with col_sankey1:
    st.subheader("📍 Scenario A - Energy Flow")

    # Sankey diagram for Scenario A
    scenarios_flow_a = ["Load", "Solar", "Battery", "Grid"]
    values_a = [
        sim_a["p_load"],
        sim_a["p_solar"],
        sim_a["p_battery"],
        max(0, sim_a["p_load"] - sim_a["p_solar"] - sim_a["p_battery"]),
    ]

    fig_sankey_a = go.Figure(
        data=[
            go.Bar(
                x=["Load", "Solar", "Battery", "Grid"],
                y=values_a,
                marker_color=["#FF6B6B", "#FFD700", "#87CEEB", "#90EE90"],
                text=[f"{v:.2f} MW" for v in values_a],
                textposition="outside",
            )
        ]
    )
    fig_sankey_a.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis_title="Power (MW)",
    )
    st.plotly_chart(fig_sankey_a, use_container_width=True)

with col_sankey2:
    st.subheader("📍 Scenario B - Energy Flow")

    scenarios_flow_b = ["Load", "Solar", "Battery", "Grid"]
    values_b = [
        sim_b["p_load"],
        sim_b["p_solar"],
        sim_b["p_battery"],
        max(0, sim_b["p_load"] - sim_b["p_solar"] - sim_b["p_battery"]),
    ]

    fig_sankey_b = go.Figure(
        data=[
            go.Bar(
                x=["Load", "Solar", "Battery", "Grid"],
                y=values_b,
                marker_color=["#FF6B6B", "#FFD700", "#87CEEB", "#90EE90"],
                text=[f"{v:.2f} MW" for v in values_b],
                textposition="outside",
            )
        ]
    )
    fig_sankey_b.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis_title="Power (MW)",
    )
    st.plotly_chart(fig_sankey_b, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# VOLTAGE PROFILE COMPARISON
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚡ Voltage Stability Analysis")

col_volt1, col_volt2 = st.columns(2)

with col_volt1:
    fig_volt_a = go.Figure()

    # Range zones
    fig_volt_a.add_hline(
        y=1.05, line_dash="dash", line_color="red", annotation_text="Max (1.05 pu)"
    )
    fig_volt_a.add_hline(
        y=0.93, line_dash="dash", line_color="red", annotation_text="Min (0.93 pu)"
    )
    fig_volt_a.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_volt_a.add_trace(
        go.Scatter(
            x=["Vmin", "Vmean", "Vmax"],
            y=[sim_a["v_min"], sim_a["v_mean"], sim_a["v_max"]],
            mode="lines+markers",
            name="Scenario A",
            line=dict(color="#4472C4", width=3),
            marker=dict(size=10),
        )
    )

    fig_volt_a.update_layout(
        title="Scenario A - Voltage Profile",
        yaxis_title="Voltage (pu)",
        height=350,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_volt_a, use_container_width=True)

with col_volt2:
    fig_volt_b = go.Figure()

    # Range zones
    fig_volt_b.add_hline(
        y=1.05, line_dash="dash", line_color="red", annotation_text="Max (1.05 pu)"
    )
    fig_volt_b.add_hline(
        y=0.93, line_dash="dash", line_color="red", annotation_text="Min (0.93 pu)"
    )
    fig_volt_b.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_volt_b.add_trace(
        go.Scatter(
            x=["Vmin", "Vmean", "Vmax"],
            y=[sim_b["v_min"], sim_b["v_mean"], sim_b["v_max"]],
            mode="lines+markers",
            name="Scenario B",
            line=dict(color="#ED7D31", width=3),
            marker=dict(size=10),
        )
    )

    fig_volt_b.update_layout(
        title="Scenario B - Voltage Profile",
        yaxis_title="Voltage (pu)",
        height=350,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_volt_b, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# RISK METRICS COMPARISON
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔴 Risk Assessment")

col_risk1, col_risk2, col_risk3 = st.columns(3)

with col_risk1:
    st.markdown("**Blackout Risk Comparison**")

    risk_data = {
        "Scenario": ["A", "B"],
        "Risk %": [sim_a["blackout_risk"] * 100, sim_b["blackout_risk"] * 100],
    }
    risk_df = pd.DataFrame(risk_data)

    fig_risk = go.Figure(
        data=[
            go.Bar(
                x=["Scenario A", "Scenario B"],
                y=[sim_a["blackout_risk"] * 100, sim_b["blackout_risk"] * 100],
                marker_color=[
                    get_risk_color(sim_a["blackout_risk"]),
                    get_risk_color(sim_b["blackout_risk"]),
                ],
                text=[
                    f"{sim_a['blackout_risk']*100:.1f}%",
                    f"{sim_b['blackout_risk']*100:.1f}%",
                ],
                textposition="outside",
            )
        ]
    )
    fig_risk.update_layout(
        yaxis_title="Risk (%)",
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with col_risk2:
    st.markdown("**Line Loading Comparison**")

    fig_loading = go.Figure(
        data=[
            go.Bar(
                x=["Scenario A", "Scenario B"],
                y=[sim_a["line_loading"], sim_b["line_loading"]],
                marker_color=[
                    "#90EE90" if sim_a["line_loading"] < 80 else "#FFD700"
                    if sim_a["line_loading"] < 100
                    else "#FF6B6B",
                    "#90EE90" if sim_b["line_loading"] < 80 else "#FFD700"
                    if sim_b["line_loading"] < 100
                    else "#FF6B6B",
                ],
                text=[f"{sim_a['line_loading']:.1f}%", f"{sim_b['line_loading']:.1f}%"],
                textposition="outside",
            )
        ]
    )
    fig_loading.update_layout(
        yaxis_title="Loading (%)",
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig_loading.add_hline(y=100, line_dash="dash", line_color="red")
    st.plotly_chart(fig_loading, use_container_width=True)

with col_risk3:
    st.markdown("**Power Loss Comparison**")

    fig_loss = go.Figure(
        data=[
            go.Bar(
                x=["Scenario A", "Scenario B"],
                y=[sim_a["p_loss"], sim_b["p_loss"]],
                marker_color=["#4472C4", "#ED7D31"],
                text=[f"{sim_a['p_loss']:.3f} MW", f"{sim_b['p_loss']:.3f} MW"],
                textposition="outside",
            )
        ]
    )
    fig_loss.update_layout(
        yaxis_title="Loss (MW)",
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_loss, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# 24-HOUR PROFILE SIMULATION
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 24-Hour Profile Simulation")

hours = np.arange(24)

@st.cache_data
def simulate_24h_profile(p_load_base: float, p_solar_base: float, p_batt_base: float, use_zip: bool) -> tuple:
    """Simulate 24-hour profile (cached for speed)"""
    hours = np.arange(24)
    probs_24 = []
    voltages_24 = []

    for h in hours:
        load_h = p_load_base + 0.3 * np.sin(h / 24 * 2 * np.pi)
        solar_h = max(0, p_solar_base * np.sin(h / 24 * np.pi))

        net = create_network(use_zip=use_zip)
        res = run_load_flow(net, load_h, solar_h, p_batt_base)

        # Get line loading for risk calculation
        line_loading = res["line_loading_max"]
        prob = predict_blackout_risk(load_h, solar_h, line_loading)

        probs_24.append(prob)
        voltages_24.append(res["v_min"])

    return probs_24, voltages_24


# Get 24-hour profiles
probs_a_24, voltages_a_24 = simulate_24h_profile(
    st.session_state.scenarios["Scenario A"]["p_load"],
    st.session_state.scenarios["Scenario A"]["p_solar"],
    st.session_state.scenarios["Scenario A"]["p_battery"],
    st.session_state.scenarios["Scenario A"]["use_zip"],
)

probs_b_24, voltages_b_24 = simulate_24h_profile(
    st.session_state.scenarios["Scenario B"]["p_load"],
    st.session_state.scenarios["Scenario B"]["p_solar"],
    st.session_state.scenarios["Scenario B"]["p_battery"],
    st.session_state.scenarios["Scenario B"]["use_zip"],
)

col_24h1, col_24h2 = st.columns(2)

with col_24h1:
    fig_24h_risk = go.Figure()

    fig_24h_risk.add_trace(
        go.Scatter(
            x=hours,
            y=[p * 100 for p in probs_a_24],
            mode="lines",
            name="Scenario A",
            line=dict(color="#4472C4", width=3),
            fill="tozeroy",
        )
    )

    fig_24h_risk.add_trace(
        go.Scatter(
            x=hours,
            y=[p * 100 for p in probs_b_24],
            mode="lines",
            name="Scenario B",
            line=dict(color="#ED7D31", width=3),
            fill="tozeroy",
        )
    )

    fig_24h_risk.update_layout(
        title="24-Hour Blackout Risk Profile",
        xaxis_title="Hour of Day",
        yaxis_title="Risk (%)",
        height=350,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_24h_risk, use_container_width=True)

with col_24h2:
    fig_24h_volt = go.Figure()

    fig_24h_volt.add_trace(
        go.Scatter(
            x=hours,
            y=voltages_a_24,
            mode="lines",
            name="Scenario A",
            line=dict(color="#4472C4", width=3),
        )
    )

    fig_24h_volt.add_trace(
        go.Scatter(
            x=hours,
            y=voltages_b_24,
            mode="lines",
            name="Scenario B",
            line=dict(color="#ED7D31", width=3),
        )
    )

    fig_24h_volt.add_hline(y=0.93, line_dash="dash", line_color="red")
    fig_24h_volt.add_hline(y=1.05, line_dash="dash", line_color="red")

    fig_24h_volt.update_layout(
        title="24-Hour Voltage (Vmin) Profile",
        xaxis_title="Hour of Day",
        yaxis_title="Voltage (pu)",
        height=350,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_24h_volt, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# DETAILED SIMULATION RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Detailed Results Comparison")

results_comparison = pd.DataFrame(
    {
        "Metric": [
            "Load (MW)",
            "Solar (MW)",
            "Battery (MW)",
            "Net Load (MW)",
            "Converged",
            "Vmin (pu)",
            "Vmean (pu)",
            "Vmax (pu)",
            "Voltage Violation",
            "Line Loading (%)",
            "Power Loss (MW)",
            "Blackout Risk (%)",
            "ZIP Model",
        ],
        "Scenario A": [
            f"{sim_a['p_load']:.2f}",
            f"{sim_a['p_solar']:.2f}",
            f"{sim_a['p_battery']:.2f}",
            f"{sim_a['net_load']:.2f}",
            "✓" if sim_a["converged"] else "✗",
            f"{sim_a['v_min']:.4f}",
            f"{sim_a['v_mean']:.4f}",
            f"{sim_a['v_max']:.4f}",
            "✗ No" if not sim_a["v_violation"] else "✓ Yes",
            f"{sim_a['line_loading']:.2f}",
            f"{sim_a['p_loss']:.4f}",
            f"{sim_a['blackout_risk']*100:.2f}",
            "✓" if sim_a["zip_mode"] else "✗",
        ],
        "Scenario B": [
            f"{sim_b['p_load']:.2f}",
            f"{sim_b['p_solar']:.2f}",
            f"{sim_b['p_battery']:.2f}",
            f"{sim_b['net_load']:.2f}",
            "✓" if sim_b["converged"] else "✗",
            f"{sim_b['v_min']:.4f}",
            f"{sim_b['v_mean']:.4f}",
            f"{sim_b['v_max']:.4f}",
            "✗ No" if not sim_b["v_violation"] else "✓ Yes",
            f"{sim_b['line_loading']:.2f}",
            f"{sim_b['p_loss']:.4f}",
            f"{sim_b['blackout_risk']*100:.2f}",
            "✓" if sim_b["zip_mode"] else "✗",
        ],
    }
)

st.dataframe(results_comparison, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Digital Twin Microgrid Simulator | IEEE 33-bus Network | ZIP Load Model | ML Risk Prediction"
)
