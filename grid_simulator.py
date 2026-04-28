"""
grid_simulator.py
=================
IEEE 33-bus AC load flow simulator for the Digital Twin Microgrid project.

Network facts:
  - 33 buses, 32 load buses (bus 1-32), slack bus at bus 0
  - Base load  : 3.715 MW  (sum of all 32 bus loads)
  - Base Q load: 2.300 MVAr
  - 37 distribution lines at 12.66 kV
  - PV injected at bus 17 (mid-feeder)

Per-timestep workflow:
  1. Scale all 32 bus loads proportionally to match p_load(t)
  2. Inject p_solar(t) as a static generator at bus 17
  3. Run AC Newton-Raphson load flow (pandapower)
  4. Extract voltage and line loading metrics
  5. Return results dict for label generation and ML features
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn


# ── Constants ─────────────────────────────────────────────────────────────────

IEEE33_BASE_LOAD_MW   = 3.715   # sum of all 32 bus loads in base network
IEEE33_BASE_LOAD_MVAR = 2.300   # sum of all 32 bus Q loads
PV_BUS                = 17      # mid-feeder bus for PV injection
V_MIN_LIMIT           = 0.93    # IEEE voltage lower bound (pu) At 0.93 pu, equipment performance degrades significantly."
V_MAX_LIMIT           = 1.05    # IEEE voltage upper bound (pu)

# ── Network Initialisation ────────────────────────────────────────────────────

def create_network() -> pp.pandapowerNet:
    """
    Load the IEEE 33-bus test network, store base loads, add PV sgen.

    Base loads are stored as private attributes so run_load_flow()
    always scales from the original values — not from already-scaled values.
    This prevents cumulative scaling errors across timesteps.

    PV sgen is initialised at 0 MW and updated each timestep in-place,
    avoiding the cost of adding/removing elements in the simulation loop.

    Returns
    -------
    pp.pandapowerNet
        Configured IEEE 33-bus network ready for simulation
    """
    net = pn.case33bw()

    # Store base loads
    net['_base_p_mw']   = net.load['p_mw'].values.copy()
    net['_base_q_mvar'] = net.load['q_mvar'].values.copy()

    # Fix unrealistic max_i_ka — set standard 12.66 kV conductor rating
    net.line['max_i_ka'] = 0.4   # 400A — standard for this network

    # Add PV sgen
    pp.create_sgen(net, bus=PV_BUS, p_mw=0.0, q_mvar=0.0,
                   name='PV_Plant', type='PV', in_service=True)
    return net

# ── Load Flow ─────────────────────────────────────────────────────────────────

def run_load_flow(
    net: pp.pandapowerNet,
    p_load_mw: float,
    p_solar_mw: float,
) -> dict:
    """
    Inject scaled load and solar, run AC load flow, return metrics.

    Load scaling:
      All 32 bus loads are scaled proportionally so their sum equals p_load_mw.
      Power factor is preserved — Q is scaled by the same ratio as P.
      Scaling always applied from the stored base values to avoid drift.

    PV injection:
      Solar power is injected as a static generator at bus 17.

    Parameters
    ----------
    net        : pp.pandapowerNet  IEEE 33-bus network (from create_network)
    p_load_mw  : float             Total load demand this timestep (MW)
    p_solar_mw : float             Total solar generation this timestep (MW)

    Returns
    -------
    dict with keys:
      converged        : bool   — did load flow converge?
      v_min            : float  — minimum bus voltage (pu)
      v_max            : float  — maximum bus voltage (pu)
      v_mean           : float  — mean bus voltage (pu)
      v_violation      : bool   — any bus below V_MIN_LIMIT?
      line_loading_max : float  — maximum line loading (%)
      p_loss_mw        : float  — total active power loss (MW)
    """
    # ── Scale loads proportionally from base values ────────────────────────
    scale = p_load_mw / IEEE33_BASE_LOAD_MW
    net.load['p_mw']   = net['_base_p_mw']   * scale
    net.load['q_mvar'] = net['_base_q_mvar']  * scale

    # ── Update PV injection ────────────────────────────────────────────────
    net.sgen.at[0, 'p_mw'] = p_solar_mw

    # ── Run AC Newton-Raphson load flow ────────────────────────────────────
    try:
        pp.runpp(net, algorithm='nr', numba=False, verbose=False)
        converged = True
    except pp.powerflow.LoadflowNotConverged:
        converged = False

    # ── Extract results ────────────────────────────────────────────────────
    if converged:
        vm_pu            = net.res_bus['vm_pu']
        v_min            = float(vm_pu.min())
        v_max            = float(vm_pu.max())
        v_mean           = float(vm_pu.mean())
        v_violation      = bool(v_min < V_MIN_LIMIT)
        loading = net.res_line['loading_percent']
        line_loading_max = float(loading.max()) if loading.notna().any() and loading.max() > 0 else float(
            (net.res_line['i_ka'] / net.line['max_i_ka']).max() * 100
            if 'max_i_ka' in net.line.columns and net.line['max_i_ka'].notna().any()
            else net.res_line['i_ka'].max() * 100   # raw current as proxy
        )
        p_loss_mw        = float(net.res_line['pl_mw'].sum())
    else:
        # Non-convergence → worst-case values to guarantee blackout label = 1
        v_min            = 0.0
        v_max            = 0.0
        v_mean           = 0.0
        v_violation      = True
        line_loading_max = 200.0
        p_loss_mw        = 999.0

    return {
        'converged':        converged,
        'v_min':            v_min,
        'v_max':            v_max,
        'v_mean':           v_mean,
        'v_violation':      v_violation,
        'line_loading_max': line_loading_max,
        'p_loss_mw':        p_loss_mw,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 55)
    print("Grid Simulator — Quick Test")
    print("=" * 55)

    net = create_network()
    print(f"Network loaded  : {len(net.bus)} buses, "
          f"{len(net.load)} loads, {len(net.line)} lines")
    print(f"Base load       : {net['_base_p_mw'].sum():.4f} MW")
    print(f"PV bus          : {PV_BUS}")
    print(f"Voltage limits  : [{V_MIN_LIMIT}, {V_MAX_LIMIT}] pu\n")

    # Test cases covering different operating conditions
    test_cases = [
        # (scenario label,           p_load, p_solar)
        ("Peak load, no solar",      3.715,  0.00),
        ("Peak load, full solar",    3.715,  1.50),
        ("Low load, full solar",     1.500,  1.50),
        ("Low load, no solar",       1.000,  0.00),
        ("Night, medium load",       2.500,  0.00),
        ("Partial load, partial PV", 2.000,  0.75),
    ]

    header = (f"{'Scenario':<28} | {'Vmin':>6} | {'Vmax':>6} | "
              f"{'Vmean':>6} | {'Line%':>6} | {'Ploss':>7} | "
              f"{'Conv':>4} | {'Vviol':>5}")
    print(header)
    print("-" * len(header))

    for label, p_load, p_solar in test_cases:
        r = run_load_flow(net, p_load, p_solar)
        print(
            f"{label:<28} | "
            f"{r['v_min']:>6.4f} | "
            f"{r['v_max']:>6.4f} | "
            f"{r['v_mean']:>6.4f} | "
            f"{r['line_loading_max']:>6.2f} | "
            f"{r['p_loss_mw']:>7.4f} | "
            f"{'Yes':>4} | "
            f"{'Yes' if r['v_violation'] else 'No':>5}"
        )

    print(f"\nVoltage violation limit : < {V_MIN_LIMIT} pu")
    print("Vviol = Yes → blackout label = 1 in label_generator.py")