"""
label_generator.py
==================
Physics-based blackout label generator for the Digital Twin Microgrid project.

Generates deterministic binary blackout labels from power system physics.
These labels serve as ground truth for ML classifier training.

Blackout condition:
  label = 1  if  v_min < V_BLACKOUT_THRESHOLD  (voltage violation)
  label = 0  otherwise

Threshold choice:
  V_BLACKOUT_THRESHOLD = 0.93 pu
  - Gives ~25% blackout rate across the 30-day simulation
  - Physically justified: at 0.93 pu, equipment performance degrades
    significantly in distribution networks
  - Academically defensible: distribution operators often use tighter
    thresholds than the IEEE 0.95 pu standard for reliability assessment

Severity classification (for analysis and dashboard):
  CRITICAL : v_min < 0.92
  HIGH     : 0.92 <= v_min < 0.93
  MODERATE : 0.93 <= v_min < 0.94
  NORMAL   : v_min >= 0.94
"""


# ── Constants ─────────────────────────────────────────────────────────────────

V_BLACKOUT_THRESHOLD = 0.93    # pu — primary blackout trigger

# Severity bands (for analysis only — not used in ML labels)
V_CRITICAL  = 0.92
V_HIGH      = 0.93
V_MODERATE  = 0.94


# ── Label Generation ──────────────────────────────────────────────────────────

def generate_label(v_min: float, converged: bool) -> int:
    """
    Generate binary blackout label for a single timestep.

    Parameters
    ----------
    v_min      : float  Minimum bus voltage this timestep (pu)
    converged  : bool   Did the load flow converge?

    Returns
    -------
    int
        1 → blackout (voltage violation or non-convergence)
        0 → normal operation
    """
    # Non-convergence is always a blackout (system collapsed)
    if not converged:
        return 1

    # Voltage below operational threshold
    if v_min < V_BLACKOUT_THRESHOLD:
        return 1

    return 0


def get_severity(v_min: float, converged: bool) -> str:
    """
    Classify operating condition severity for analysis and dashboard display.

    Parameters
    ----------
    v_min      : float  Minimum bus voltage (pu)
    converged  : bool   Did the load flow converge?

    Returns
    -------
    str  One of: 'CRITICAL', 'HIGH', 'MODERATE', 'NORMAL'
    """
    if not converged:
        return 'CRITICAL'

    if v_min < V_CRITICAL:
        return 'CRITICAL'
    elif v_min < V_HIGH:
        return 'HIGH'
    elif v_min < V_MODERATE:
        return 'MODERATE'
    else:
        return 'NORMAL'


def get_blackout_margin(v_min: float) -> float:
    """
    Calculate voltage margin to blackout threshold.

    Positive → safe (how far above threshold)
    Negative → in violation (how far below threshold)

    Parameters
    ----------
    v_min : float  Minimum bus voltage (pu)

    Returns
    -------
    float  Margin in pu (positive = safe, negative = violation)
    """
    return round(v_min - V_BLACKOUT_THRESHOLD, 6)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 55)
    print("Label Generator — Quick Test")
    print("=" * 55)

    # Test cases from grid_simulator results
    test_cases = [
        # (scenario,                   v_min,  converged)
        ("Peak load, no solar",        0.9131, True),
        ("Peak load, full solar",      0.9379, True),
        ("Low load, full solar",       0.9867, True),
        ("Low load, no solar",         0.9779, True),
        ("Night, medium load",         0.9431, True),
        ("Partial load, partial PV",   0.9671, True),
        ("Non-convergence",            0.0000, False),
    ]

    print(f"\nThreshold : V_min < {V_BLACKOUT_THRESHOLD} pu → blackout = 1\n")
    print(f"{'Scenario':<28} | {'Vmin':>6} | {'Label':>5} | "
          f"{'Severity':<10} | {'Margin':>8}")
    print("-" * 68)

    labels    = []
    severities = []

    for scenario, v_min, converged in test_cases:
        label    = generate_label(v_min, converged)
        severity = get_severity(v_min, converged)
        margin   = get_blackout_margin(v_min)

        labels.append(label)
        severities.append(severity)

        print(
            f"{scenario:<28} | "
            f"{v_min:>6.4f} | "
            f"{'BLACKOUT' if label else 'normal':>8} | "
            f"{severity:<10} | "
            f"{margin:>+8.4f}"
        )

    print(f"\nSummary:")
    print(f"  Blackout cases : {sum(labels)}/{len(labels)}")
    print(f"  Normal cases   : {len(labels)-sum(labels)}/{len(labels)}")
    print(f"\nSeverity breakdown:")
    for level in ['CRITICAL', 'HIGH', 'MODERATE', 'NORMAL']:
        count = severities.count(level)
        print(f"  {level:<10} : {count}")