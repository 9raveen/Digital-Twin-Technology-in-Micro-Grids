"""
label_generator.py
==================
Physics-based blackout label generator for the Digital Twin Microgrid project.

Generates binary blackout labels from power system physics with realistic noise.
These labels serve as ground truth for ML classifier training.

Blackout condition:
  label = 1  if  v_min_measured < V_BLACKOUT_THRESHOLD  (voltage violation)
  label = 0  otherwise

Measurement realism:
  - v_min_measured = v_min_ideal + noise  (Gaussian, σ=0.007 pu)
  - Simulates realistic sensor error in field measurement devices
  - Breaks artificial deterministic relationship between features and labels

Threshold choice:
  V_BLACKOUT_THRESHOLD = 0.93 pu
  - Gives ~20-25% blackout rate across the 30-day simulation
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

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

V_BLACKOUT_THRESHOLD = 0.93    # pu — primary blackout trigger

# Measurement noise (realistic sensor error)
VOLTAGE_NOISE_STD = 0.007      # pu (±0.7%) — typical phasor measurement unit accuracy

# Severity bands (for analysis only — not used in ML labels)
V_CRITICAL  = 0.92
V_HIGH      = 0.93
V_MODERATE  = 0.94

# ── Label Generation ──────────────────────────────────────────────────────────

def compute_risk_score(v_min: float, converged: bool) -> float:
    """
    Compute smooth risk score using sigmoid function.

    Returns value in [0, 1] representing blackout probability:
    - 0 → safe operation
    - 1 → imminent blackout (non-convergence or critical voltage)

    Parameters
    ----------
    v_min      : float  Minimum bus voltage (pu)
    converged  : bool   Did the load flow converge?

    Returns
    -------
    float
        Risk score in [0, 1]
    """
    if not converged:
        return 1.0

    # Sigmoid-based smooth risk: k controls steepness at threshold
    k = 60
    return 1.0 / (1.0 + np.exp(k * (v_min - 0.93)))


def generate_label(v_min: float, converged: bool, add_noise: bool = True) -> int:
    """
    Generate binary blackout label for a single timestep.

    Parameters
    ----------
    v_min      : float  Minimum bus voltage this timestep (pu) or v_min_measured
    converged  : bool   Did the load flow converge?
    add_noise  : bool   Unused (kept for API compatibility) - noise applied before this call

    Returns
    -------
    int
        1 → blackout (voltage violation or non-convergence)
        0 → normal operation

    Notes
    -----
    Measurement noise is applied BEFORE this function (in digital_twin.py).
    This ensures noise is deterministic per sample and baked into the dataset,
    not re-sampled on every function call.
    """
    # Non-convergence is always a blackout (system collapsed)
    if not converged:
        return 1

    # Voltage below operational threshold (v_min already includes measurement noise)
    if v_min < V_BLACKOUT_THRESHOLD:
        return 1

    return 0


def generate_label_with_risk(v_min: float, converged: bool) -> tuple:
    """
    Generate both binary blackout label and risk score.

    Returns tuple (blackout, risk_score) for dual-target training.

    Parameters
    ----------
    v_min      : float  Minimum bus voltage (pu)
    converged  : bool   Did the load flow converge?

    Returns
    -------
    tuple
        (blackout: int, risk_score: float)
    """
    blackout = generate_label(v_min, converged)
    risk = compute_risk_score(v_min, converged)
    return blackout, risk


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