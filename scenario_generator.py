"""
scenario_generator.py
=====================
Scenario parameter generation for diverse dataset creation.

Generates randomized but physically realistic scenarios for multi-run simulations.
Each scenario defines battery parameters, grid conditions, and stress injection rules.

Scenario Types:
  NORMAL    - Light to moderate load, good solar, healthy battery
  STRESSED  - High load, reduced solar, degraded battery
  EXTREME   - Peak load, minimal solar, severely limited battery
  SEASONAL  - Winter/summer patterns with nominal battery
  RANDOM    - Fully randomized parameters
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import json


class ScenarioType(Enum):
    """Scenario classification for dataset diversity."""
    NORMAL = "normal"
    STRESSED = "stressed"
    EXTREME = "extreme"
    SEASONAL = "seasonal"
    RANDOM = "random"


@dataclass
class ScenarioConfig:
    """Configuration for a single 720-hour scenario."""

    # Scenario metadata
    scenario_id: int
    scenario_type: ScenarioType
    seed: int

    # Battery parameters
    battery_capacity_mwh: float
    battery_soc_init: float
    battery_discharge_rate_mw: float
    battery_charge_rate_mw: float
    battery_efficiency: float
    battery_response_factor: float

    # Grid parameters
    load_scaling: float              # Multiplier on base load (e.g., 0.8 to 1.5)
    solar_capacity_mw: float         # Max solar generation available
    grid_import_limit_mw: float      # Max grid import constraint

    # Time series selection
    load_window_start_hour: int      # Start hour in raw load CSV
    solar_window_start_hour: int     # Start hour in raw solar CSV

    # Perturbations
    load_noise_std: float            # Gaussian noise on load (fraction of value)
    solar_intermittency_factor: float # Cloud simulation (0 = none, 0.3 = high)

    # Stress injection
    load_spike_count: int            # Number of demand spikes to inject
    load_spike_magnitude: float      # Multiplier on demand during spike (e.g., 1.5x)
    solar_dip_count: int             # Number of solar drops to inject
    solar_dip_magnitude: float       # Reduction factor during dip (e.g., 0.3x)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'scenario_id': self.scenario_id,
            'scenario_type': self.scenario_type.value,
            'seed': self.seed,
            'battery_capacity_mwh': self.battery_capacity_mwh,
            'battery_soc_init': self.battery_soc_init,
            'battery_discharge_rate_mw': self.battery_discharge_rate_mw,
            'battery_charge_rate_mw': self.battery_charge_rate_mw,
            'battery_efficiency': self.battery_efficiency,
            'battery_response_factor': self.battery_response_factor,
            'load_scaling': self.load_scaling,
            'solar_capacity_mw': self.solar_capacity_mw,
            'grid_import_limit_mw': self.grid_import_limit_mw,
            'load_window_start_hour': self.load_window_start_hour,
            'solar_window_start_hour': self.solar_window_start_hour,
            'load_noise_std': self.load_noise_std,
            'solar_intermittency_factor': self.solar_intermittency_factor,
            'load_spike_count': self.load_spike_count,
            'load_spike_magnitude': self.load_spike_magnitude,
            'solar_dip_count': self.solar_dip_count,
            'solar_dip_magnitude': self.solar_dip_magnitude,
        }


def sample_scenario_parameters(
    scenario_type: ScenarioType,
    scenario_id: int,
    seed: int,
    n_available_hours: int = 8760
) -> ScenarioConfig:
    """
    Sample parameters for a scenario type.

    Parameters
    ----------
    scenario_type : ScenarioType
        Type of scenario to generate
    scenario_id : int
        Unique ID for this scenario
    seed : int
        Random seed for reproducibility
    n_available_hours : int
        Total hours available in load/solar CSVs (default: 1 year = 8760)

    Returns
    -------
    ScenarioConfig
        Randomized but physically realistic scenario configuration
    """
    np.random.seed(seed)
    window_size = 720

    # Validate that we have enough data
    max_start_hour = max(0, n_available_hours - window_size - 1)

    if scenario_type == ScenarioType.NORMAL:
        # Light to moderate load, good solar, healthy battery
        battery_capacity = np.random.uniform(0.4, 0.7)
        discharge_rate = np.random.uniform(0.20, 0.30)
        charge_rate = np.random.uniform(0.20, 0.30)
        efficiency = np.random.uniform(0.93, 0.96)
        response_factor = np.random.uniform(0.4, 0.6)

        load_scaling = np.random.uniform(0.8, 1.2)
        solar_capacity = np.random.uniform(2.5, 3.5)
        grid_import_limit = 5.0

        load_noise_std = np.random.uniform(0.01, 0.03)
        solar_intermittency = np.random.uniform(0.0, 0.1)

        load_spike_count = 0
        solar_dip_count = 0

    elif scenario_type == ScenarioType.STRESSED:
        # High load, reduced solar, degraded battery
        battery_capacity = np.random.uniform(0.25, 0.45)
        discharge_rate = np.random.uniform(0.10, 0.20)
        charge_rate = np.random.uniform(0.15, 0.25)
        efficiency = np.random.uniform(0.90, 0.94)
        response_factor = np.random.uniform(0.3, 0.5)

        load_scaling = np.random.uniform(1.1, 1.5)
        solar_capacity = np.random.uniform(0.5, 1.5)
        grid_import_limit = np.random.uniform(4.0, 5.5)

        load_noise_std = np.random.uniform(0.02, 0.04)
        solar_intermittency = np.random.uniform(0.1, 0.25)

        load_spike_count = np.random.randint(1, 4)
        solar_dip_count = np.random.randint(1, 3)

    elif scenario_type == ScenarioType.EXTREME:
        # Peak load, minimal solar, severely limited battery
        battery_capacity = np.random.uniform(0.15, 0.30)
        discharge_rate = np.random.uniform(0.05, 0.10)
        charge_rate = np.random.uniform(0.10, 0.15)
        efficiency = np.random.uniform(0.85, 0.92)
        response_factor = np.random.uniform(0.15, 0.35)

        load_scaling = np.random.uniform(1.3, 1.6)
        solar_capacity = np.random.uniform(0.2, 0.8)
        grid_import_limit = np.random.uniform(3.0, 4.5)

        load_noise_std = np.random.uniform(0.02, 0.05)
        solar_intermittency = np.random.uniform(0.2, 0.35)

        load_spike_count = np.random.randint(2, 6)
        solar_dip_count = np.random.randint(2, 5)

    elif scenario_type == ScenarioType.SEASONAL:
        # Winter/summer alternation with nominal battery
        battery_capacity = np.random.uniform(0.4, 0.7)
        discharge_rate = np.random.uniform(0.20, 0.30)
        charge_rate = np.random.uniform(0.20, 0.30)
        efficiency = np.random.uniform(0.93, 0.96)
        response_factor = np.random.uniform(0.4, 0.6)

        # Alternate between winter (high load, low solar) and summer patterns
        is_winter = scenario_id % 2 == 0
        if is_winter:
            load_scaling = np.random.uniform(1.1, 1.4)
            solar_capacity = np.random.uniform(0.5, 1.2)
        else:
            load_scaling = np.random.uniform(0.7, 0.95)
            solar_capacity = np.random.uniform(2.8, 3.5)

        grid_import_limit = 5.0

        load_noise_std = np.random.uniform(0.01, 0.03)
        solar_intermittency = np.random.uniform(0.05, 0.15)

        load_spike_count = np.random.randint(0, 2)
        solar_dip_count = np.random.randint(0, 2)

    else:  # RANDOM
        # Fully randomized within safe bounds
        battery_capacity = np.random.uniform(0.15, 0.7)
        discharge_rate = np.random.uniform(0.05, 0.30)
        charge_rate = np.random.uniform(0.10, 0.30)
        efficiency = np.random.uniform(0.85, 0.96)
        response_factor = np.random.uniform(0.15, 0.6)

        load_scaling = np.random.uniform(0.7, 1.6)
        solar_capacity = np.random.uniform(0.2, 3.5)
        grid_import_limit = np.random.uniform(3.0, 5.5)

        load_noise_std = np.random.uniform(0.01, 0.05)
        solar_intermittency = np.random.uniform(0.0, 0.35)

        load_spike_count = np.random.randint(0, 6)
        solar_dip_count = np.random.randint(0, 5)

    # Always keep: soc_init, charge_rate, spike magnitudes in safe ranges
    soc_init = np.random.uniform(0.3, 0.7)
    load_spike_magnitude = np.random.uniform(1.3, 1.8)
    solar_dip_magnitude = np.random.uniform(0.1, 0.4)

    # Ensure solar capacity is at least 0.2 MW (physically realistic)
    solar_capacity = max(solar_capacity, 0.2)

    # Window selection (non-overlapping preferred)
    load_window_start = np.random.randint(0, max(1, max_start_hour))
    solar_window_start = np.random.randint(0, max(1, max_start_hour))

    return ScenarioConfig(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        seed=seed,
        battery_capacity_mwh=battery_capacity,
        battery_soc_init=soc_init,
        battery_discharge_rate_mw=discharge_rate,
        battery_charge_rate_mw=charge_rate,
        battery_efficiency=efficiency,
        battery_response_factor=response_factor,
        load_scaling=load_scaling,
        solar_capacity_mw=solar_capacity,
        grid_import_limit_mw=grid_import_limit,
        load_window_start_hour=load_window_start,
        solar_window_start_hour=solar_window_start,
        load_noise_std=load_noise_std,
        solar_intermittency_factor=solar_intermittency,
        load_spike_count=load_spike_count,
        load_spike_magnitude=load_spike_magnitude,
        solar_dip_count=solar_dip_count,
        solar_dip_magnitude=solar_dip_magnitude,
    )


def create_scenario_distribution(
    n_total: int,
    distribution: Dict[ScenarioType, float] = None,
    seed: int = 42
) -> List[ScenarioConfig]:
    """
    Generate list of N scenarios with given type distribution.

    Parameters
    ----------
    n_total : int
        Total number of scenarios to generate
    distribution : dict, optional
        Fraction for each scenario type. Default: {NORMAL: 0.5, STRESSED: 0.3, EXTREME: 0.15, SEASONAL: 0.05}
    seed : int
        Random seed for reproducibility

    Returns
    -------
    List[ScenarioConfig]
        List of N scenario configurations
    """
    if distribution is None:
        distribution = {
            ScenarioType.NORMAL: 0.5,
            ScenarioType.STRESSED: 0.3,
            ScenarioType.EXTREME: 0.15,
            ScenarioType.SEASONAL: 0.05,
        }

    # Validate distribution
    assert abs(sum(distribution.values()) - 1.0) < 0.01, \
        f"Distribution must sum to 1.0, got {sum(distribution.values())}"

    scenarios = []
    scenario_id = 0

    for scenario_type, fraction in distribution.items():
        n_this_type = int(n_total * fraction)

        for i in range(n_this_type):
            # Unique seed for each scenario
            scenario_seed = seed + scenario_id

            config = sample_scenario_parameters(
                scenario_type=scenario_type,
                scenario_id=scenario_id,
                seed=scenario_seed
            )

            scenarios.append(config)
            scenario_id += 1

    # Handle rounding: add remaining scenarios to NORMAL type
    while len(scenarios) < n_total:
        config = sample_scenario_parameters(
            scenario_type=ScenarioType.NORMAL,
            scenario_id=scenario_id,
            seed=seed + scenario_id
        )
        scenarios.append(config)
        scenario_id += 1

    return scenarios


def save_scenarios(
    scenarios: List[ScenarioConfig],
    output_path: str
) -> None:
    """Save scenario configurations to JSON."""
    data = [s.to_dict() for s in scenarios]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(scenarios)} scenarios to {output_path}")


def load_scenarios(input_path: str) -> List[ScenarioConfig]:
    """Load scenario configurations from JSON."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    scenarios = []
    for item in data:
        config = ScenarioConfig(
            scenario_id=item['scenario_id'],
            scenario_type=ScenarioType(item['scenario_type']),
            seed=item['seed'],
            battery_capacity_mwh=item['battery_capacity_mwh'],
            battery_soc_init=item['battery_soc_init'],
            battery_discharge_rate_mw=item['battery_discharge_rate_mw'],
            battery_charge_rate_mw=item['battery_charge_rate_mw'],
            battery_efficiency=item['battery_efficiency'],
            battery_response_factor=item['battery_response_factor'],
            load_scaling=item['load_scaling'],
            solar_capacity_mw=item['solar_capacity_mw'],
            grid_import_limit_mw=item['grid_import_limit_mw'],
            load_window_start_hour=item['load_window_start_hour'],
            solar_window_start_hour=item['solar_window_start_hour'],
            load_noise_std=item['load_noise_std'],
            solar_intermittency_factor=item['solar_intermittency_factor'],
            load_spike_count=item['load_spike_count'],
            load_spike_magnitude=item['load_spike_magnitude'],
            solar_dip_count=item['solar_dip_count'],
            solar_dip_magnitude=item['solar_dip_magnitude'],
        )
        scenarios.append(config)

    print(f"Loaded {len(scenarios)} scenarios from {input_path}")
    return scenarios


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 80)
    print("Scenario Generator — Quick Test")
    print("=" * 80)

    # Generate 10 scenarios with balanced distribution
    scenarios = create_scenario_distribution(
        n_total=10,
        distribution={
            ScenarioType.NORMAL: 0.5,
            ScenarioType.STRESSED: 0.3,
            ScenarioType.EXTREME: 0.15,
            ScenarioType.SEASONAL: 0.05,
        },
        seed=42
    )

    # Print summary
    print(f"\nGenerated {len(scenarios)} scenarios:\n")
    for s in scenarios:
        print(f"  [{s.scenario_id:>2}] {s.scenario_type.value:<10} | "
              f"Battery: {s.battery_capacity_mwh:.2f} MWh | "
              f"Load scale: {s.load_scaling:.2f}x | "
              f"Solar cap: {s.solar_capacity_mw:.1f} MW | "
              f"Spikes: {s.load_spike_count}, Dips: {s.solar_dip_count}")

    print("\n" + "=" * 80)
