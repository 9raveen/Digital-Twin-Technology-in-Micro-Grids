"""
dataset_generator.py
====================
Main orchestrator for multi-scenario dataset generation.

Generates large-scale, diverse datasets by running N independent 720-hour
simulations with randomized parameters and injected stress events.

Pipeline:
  1. Load raw load and solar time series
  2. Generate N scenario configurations
  3. For each scenario:
     - Extract 720-hour windows
     - Apply stochastic perturbations
     - Run digital twin simulation
     - Record all features and labels
  4. Combine all scenarios into unified DataFrame
  5. Validate and export
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm

from scenario_generator import (
    ScenarioType,
    ScenarioConfig,
    create_scenario_distribution,
    save_scenarios,
)
from data_load import (
    prepare_load_series,
    extract_load_windows,
    add_stochastic_noise,
    scale_load_series,
    inject_load_spikes,
)
from data_solar import (
    prepare_solar_series,
    extract_solar_windows,
    add_cloud_intermittency,
    scale_solar_capacity,
    inject_solar_dips,
)
from battery_model import BatteryModel
from grid_simulator import create_network, run_load_flow
from label_generator import generate_label_with_risk, get_severity, get_blackout_margin
from config import (
    DAY_START_HOUR,
    DAY_END_HOUR,
    SIMULATION_HOURS,
    DISPLAY_PROGRESS_INTERVAL,
)


class DatasetGenerator:
    """Orchestrate N independent simulations with different scenarios."""

    def __init__(
        self,
        n_scenarios: int = 50,
        load_csv_path: str = "datasets/processed/load_scaled.csv",
        solar_csv_path: str = "datasets/processed/solar_scaled.csv",
        output_dir: str = "datasets/generated",
        seed: int = 42,
    ):
        """
        Initialize dataset generator.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to generate (default 50)
        load_csv_path : str
            Path to load CSV (optional; will load from raw data if not found)
        solar_csv_path : str
            Path to solar CSV (optional; will load from raw data if not found)
        output_dir : str
            Output directory for generated datasets
        seed : int
            Random seed for reproducibility
        """
        self.n_scenarios = n_scenarios
        self.load_csv_path = load_csv_path
        self.solar_csv_path = solar_csv_path
        self.output_dir = Path(output_dir)
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load raw data (full year if available)
        print("[1/4] Loading raw load and solar data...")
        try:
            self.raw_load = pd.read_csv(load_csv_path)["load_mw"].reset_index(drop=True)
        except Exception as e:
            print(f"  Warning: Could not load {load_csv_path}, will prepare from raw UCI data")
            self.raw_load = prepare_load_series("datasets/LD2011_2014.txt", n_days=365)

        try:
            self.raw_solar = pd.read_csv(solar_csv_path)["p_solar_mw"].reset_index(drop=True)
        except Exception as e:
            print(f"  Warning: Could not load {solar_csv_path}, will prepare from raw data")
            self.raw_solar = prepare_solar_series(
                "datasets/Solar Power Generation Data/Plant_1_Generation_Data.csv",
                n_days=34,
                pv_capacity_mw=3.0,
            )

        print(f"  Load data: {len(self.raw_load)} hours loaded")
        print(f"  Solar data: {len(self.raw_solar)} hours loaded")

    def generate_dataset(self) -> pd.DataFrame:
        """
        Main pipeline: generate N scenarios and combine into dataset.

        Returns
        -------
        pd.DataFrame
            Combined dataset with all scenarios
        """
        print("\n[2/4] Generating scenario configurations...")
        scenario_configs = create_scenario_distribution(
            n_total=self.n_scenarios,
            distribution={
                ScenarioType.NORMAL: 0.5,
                ScenarioType.STRESSED: 0.3,
                ScenarioType.EXTREME: 0.15,
                ScenarioType.SEASONAL: 0.05,
            },
            seed=self.seed,
        )

        # Save scenario configs
        save_scenarios(scenario_configs, str(self.output_dir / "scenario_configs.json"))

        print(f"\n[3/4] Running {len(scenario_configs)} scenarios...")
        all_samples = []

        for scenario in tqdm(scenario_configs, desc="Scenarios"):
            # Get time series for this scenario
            p_load = self._get_load_series(scenario)
            p_solar = self._get_solar_series(scenario)

            # Run simulation
            df_scenario = self._run_scenario_simulation(p_load, p_solar, scenario)

            # Add scenario metadata
            df_scenario["scenario_id"] = scenario.scenario_id
            df_scenario["scenario_type"] = scenario.scenario_type.value

            all_samples.append(df_scenario)

        # Combine all scenarios
        print("\nCombining scenarios...")
        df_combined = pd.concat(all_samples, ignore_index=True)

        return df_combined

    def _get_load_series(self, scenario: ScenarioConfig) -> pd.Series:
        """Prepare load series for scenario: window, scale, noise, spikes."""
        # Extract window
        start_idx = scenario.load_window_start_hour
        end_idx = start_idx + SIMULATION_HOURS

        if end_idx > len(self.raw_load):
            # Wrap around
            load_window = pd.concat(
                [
                    self.raw_load.iloc[start_idx:],
                    self.raw_load.iloc[: end_idx - len(self.raw_load)],
                ],
                ignore_index=True,
            )
        else:
            load_window = self.raw_load.iloc[start_idx:end_idx].reset_index(drop=True)

        # Scale by scenario factor
        load_scaled = scale_load_series(load_window, scenario.load_scaling)

        # Add noise
        load_noisy = add_stochastic_noise(load_scaled, scenario.load_noise_std, seed=scenario.seed)

        # Inject spikes
        load_perturbed = inject_load_spikes(
            load_noisy,
            spike_count=scenario.load_spike_count,
            spike_magnitude=scenario.load_spike_magnitude,
            seed=scenario.seed,
        )

        return load_perturbed

    def _get_solar_series(self, scenario: ScenarioConfig) -> pd.Series:
        """Prepare solar series for scenario: window, scale, clouds, dips."""
        # Extract window
        start_idx = scenario.solar_window_start_hour
        end_idx = start_idx + SIMULATION_HOURS

        if end_idx > len(self.raw_solar):
            # Wrap around
            solar_window = pd.concat(
                [
                    self.raw_solar.iloc[start_idx:],
                    self.raw_solar.iloc[: end_idx - len(self.raw_solar)],
                ],
                ignore_index=True,
            )
        else:
            solar_window = self.raw_solar.iloc[start_idx:end_idx].reset_index(drop=True)

        # Scale by scenario PV capacity
        solar_scaled = scale_solar_capacity(
            solar_window, scaling_factor=scenario.solar_capacity_mw / 3.0
        )  # Normalize to default 3.0 MW

        # Add cloud intermittency
        solar_cloudy = add_cloud_intermittency(
            solar_scaled, scenario.solar_intermittency_factor, seed=scenario.seed
        )

        # Inject dips
        solar_perturbed = inject_solar_dips(
            solar_cloudy,
            dip_count=scenario.solar_dip_count,
            dip_magnitude=scenario.solar_dip_magnitude,
            seed=scenario.seed,
        )

        return solar_perturbed

    def _run_scenario_simulation(
        self,
        p_load_series: pd.Series,
        p_solar_series: pd.Series,
        scenario: ScenarioConfig,
    ) -> pd.DataFrame:
        """Run one 720-hour scenario simulation."""
        T = len(p_load_series)

        # Initialize components
        battery = BatteryModel(
            capacity_mwh=scenario.battery_capacity_mwh,
            soc_init=scenario.battery_soc_init,
            soc_min=0.1,
            soc_max=0.9,
            charge_rate=scenario.battery_charge_rate_mw,
            discharge_rate=scenario.battery_discharge_rate_mw,
            efficiency=scenario.battery_efficiency,
            response_factor=scenario.battery_response_factor,
            use_crate_efficiency=True,
            use_ramp_limits=True,
            use_voltage_control=True,
            use_degradation=True,
            use_headroom_awareness=True,
            use_fullness_penalty=True,
            use_degradation_aware=True,
            use_foresight=True,
            use_grid_constraint=True,
            grid_import_limit=scenario.grid_import_limit_mw,
        )

        net = create_network(use_zip=True)

        # Pre-allocate records
        records = {
            "timestep": [],
            "day": [],
            "hour_of_day": [],
            "is_night": [],
            "p_load_mw": [],
            "p_solar_mw": [],
            "p_battery_mw": [],
            "p_grid_mw": [],
            "soc": [],
            "v_min": [],
            "v_min_measured": [],
            "v_max": [],
            "v_mean": [],
            "line_loading_max": [],
            "p_loss_mw": [],
            "converged": [],
            "v_margin": [],
            "severity": [],
            "blackout": [],
            "risk_score": [],
        }

        # Simulation loop
        for t in range(T):
            p_load = float(p_load_series.iloc[t])
            p_solar = float(p_solar_series.iloc[t])

            # Preliminary load flow
            lf_init = run_load_flow(net, p_load, p_solar, p_battery_mw=0.0)
            v_min_grid = lf_init["v_min"]

            # Foresight (6-hour lookahead)
            look_ahead = min(t + 6, T)
            future_load = float(np.mean(p_load_series.iloc[t:look_ahead]))
            future_solar = float(np.mean(p_solar_series.iloc[t:look_ahead]))

            battery.future_load_forecast = future_load
            battery.future_solar_forecast = future_solar

            # Dynamic SOC target
            future_deficit = future_load - future_solar
            if future_deficit > 0.5:
                battery.target_soc = 0.75
            elif future_deficit < -0.5:
                battery.target_soc = 0.45
            else:
                battery.target_soc = 0.6

            # Battery decision
            p_battery, soc = battery.step(p_load, p_solar, v_min=v_min_grid)

            # Final load flow
            lf = run_load_flow(net, p_load, p_solar, p_battery_mw=p_battery)

            # Grid residual
            p_grid = p_load - p_solar - p_battery

            # Measurement noise
            v_min_measured = lf["v_min"] + np.random.normal(0, 0.007)

            # Labels
            label, risk = generate_label_with_risk(v_min_measured, lf["converged"])
            severity = get_severity(lf["v_min"], lf["converged"])
            margin = get_blackout_margin(lf["v_min"])

            # Time features
            hour_of_day = t % 24
            day = t // 24
            is_night = int(hour_of_day < DAY_START_HOUR or hour_of_day >= DAY_END_HOUR)

            # Record
            records["timestep"].append(t)
            records["day"].append(day)
            records["hour_of_day"].append(hour_of_day)
            records["is_night"].append(is_night)
            records["p_load_mw"].append(p_load)
            records["p_solar_mw"].append(p_solar)
            records["p_battery_mw"].append(p_battery)
            records["p_grid_mw"].append(p_grid)
            records["soc"].append(soc)
            records["v_min"].append(lf["v_min"])
            records["v_min_measured"].append(v_min_measured)
            records["v_max"].append(lf["v_max"])
            records["v_mean"].append(lf["v_mean"])
            records["line_loading_max"].append(lf["line_loading_max"])
            records["p_loss_mw"].append(lf["p_loss_mw"])
            records["converged"].append(int(lf["converged"]))
            records["v_margin"].append(margin)
            records["severity"].append(severity)
            records["blackout"].append(label)
            records["risk_score"].append(risk)

        # Build DataFrame
        df = pd.DataFrame(records)

        # Add lag features
        for lag in range(1, 4):
            df[f"p_load_lag_{lag}"] = df["p_load_mw"].shift(lag)
            df[f"p_solar_lag_{lag}"] = df["p_solar_mw"].shift(lag)

        # Drop NaN from lags
        df = df.dropna()

        return df

    def generate_and_save(self) -> None:
        """Generate dataset and save to CSV."""
        print("\n" + "=" * 80)
        print("DATASET GENERATION PIPELINE")
        print("=" * 80)

        # Generate
        df = self.generate_dataset()

        # Save
        print("\n[4/4] Saving dataset...")
        output_csv = self.output_dir / "full_dataset.csv"
        df.to_csv(output_csv, index=False)

        print(f"✓ Saved to {output_csv}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Blackout rate: {df['blackout'].sum() / len(df) * 100:.1f}%")

        print("\n" + "=" * 80)

        return df


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generator = DatasetGenerator(
        n_scenarios=5,  # Small test run
        output_dir="datasets/generated_test",
        seed=42,
    )

    df = generator.generate_and_save()
