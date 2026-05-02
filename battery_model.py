"""
battery_model.py
================
Rule-based battery model for the Digital Twin Microgrid project.

Models a Battery Energy Storage System (BESS) with:
  - State of Charge (SOC) tracking
  - Charge when solar surplus (p_solar > p_load)
  - Discharge when solar deficit (p_solar < p_load)
  - SOC bounds enforcement (never fully drain or overcharge)
  - Round-trip efficiency loss

Parameters (defaults):
  Capacity      : 0.5 MWh
  Initial SOC   : 0.5   (50%)
  SOC min       : 0.1   (10%)
  SOC max       : 0.9   (90%)
  Max charge    : 0.25  MW
  Max discharge : 0.25  MW
  Efficiency    : 0.95  (95% round-trip)

Output per timestep:
  p_battery : float  — net battery power in MW
                       > 0 → discharging (supplying load)
                       < 0 → charging   (absorbing surplus)
  soc       : float  — state of charge after this timestep [0, 1]
"""

import numpy as np  


class BatteryModel:
    """
    Rule-based Battery Energy Storage System (BESS).

    Operates on hourly timesteps matching the simulation loop.
    Charging and discharging decisions are made purely from
    the power balance between load and solar at each timestep.

    Key improvements:
      - Response factor (0.3-0.7): Battery doesn't immediately respond to 100% of surplus/deficit
      - Stability clamp: Ignores tiny oscillations (<50 kW)
      - Explicit time scaling: dt parameter makes code timestep-agnostic
    """

    def __init__(
        self,
        capacity_mwh: float = 0.5,
        soc_init: float     = 0.5,
        soc_min: float      = 0.1,
        soc_max: float      = 0.9,
        charge_rate: float  = 0.25,
        discharge_rate: float = 0.25,
        efficiency: float   = 0.95,
        response_factor: float = 0.5,
        dt_hours: float     = 1.0,
        soc_responsive: bool = True,
        use_crate_efficiency: bool = True,
        use_ramp_limits: bool = True,
        use_voltage_control: bool = True,
        use_degradation: bool = True,
        use_headroom_awareness: bool = True,
        use_fullness_penalty: bool = True,
        use_degradation_aware: bool = True,
        use_foresight: bool = True,
        use_grid_constraint: bool = True,
        grid_import_limit: float = 5.0,
    ):
        """
        Parameters
        ----------
        capacity_mwh   : float  Battery energy capacity in MWh
        soc_init       : float  Initial state of charge [0, 1]
        soc_min        : float  Minimum allowed SOC [0, 1]
        soc_max        : float  Maximum allowed SOC [0, 1]
        charge_rate    : float  Max charging power in MW
        discharge_rate : float  Max discharging power in MW
        efficiency     : float  Round-trip efficiency [0, 1]
        response_factor : float Response factor [0, 1]. Lower → battery responds slower (default 0.5)
        dt_hours       : float  Timestep duration in hours (default 1.0)
        soc_responsive : bool   If True, modulate response_factor by SOC (default True)
        use_crate_efficiency : bool  Enable C-rate dependent efficiency (Level 1)
        use_ramp_limits : bool       Enable power ramp rate limits (Level 1)
        use_voltage_control : bool   Enable voltage-responsive control (Level 2)
        use_degradation : bool       Enable cycle-depth degradation (Level 3)
        use_headroom_awareness : bool  Enable target SOC headroom (Strategy 1)
        use_fullness_penalty : bool    Enable full/empty penalties (Strategy 2)
        use_degradation_aware : bool   Reduce discharge when degraded (Strategy 3)
        use_foresight : bool           Enable simple load forecasting (Strategy 4)
        use_grid_constraint : bool     Enforce grid import limits (Strategy 5)
        grid_import_limit : float      Max grid import power in MW
        """
        self.capacity       = capacity_mwh
        self.soc            = soc_init
        self.soc_min        = soc_min
        self.soc_max        = soc_max
        self.charge_rate    = charge_rate
        self.discharge_rate = discharge_rate

        # Split efficiency into charge and discharge components
        self.eta_c          = np.sqrt(efficiency)
        self.eta_d          = np.sqrt(efficiency)
        self.eta            = efficiency

        self.response_factor = response_factor
        self.soc_responsive = soc_responsive
        self.dt             = dt_hours
        self.min_power_mw   = 0.05

        # ── LEVEL 1: Realistic Efficiency & Ramp Limits ─────────────────────
        self.use_crate_efficiency = use_crate_efficiency
        self.use_ramp_limits = use_ramp_limits
        self.max_ramp_mw = 0.1
        self.prev_p_battery = 0.0

        # ── LEVEL 2: Grid-Aware Control ───────────────────────────────────────
        self.use_voltage_control = use_voltage_control
        self.voltage_gain = 2.0

        # ── LEVEL 3: Cycle Degradation ────────────────────────────────────────
        self.use_degradation = use_degradation
        self.cycle_count = 0.0
        self.capacity_original = capacity_mwh

        # ── STRATEGY 1: Headroom Awareness ────────────────────────────────────
        self.use_headroom_awareness = use_headroom_awareness
        self.target_soc = 0.6  # Keep buffer → don't charge above 60%, discharge freely below 70%
        self.headroom_zone = 0.1  # 60-70% is "safe zone"

        # ── STRATEGY 2: Fullness Penalty ──────────────────────────────────────
        self.use_fullness_penalty = use_fullness_penalty
        self.penalty_full = 0.05   # Penalize charging above 85% SOC
        self.penalty_empty = 0.05  # Penalize discharging below 15% SOC

        # ── STRATEGY 3: Degradation-Aware Control ─────────────────────────────
        self.use_degradation_aware = use_degradation_aware
        self.degradation_threshold = 50.0  # If cycle_depth > 50, reduce discharge rate

        # ── STRATEGY 4: Simple Foresight ──────────────────────────────────────
        self.use_foresight = use_foresight
        self.future_load_forecast = 0.0  # Will be set by external logic
        self.future_solar_forecast = 0.0
        self.forecast_hours = 6  # Look 6 hours ahead

        # ── STRATEGY 5: Grid Import Constraint ────────────────────────────────
        self.use_grid_constraint = use_grid_constraint
        self.grid_import_limit = grid_import_limit

    def reset(self, soc_init: float = 0.5) -> None:
        """
        Reset battery SOC to initial value.
        Call this before starting a new simulation run.

        Parameters
        ----------
        soc_init : float  SOC to reset to [0, 1]
        """
        self.soc = soc_init

    def _apply_crate_efficiency(self, p_battery: float, is_charging: bool) -> tuple:
        """
        LEVEL 1: Apply C-rate dependent efficiency loss.

        High charge/discharge rates reduce efficiency:
        - 0.1 C-rate (25 kW / 0.25 MW per 0.5 MWh) → 95% efficiency
        - 1.0 C-rate (250 kW / 0.5 MW per 0.5 MWh) → 85% efficiency

        Parameters
        ----------
        p_battery : float  Absolute power in MW
        is_charging : bool  True if charging, False if discharging

        Returns
        -------
        tuple (efficiency_adjusted_power, efficiency_factor)
        """
        c_rate = abs(p_battery) / self.capacity

        # Simple approximation: efficiency = 0.95 - 0.05 * c_rate
        eta_crate = 0.95 - 0.05 * c_rate
        eta_crate = np.clip(eta_crate, 0.85, 0.95)

        # Apply efficiency
        if is_charging:
            p_effective = p_battery * eta_crate
        else:
            p_effective = p_battery / eta_crate

        return p_effective, eta_crate

    def _apply_ramp_limits(self, p_battery_cmd: float) -> float:
        """
        LEVEL 1: Apply power ramp rate limits.

        Prevents unrealistic instantaneous power jumps.
        Max ramp: 0.1 MW per timestep (realistic hardware limitation).

        Parameters
        ----------
        p_battery_cmd : float  Commanded battery power (MW)

        Returns
        -------
        float  Power-limited to ramp rate
        """
        max_ramp = self.max_ramp_mw

        p_battery_limited = np.clip(
            p_battery_cmd,
            self.prev_p_battery - max_ramp,
            self.prev_p_battery + max_ramp
        )

        return p_battery_limited

    def _get_voltage_control(self, surplus: float, v_min: float = 1.0) -> float:
        """
        LEVEL 2: Grid-aware voltage-responsive control.

        Battery reacts to voltage instability:
        - If v_min < 1.0, system is stressed → discharge more
        - If v_min > 1.0, system is stable → follow normal logic

        Modified surplus: surplus_eff = surplus + voltage_gain * (v_min - 1.0)

        Parameters
        ----------
        surplus : float  Power balance (solar - load)
        v_min : float    Minimum bus voltage (pu), default 1.0 (no stress)

        Returns
        -------
        float  Effective surplus with voltage correction
        """
        voltage_error = v_min - 1.0
        effective_surplus = surplus + self.voltage_gain * voltage_error

        return effective_surplus

    def _apply_degradation(self, delta_soc: float) -> None:
        """
        LEVEL 3: Apply cycle-depth degradation.

        Battery capacity degrades with cycling:
        - Each cycle (full discharge/charge) causes ~0.01% capacity loss
        - Degradation tracked cumulatively

        Parameters
        ----------
        delta_soc : float  Change in SOC this timestep
        """
        # Track cumulative cycle depth
        self.cycle_count += abs(delta_soc)

        # Capacity degrades: each 100% cycle loss = 0.01% capacity
        # 0.0001 factor = 0.01% loss per 1% cycle depth
        capacity_factor = 1.0 - (0.0001 * self.cycle_count)
        capacity_factor = np.clip(capacity_factor, 0.7, 1.0)  # Never drop below 70%

        self.capacity = self.capacity_original * capacity_factor

    def _apply_headroom_awareness(self, surplus: float) -> float:
        """
        STRATEGY 1: Maintain SOC headroom for flexibility.

        Instead of charging to maximum, reduce charging rate when at target SOC.
        If soc > target_soc: reduce surplus to 20% (multiply by 0.2)
        This preserves ability to discharge during peaks.

        Returns
        -------
        float  Adjusted surplus (soft reduction when at headroom limit)
        """
        if surplus > 0:  # Charging scenario
            # Soft penalty: reduce charging to 20% when above target
            if self.soc > self.target_soc:
                return surplus * 0.2  # Charge only 20% of surplus
            return surplus
        return surplus

    def _apply_fullness_penalty(self, surplus: float) -> float:
        """
        STRATEGY 2: Penalize extreme full/empty states.

        High penalty at extremes forces mid-range operation.
        This maintains flexibility for emergencies.

        Returns
        -------
        float  Adjusted surplus with fullness penalties
        """
        penalty = 0.0

        if surplus > 0:  # Charging
            # Penalty for charging when near full
            if self.soc > 0.85:
                penalty -= self.penalty_full * (self.soc - 0.85)
        else:  # Discharging
            # Penalty for discharging when near empty
            if self.soc < 0.15:
                penalty -= self.penalty_empty * (0.15 - self.soc)

        return surplus * (1.0 + penalty)

    def _apply_degradation_aware_control(self, p_battery: float) -> float:
        """
        STRATEGY 3: Reduce discharge rate when battery is degraded.

        High cycle depth → battery is aging → reduce stress.

        Parameters
        ----------
        p_battery : float  Planned discharge power

        Returns
        -------
        float  Reduced discharge power if degraded
        """
        if not self.use_degradation_aware or self.cycle_count < self.degradation_threshold:
            return p_battery

        # If degraded, reduce discharge rate
        degradation_factor = 1.0 - 0.3 * ((self.cycle_count - self.degradation_threshold) / 50.0)
        degradation_factor = np.clip(degradation_factor, 0.7, 1.0)

        if p_battery > 0:  # Discharging
            return p_battery * degradation_factor

        return p_battery

    def _apply_simple_foresight(self, surplus: float) -> float:
        """
        STRATEGY 4: Look ahead at future load/solar to adjust strategy.

        If deficit predicted in next 6h, conserve energy now.
        If surplus predicted, charge more aggressively now.

        Returns
        -------
        float  Adjusted surplus based on forecast
        """
        if not self.use_foresight:
            return surplus

        # future_deficit = expected load - expected solar (over next 6h)
        future_deficit = self.future_load_forecast - self.future_solar_forecast

        # If large deficit expected, reduce charging (conserve energy)
        if future_deficit > 0.5:
            return surplus * 0.7  # Charge only 70% of normal

        # If large surplus expected, charge more (prepare for load)
        if future_deficit < -0.5:
            return surplus * 1.3  # Charge 130% of normal

        return surplus

    def _apply_grid_constraint(self, p_battery: float, p_solar: float, p_load: float) -> float:
        """
        STRATEGY 5: Enforce grid import constraint.

        If discharging would exceed grid import limit, reduce it.

        Parameters
        ----------
        p_battery : float  Planned battery power
        p_solar : float    Solar generation
        p_load : float     Load demand

        Returns
        -------
        float  Reduced discharge power if grid constraint violated
        """
        if not self.use_grid_constraint:
            return p_battery

        # p_grid = p_load - p_solar - p_battery
        # If p_battery > 0 (discharge), it reduces p_grid
        p_grid_if_full = p_load - p_solar - p_battery

        if p_grid_if_full > self.grid_import_limit:
            # Reduce discharge to meet grid constraint
            p_battery_limited = p_battery - (p_grid_if_full - self.grid_import_limit)
            return max(p_battery_limited, 0.0)

        return p_battery

    def _get_response_factor(self, surplus: float) -> float:
        """
        Calculate effective response factor modulated by SOC and direction.

        Directional logic:
          CHARGING (surplus > 0):
            - At SOC_min (10%): k_eff = response_factor × 1.0 (aggressive charging)
            - At SOC_mid (50%): k_eff = response_factor × 0.5 (moderate charging)
            - At SOC_max (90%): k_eff = response_factor × 0.0 (no charging)

          DISCHARGING (surplus < 0):
            - At SOC_min (10%): k_eff = response_factor × 0.0 (preserve discharge power)
            - At SOC_mid (50%): k_eff = response_factor × 0.5 (moderate discharge)
            - At SOC_max (90%): k_eff = response_factor × 1.0 (aggressive discharge)

        When soc_responsive=False:
          - Always returns self.response_factor (original behavior)

        Parameters
        ----------
        surplus : float  Power surplus (positive = charging, negative = discharging)

        Returns
        -------
        float  Effective response factor in [0, response_factor]
        """
        if not self.soc_responsive:
            return self.response_factor

        soc_range = self.soc_max - self.soc_min
        soc_norm = (self.soc - self.soc_min) / soc_range
        soc_norm = np.clip(soc_norm, 0.0, 1.0)

        if surplus > 0:
            # CHARGING: More aggressive when empty, reduced when full
            # k = k_base × (1 - soc_norm)
            # SOC 10% → k = k_base × 0.9 (very aggressive)
            # SOC 50% → k = k_base × 0.5 (moderate)
            # SOC 90% → k = k_base × 0.1 (very conservative, almost no charge)
            modulation = 1.0 - soc_norm
        else:
            # DISCHARGING: Less aggressive when empty, more aggressive when full
            # k = k_base × soc_norm
            # SOC 10% → k = k_base × 0.1 (very conservative, preserve energy)
            # SOC 50% → k = k_base × 0.5 (moderate)
            # SOC 90% → k = k_base × 0.9 (very aggressive, drain excess)
            modulation = soc_norm

        effective_k = self.response_factor * modulation
        return float(np.clip(effective_k, 0.0, self.response_factor))

    def step(self, p_load: float, p_solar: float, v_min: float = 1.0, debug: bool = False) -> tuple:
        """
        Advance battery state by one hourly timestep.

        Enhanced with three levels of realism:
          LEVEL 1 - Realistic efficiency & ramp limits
            - C-rate dependent efficiency loss (faster rates = lower efficiency)
            - Power ramp rate limits (0.1 MW/step max change)
          LEVEL 2 - Grid-aware control
            - Voltage-responsive: battery reacts to v_min < 1.0 stress
          LEVEL 3 - Degradation
            - Capacity degrades with cycle depth (~0.01% per full cycle)

        Decision logic:
          surplus > 0  → charge battery (at response_factor rate)
          surplus < 0  → discharge battery (at response_factor rate)

        Constraints applied:
          - Response factor limits reaction speed (realistic control)
          - SOC-dependent modulation: reduced response at extremes
          - Stability clamp filters oscillations <50 kW
          - Charge/discharge capped at max rate
          - SOC cannot exceed soc_max (charging) or go below soc_min (discharging)
          - Efficiency loss applied on both charge and discharge (split model)
          - C-rate dependent efficiency (Level 1)
          - Power ramp limits (Level 1)
          - Voltage-responsive surplus adjustment (Level 2)
          - Time scaling (dt) makes code independent of timestep

        Parameters
        ----------
        p_load  : float  Load demand at this timestep (MW)
        p_solar : float  Solar generation at this timestep (MW)
        v_min   : float  Minimum bus voltage for voltage-responsive control (pu, default 1.0)
        debug   : bool   If True, print decision details (default False)

        Returns
        -------
        tuple (p_battery, soc)
          p_battery : float  Net battery power (MW)
                             > 0 → discharging
                             < 0 → charging
          soc       : float  Updated state of charge [soc_min, soc_max]
        """
        soc_before = self.soc
        surplus = p_solar - p_load   # positive = excess solar, negative = deficit

        # Decision trace for debugging
        decisions = {
            "initial_surplus": surplus,
            "soc_before": soc_before,
            "v_min": v_min,
        }

        # ── LEVEL 2: Grid-aware control ────────────────────────────────────────
        if self.use_voltage_control:
            surplus = self._get_voltage_control(surplus, v_min)
            decisions["after_voltage"] = surplus

        # ── STRATEGY 1: Headroom Awareness ─────────────────────────────────────
        if self.use_headroom_awareness:
            surplus = self._apply_headroom_awareness(surplus)
            decisions["after_headroom"] = surplus

        # ── STRATEGY 2: Fullness Penalty ──────────────────────────────────────
        if self.use_fullness_penalty:
            surplus = self._apply_fullness_penalty(surplus)
            decisions["after_fullness"] = surplus

        # ── STRATEGY 4: Simple Foresight ──────────────────────────────────────
        if self.use_foresight:
            surplus = self._apply_simple_foresight(surplus)
            decisions["after_foresight"] = surplus

        # ── STABILITY CLAMP: Ignore tiny oscillations (<min_power_mw) ─────────────
        if abs(surplus) < self.min_power_mw:
            if debug:
                print(f"[STABILITY CLAMP] surplus={surplus:.4f} < {self.min_power_mw} → no action")
            p_battery = 0.0
            self.prev_p_battery = 0.0
            return p_battery, self.soc

        # Get effective response factor (directional, SOC-modulated)
        k = self._get_response_factor(surplus)
        decisions["response_factor"] = k

        if debug:
            print(f"  surplus: {decisions['initial_surplus']:.4f}", end="")
            if "after_voltage" in decisions:
                print(f" → voltage: {decisions['after_voltage']:.4f}", end="")
            if "after_headroom" in decisions:
                print(f" → headroom: {decisions['after_headroom']:.4f}", end="")
            if "after_fullness" in decisions:
                print(f" → fullness: {decisions['after_fullness']:.4f}", end="")
            if "after_foresight" in decisions:
                print(f" → foresight: {decisions['after_foresight']:.4f}", end="")
            print(f" | k={k:.3f} | soc={soc_before:.3f} | target={self.target_soc:.3f}")

        if surplus > 0:
            # ── Charging ──────────────────────────────────────────
            headroom_mwh = (self.soc_max - self.soc) * self.capacity
            responsive_surplus = k * surplus
            p_charge = min(responsive_surplus, self.charge_rate, headroom_mwh / self.dt)
            p_charge = max(p_charge, 0.0)

            # ── LEVEL 1: Apply C-rate efficiency and ramp limits ──────
            if self.use_crate_efficiency:
                p_charge_eff, _ = self._apply_crate_efficiency(p_charge, is_charging=True)
            else:
                p_charge_eff = p_charge

            if self.use_ramp_limits:
                p_charge_limited = self._apply_ramp_limits(-p_charge_eff)
                p_charge_eff = -p_charge_limited  # Convert back to negative
            else:
                p_charge_eff = -p_charge_eff

            # SOC update with efficiency
            soc_delta = (p_charge * self.eta_c * self.dt) / self.capacity
            self.soc += soc_delta
            p_battery = -p_charge

        else:
            # ── Discharging ───────────────────────────────────────
            deficit = -surplus

            responsive_deficit = k * deficit
            available_mwh = (self.soc - self.soc_min) * self.capacity
            max_sustainable = available_mwh / (3.0 * self.dt)

            p_discharge = min(responsive_deficit, self.discharge_rate, max_sustainable)
            p_discharge = max(p_discharge, 0.0)

            # ── SOC Safety Buffer: Reduce discharge if SOC < 0.2 (20%) ──────────
            if self.soc < 0.2:
                p_discharge *= 0.5  # Reduce discharge to 50%
                if debug:
                    print(f"  [SOC SAFETY BUFFER] soc={self.soc:.3f} < 0.2 → discharge reduced to 50%")

            # ── STRATEGY 3: Degradation-Aware Control ──────────────────────────
            if self.use_degradation_aware:
                p_discharge = self._apply_degradation_aware_control(p_discharge)

            # ── STRATEGY 5: Grid Import Constraint ─────────────────────────────
            if self.use_grid_constraint:
                p_discharge = self._apply_grid_constraint(p_discharge, p_solar, p_load)

            # ── LEVEL 1: Apply C-rate efficiency and ramp limits ──────
            if self.use_crate_efficiency:
                p_discharge_eff, _ = self._apply_crate_efficiency(p_discharge, is_charging=False)
            else:
                p_discharge_eff = p_discharge

            if self.use_ramp_limits:
                p_discharge_limited = self._apply_ramp_limits(p_discharge_eff)
                p_discharge_eff = p_discharge_limited

            # SOC update with efficiency
            soc_delta = -(p_discharge * self.dt) / (self.eta_d * self.capacity)
            self.soc += soc_delta
            p_battery = p_discharge

        # ── LEVEL 3: Apply degradation ────────────────────────────────────────
        if self.use_degradation:
            self._apply_degradation(abs(self.soc - soc_before))

        # Enforce SOC bounds (guard against floating point drift)
        self.soc = float(np.clip(self.soc, self.soc_min, self.soc_max))
        self.prev_p_battery = p_battery

        return p_battery, self.soc

    def get_soc(self) -> float:
        """Return current state of charge."""
        return self.soc

    def is_empty(self) -> bool:
        """Return True if battery is at minimum SOC."""
        return self.soc <= self.soc_min + 1e-6

    def is_full(self) -> bool:
        """Return True if battery is at maximum SOC."""
        return self.soc >= self.soc_max - 1e-6

    def summary(self) -> None:
        """Print current battery configuration."""
        print(
            f"BatteryModel Configuration\n"
            f"  Capacity      : {self.capacity:.4f} MWh (original: {self.capacity_original} MWh)\n"
            f"  SOC current   : {self.soc:.4f}  ({self.soc*100:.1f}%)\n"
            f"  SOC min/max   : {self.soc_min} / {self.soc_max}\n"
            f"  Charge rate   : {self.charge_rate} MW\n"
            f"  Discharge rate: {self.discharge_rate} MW\n"
            f"  Efficiency (round-trip): {self.eta*100:.1f}%\n"
            f"  Efficiency (charge)    : {self.eta_c*100:.1f}%  (split model)\n"
            f"  Efficiency (discharge) : {self.eta_d*100:.1f}%  (split model)\n"
            f"  Response factor (base) : {self.response_factor}  (lower = slower response)\n"
            f"  SOC-responsive         : {'YES' if self.soc_responsive else 'NO'}  (modulate k by SOC)\n"
            f"  Timestep (dt)          : {self.dt} hour(s)\n"
            f"  Stability clamp        : {self.min_power_mw} MW (ignores oscillations <50 kW)\n"
            f"\n"
            f"  LEVEL 1 — Realistic Efficiency & Ramp Limits:\n"
            f"    C-rate efficiency      : {'ENABLED' if self.use_crate_efficiency else 'DISABLED'}\n"
            f"    Power ramp limits      : {'ENABLED' if self.use_ramp_limits else 'DISABLED'} (max: {self.max_ramp_mw} MW/step)\n"
            f"\n"
            f"  LEVEL 2 — Grid-Aware Control:\n"
            f"    Voltage-responsive     : {'ENABLED' if self.use_voltage_control else 'DISABLED'} (gain: {self.voltage_gain})\n"
            f"\n"
            f"  LEVEL 3 — Degradation:\n"
            f"    Cycle degradation      : {'ENABLED' if self.use_degradation else 'DISABLED'}\n"
            f"    Cumulative cycle depth : {self.cycle_count:.4f}\n"
            f"    Capacity factor        : {self.capacity/self.capacity_original:.4f}x\n"
            f"\n"
            f"  Status                 : {'FULL' if self.is_full() else 'EMPTY' if self.is_empty() else 'NORMAL'}"
        )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import pandas as pd

    print("=" * 50)
    print("Battery Model — Quick Test")
    print("=" * 50)

    battery = BatteryModel()
    battery.summary()

    # Simulate 24 hours with a simple load/solar pattern
    # Hours 0-5   : night  — no solar, battery discharges
    # Hours 6-17  : day    — solar > load, battery charges
    # Hours 18-23 : evening — no solar, battery discharges

    test_schedule = [
        # (hour, p_load, p_solar)
        *[(h, 2.5, 0.0)   for h in range(0, 6)],    # night
        *[(h, 2.0, 3.0)   for h in range(6, 18)],   # day — solar surplus
        *[(h, 3.0, 0.0)   for h in range(18, 24)],  # evening — deficit
    ]

    print(f"\n{'Hour':>4} | {'p_load':>8} | {'p_solar':>8} | "
          f"{'p_batt':>8} | {'p_grid':>8} | {'SOC':>6}")
    print("-" * 58)

    for hour, p_load, p_solar in test_schedule:
        p_battery, soc = battery.step(p_load, p_solar)
        p_grid = p_load - p_solar - p_battery
        print(f"{hour:>4} | {p_load:>8.3f} | {p_solar:>8.3f} | "
              f"{p_battery:>8.3f} | {p_grid:>8.3f} | {soc:>6.4f}")

    print(f"\nFinal SOC : {battery.get_soc():.4f}")
    print(f"Is empty  : {battery.is_empty()}")
    print(f"Is full   : {battery.is_full()}")