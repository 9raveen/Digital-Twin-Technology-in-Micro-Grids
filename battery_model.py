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
        """
        self.capacity       = capacity_mwh
        self.soc            = soc_init
        self.soc_min        = soc_min
        self.soc_max        = soc_max
        self.charge_rate    = charge_rate
        self.discharge_rate = discharge_rate

        # Split efficiency into charge and discharge components
        # Round-trip = eta_c × eta_d = eta_c² = eta → eta_c = sqrt(eta)
        self.eta_c          = np.sqrt(efficiency)  # Charge efficiency ~0.9747 for 0.95 round-trip
        self.eta_d          = np.sqrt(efficiency)  # Discharge efficiency ~0.9747
        self.eta            = efficiency           # Keep for backward compat (round-trip)

        self.response_factor = response_factor     # 0.5 = 50% of surplus/deficit
        self.soc_responsive = soc_responsive       # Enable SOC-dependent response modulation
        self.dt             = dt_hours             # 1.0 hour for standard simulations
        self.min_power_mw   = 0.05                 # Stability clamp: ignore <50 kW oscillations

    def reset(self, soc_init: float = 0.5) -> None:
        """
        Reset battery SOC to initial value.
        Call this before starting a new simulation run.

        Parameters
        ----------
        soc_init : float  SOC to reset to [0, 1]
        """
        self.soc = soc_init

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

    def step(self, p_load: float, p_solar: float) -> tuple:
        """
        Advance battery state by one hourly timestep.

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
          - Time scaling (dt) makes code independent of timestep

        Parameters
        ----------
        p_load  : float  Load demand at this timestep (MW)
        p_solar : float  Solar generation at this timestep (MW)

        Returns
        -------
        tuple (p_battery, soc)
          p_battery : float  Net battery power (MW)
                             > 0 → discharging
                             < 0 → charging
          soc       : float  Updated state of charge [soc_min, soc_max]
        """
        surplus = p_solar - p_load   # positive = excess solar, negative = deficit

        # ── STABILITY CLAMP: Ignore tiny oscillations (<min_power_mw) ─────────────
        if abs(surplus) < self.min_power_mw:
            p_battery = 0.0
            return p_battery, self.soc

        # Get effective response factor (directional, SOC-modulated)
        k = self._get_response_factor(surplus)

        if surplus > 0:
            # ── Charging ──────────────────────────────────────────
            # How much energy headroom remains in battery (MWh)
            headroom_mwh = (self.soc_max - self.soc) * self.capacity

            # Apply response factor: battery responds to k% of surplus
            # This makes battery more realistic (doesn't immediately absorb all excess)
            responsive_surplus = k * surplus

            # Charge power limited by: responsive surplus, max rate, available headroom
            p_charge = min(responsive_surplus, self.charge_rate, headroom_mwh / self.dt)
            p_charge = max(p_charge, 0.0)   # never negative

            # SOC increases — efficiency loss on the way in (charge efficiency)
            # Explicit time scaling: dt makes this work for any timestep
            # Using split efficiency model: eta_c ≈ sqrt(0.95) for each direction
            self.soc += (p_charge * self.eta_c * self.dt) / self.capacity
            p_battery = -p_charge           # negative = charging

        else:
            # ── Discharging ───────────────────────────────────────
            deficit = -surplus              # how much solar falls short

            # Apply response factor: battery responds to k% of deficit
            # Directional: more aggressive when SOC is high, conservative when low
            responsive_deficit = k * deficit

            # How much energy available in battery (MWh)
            available_mwh = (self.soc - self.soc_min) * self.capacity

            # Limit discharge to spread remaining energy over 3 hours minimum
            # This prevents battery from draining too quickly
            # Ensures sustainable discharge rate even under stress
            max_sustainable = available_mwh / (3.0 * self.dt)

            # Discharge power limited by:
            #   1. responsive deficit (what's needed × response factor)
            #   2. max discharge rate (hardware limit)
            #   3. max sustainable rate (energy sustainability over 3h)
            p_discharge = min(responsive_deficit, self.discharge_rate, max_sustainable)
            p_discharge = max(p_discharge, 0.0)   # never negative

            # SOC decreases — efficiency loss on the way out (discharge efficiency)
            # Explicit time scaling: dt makes this work for any timestep
            # Using split efficiency model: eta_d ≈ sqrt(0.95) for each direction
            self.soc -= (p_discharge / self.eta_d * self.dt) / self.capacity
            p_battery = p_discharge         # positive = discharging

        # Enforce SOC bounds (guard against floating point drift)
        self.soc = float(np.clip(self.soc, self.soc_min, self.soc_max))

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
            f"  Capacity      : {self.capacity} MWh\n"
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