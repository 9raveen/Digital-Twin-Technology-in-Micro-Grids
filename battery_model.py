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
        """
        self.capacity      = capacity_mwh
        self.soc           = soc_init
        self.soc_min       = soc_min
        self.soc_max       = soc_max
        self.charge_rate   = charge_rate
        self.discharge_rate = discharge_rate
        self.eta           = efficiency

    def reset(self, soc_init: float = 0.5) -> None:
        """
        Reset battery SOC to initial value.
        Call this before starting a new simulation run.

        Parameters
        ----------
        soc_init : float  SOC to reset to [0, 1]
        """
        self.soc = soc_init

    def step(self, p_load: float, p_solar: float) -> tuple:
        """
        Advance battery state by one hourly timestep.

        Decision logic:
          surplus > 0  → charge battery
          surplus < 0  → discharge battery to cover deficit

        Constraints applied:
          - Charge/discharge capped at max rate
          - SOC cannot exceed soc_max (charging) or go below soc_min (discharging)
          - Efficiency loss applied on both charge and discharge

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

        if surplus > 0:
            # ── Charging ──────────────────────────────────────────
            # How much energy headroom remains in battery (MWh)
            headroom_mwh = (self.soc_max - self.soc) * self.capacity

            # Charge power limited by: surplus, max rate, available headroom
            p_charge = min(surplus, self.charge_rate, headroom_mwh)
            p_charge = max(p_charge, 0.0)   # never negative

            # SOC increases — efficiency loss on the way in
            self.soc += (p_charge * self.eta) / self.capacity
            p_battery = -p_charge           # negative = charging

        else:
            # ── Discharging ───────────────────────────────────────
            deficit = -surplus              # how much solar falls short

            # How much energy available in battery (MWh)
            available_mwh = (self.soc - self.soc_min) * self.capacity

            # Discharge power limited by: deficit, max rate, available energy
            p_discharge = min(deficit, self.discharge_rate, available_mwh)
            p_discharge = max(p_discharge, 0.0)   # never negative

            # SOC decreases — efficiency loss on the way out
            self.soc -= p_discharge / (self.capacity * self.eta)
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
            f"  Efficiency    : {self.eta*100:.0f}%\n"
            f"  Status        : {'FULL' if self.is_full() else 'EMPTY' if self.is_empty() else 'NORMAL'}"
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