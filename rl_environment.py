"""
rl_environment.py
=================
Gymnasium environment for microgrid RL battery dispatch.

Uses a pre-computed load flow lookup table (lf_lookup.csv) instead of
calling pandapower at every training step. This makes training ~1000x
faster while preserving physical accuracy.

Dependencies
------------
    Run rl_precompute.py FIRST to generate lf_lookup.csv.

Observation space (7 features)
-------------------------------
    0  soc              — battery state of charge [0, 1]
    1  p_load_norm      — normalised load demand
    2  p_solar_norm     — normalised solar generation
    3  hour_norm        — hour of day / 23
    4  is_night         — 1 if hour < 6 or hour >= 20
    5  load_avg6h_norm  — 6-hour rolling mean of load
    6  solar_avg6h_norm — 6-hour rolling mean of solar

Action space (continuous)
--------------------------
    Float in [-1.0, +1.0]
    Negative = charge,  Positive = discharge,  Zero = idle

Reward
------
    -50   per blackout   (dominant signal)
    +1.0  if v_min >= 0.93 else -5.0   (voltage quality)
    +0.5 * soc if soc > 0.2 else -2.0  (SOC health)
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from battery_model   import BatteryModel
from label_generator import generate_label
from data_load       import prepare_load_series
from data_solar      import prepare_solar_series


# ── Constants ─────────────────────────────────────────────────────────────────

LOOKUP_PATH = 'datasets/processed/lf_lookup.csv'
V_THRESHOLD = 0.93     # must match label_generator.py


# ── Environment ───────────────────────────────────────────────────────────────

class MicrogridEnv(gym.Env):
    """
    Fast Gymnasium environment using pre-computed load flow lookup table.
    No pandapower calls during training — lookup is ~0.001ms vs ~30ms.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        load_path:        str,
        solar_path:       str,
        n_days:           int   = 7,
        battery_capacity: float = 2.0,
    ):
        super().__init__()

        # ── Load time-series data ──────────────────────────────────────────
        self.p_load_series  = prepare_load_series(load_path, n_days)
        self.p_solar_series = prepare_solar_series(solar_path, n_days)

        T = min(len(self.p_load_series), len(self.p_solar_series))
        self.p_load_series  = self.p_load_series.iloc[:T].values.astype(np.float32)
        self.p_solar_series = self.p_solar_series.iloc[:T].values.astype(np.float32)
        self.T = T

        # ── Load lookup table ──────────────────────────────────────────────
        if not os.path.exists(LOOKUP_PATH):
            raise FileNotFoundError(
                f"Lookup table not found: {LOOKUP_PATH}\n"
                f"Run  python rl_precompute.py  first."
            )
        self._lf_table = pd.read_csv(LOOKUP_PATH)

        # Sorted unique bins for fast nearest-neighbour lookup
        self._load_bins  = np.sort(self._lf_table['net_load_mw'].unique())
        self._solar_bins = np.sort(self._lf_table['p_solar_mw'].unique())

        # ── Battery ────────────────────────────────────────────────────────
        self.battery_capacity = battery_capacity

        # ── Normalisation constants ────────────────────────────────────────
        self.max_load  = float(self.p_load_series.max())  or 1.0
        self.max_solar = float(self.p_solar_series.max()) or 1.0

        # ── Gymnasium spaces ───────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=np.zeros(7,  dtype=np.float32),
            high=np.full(7, 2.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode state
        self.battery:             BatteryModel = None
        self.t:                   int          = 0
        self._episode_blackouts:  int          = 0
        self._episode_reward:     float        = 0.0
        self._load_history:       list         = []
        self._solar_history:      list         = []


    # ── Lookup ────────────────────────────────────────────────────────────────

    def _lookup_lf(self, net_load: float, p_solar: float) -> dict:
        """
        Find nearest pre-computed load flow result for (net_load, p_solar).
        Clamps to table bounds — no extrapolation.
        """
        net_load = float(np.clip(net_load, self._load_bins[0],  self._load_bins[-1]))
        p_solar  = float(np.clip(p_solar,  self._solar_bins[0], self._solar_bins[-1]))

        load_bin  = self._load_bins[np.argmin(np.abs(self._load_bins  - net_load))]
        solar_bin = self._solar_bins[np.argmin(np.abs(self._solar_bins - p_solar))]

        row = self._lf_table[
            (self._lf_table['net_load_mw'] == load_bin) &
            (self._lf_table['p_solar_mw']  == solar_bin)
        ].iloc[0]

        return {
            'v_min':            float(row['v_min']),
            'v_max':            float(row['v_max']),
            'v_mean':           float(row['v_mean']),
            'line_loading_max': float(row['line_loading_max']),
            'p_loss_mw':        float(row['p_loss_mw']),
            'converged':        bool(row['converged']),
        }


    # ── Gymnasium Interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.battery = BatteryModel(
            capacity_mwh = self.battery_capacity,
            soc_init     = 0.5,
        )
        self.t                  = 0
        self._episode_blackouts = 0
        self._episode_reward    = 0.0
        self._load_history      = []
        self._solar_history     = []
        return self._get_obs(), {}


    def step(self, action):
        assert self.battery is not None, "Call reset() before step()."

        p_load  = float(self.p_load_series[self.t])
        p_solar = float(self.p_solar_series[self.t])

        # Battery dispatch
        p_cmd          = float(action[0]) * self.battery.charge_rate
        p_battery, soc = self._apply_action(p_cmd)

        # Lookup load flow — battery reduces effective load on network
        net_load = max(p_load - p_battery, 0.1)
        lf       = self._lookup_lf(net_load, p_solar)

        # Label and reward
        blackout = generate_label(lf['v_min'], lf['converged'])
        reward   = self._compute_reward(lf['v_min'], blackout, soc)

        # History
        self._load_history.append(p_load)
        self._solar_history.append(p_solar)

        # Advance
        self.t                  += 1
        self._episode_blackouts += int(blackout)
        self._episode_reward    += reward

        terminated = self.t >= self.T
        truncated  = False
        obs        = np.zeros(7, dtype=np.float32) if terminated else self._get_obs()

        info = {
            'blackout':          blackout,
            'soc':               soc,
            'v_min':             lf['v_min'],
            'p_battery_mw':      p_battery,
            'p_load_mw':         p_load,
            'p_solar_mw':        p_solar,
            'net_load_mw':       net_load,
            'converged':         lf['converged'],
            'episode_blackouts': self._episode_blackouts,
        }
        return obs, reward, terminated, truncated, info


    def render(self, mode='human'):
        if self.battery is None:
            print("[MicrogridEnv] Not initialised — call reset() first.")
            return
        t      = self.t
        p_load = float(self.p_load_series[min(t, self.T - 1)])
        p_solar= float(self.p_solar_series[min(t, self.T - 1)])
        print(
            f"t={t:4d}/{self.T} | "
            f"Load={p_load:.3f} MW | Solar={p_solar:.3f} MW | "
            f"SOC={self.battery.soc:.3f} | "
            f"Blackouts={self._episode_blackouts}"
        )

    def close(self):
        pass


    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _apply_action(self, p_cmd: float) -> tuple:
        if p_cmd < 0:
            headroom  = (self.battery.soc_max - self.battery.soc) * self.battery.capacity
            p_charge  = min(abs(p_cmd), self.battery.charge_rate, headroom)
            p_charge  = max(p_charge, 0.0)
            self.battery.soc += (p_charge * self.battery.eta) / self.battery.capacity
            p_battery = -p_charge
        else:
            available   = (self.battery.soc - self.battery.soc_min) * self.battery.capacity
            p_discharge = min(p_cmd, self.battery.discharge_rate, available)
            p_discharge = max(p_discharge, 0.0)
            self.battery.soc -= p_discharge / (self.battery.capacity * self.battery.eta)
            p_battery = p_discharge

        self.battery.soc = float(
            np.clip(self.battery.soc, self.battery.soc_min, self.battery.soc_max)
        )
        return p_battery, self.battery.soc


    def _compute_reward(self, v_min: float, blackout: int, soc: float) -> float:
        reward = 0.0
        if blackout:
            reward -= 50.0
        if v_min >= V_THRESHOLD:
            reward += 1.0
        else:
            reward -= 5.0
        if soc > 0.2:
            reward += 0.5 * soc
        else:
            reward -= 2.0
        return float(reward)


    def _get_obs(self) -> np.ndarray:
        t       = min(self.t, self.T - 1)
        p_load  = float(self.p_load_series[t])
        p_solar = float(self.p_solar_series[t])
        soc     = self.battery.soc
        hour    = t % 24

        h_load  = self._load_history[-6:]  if self._load_history  else [p_load]
        h_solar = self._solar_history[-6:] if self._solar_history else [p_solar]

        return np.array([
            soc,
            p_load  / self.max_load,
            p_solar / self.max_solar,
            hour    / 23.0,
            float(hour < 6 or hour >= 20),
            float(np.mean(h_load))  / self.max_load,
            float(np.mean(h_solar)) / self.max_solar,
        ], dtype=np.float32)


# ── Quick validation ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    from gymnasium.utils.env_checker import check_env

    LOAD_PATH  = 'datasets/electricityloaddiagrams20112014/LD2011_2014.txt'
    SOLAR_PATH = 'datasets/Solar Power Generation Data/Plant_1_Generation_Data.csv'

    print("Creating MicrogridEnv ...")
    env = MicrogridEnv(LOAD_PATH, SOLAR_PATH, n_days=7)
    print(f"  Timesteps    : {env.T}")
    print(f"  Lookup rows  : {len(env._lf_table)}")
    print(f"  Obs space    : {env.observation_space}")
    print(f"  Action space : {env.action_space}")

    print("\nRunning Gymnasium env checker ...")
    check_env(env, warn=True)
    print("[OK] Environment passed gym checks.\n")

    print("Running 1 random episode ...")
    obs, _ = env.reset()
    total_reward = 0.0
    blackouts    = 0
    done         = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        blackouts    += info['blackout']

    print(f"Episode done.")
    print(f"  Total reward : {total_reward:.2f}")
    print(f"  Blackouts    : {blackouts} / {env.T}")
    print(f"  LOLP         : {blackouts / env.T * 100:.2f}%")
    print("[OK] rl_environment.py validated successfully.")