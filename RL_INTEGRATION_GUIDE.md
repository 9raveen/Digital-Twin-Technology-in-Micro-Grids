# Reinforcement Learning Integration Guide
## Microgrid Digital Twin — Battery Dispatch Optimisation

> **Document purpose:** Explain *why* RL fits this project, *what* has to change, and *how* to implement it step-by-step,  
> using the existing codebase (`battery_model.py`, `digital_twin.py`, `grid_simulator.py`, `ml_model.py`) as the foundation.

---

## Table of Contents
1. [Why RL? — The Problem with Rule-Based Dispatch](#1-why-rl)
2. [RL Concepts Applied to the Microgrid](#2-rl-concepts)
3. [Architecture Overview](#3-architecture-overview)
4. [The RL Environment — MicrogridEnv](#4-the-rl-environment)
5. [State Space](#5-state-space)
6. [Action Space](#6-action-space)
7. [Reward Function Design](#7-reward-function-design)
8. [Algorithm Choice](#8-algorithm-choice)
9. [Step-by-Step Integration Plan](#9-step-by-step-integration-plan)
10. [Code Skeleton](#10-code-skeleton)
11. [Training Workflow](#11-training-workflow)
12. [Connecting RL Back to the Existing Pipeline](#12-connecting-to-existing-pipeline)
13. [Expected Improvements](#13-expected-improvements)
14. [Recommended Libraries](#14-recommended-libraries)

---

## 1. Why RL?

### Current Approach — Rule-Based Battery Dispatch (`battery_model.py`)

```
if solar > load:   CHARGE  (absorb surplus)
else:              DISCHARGE (cover deficit)
```

This heuristic is **reactive and myopic** — it only looks at the *current* timestep. It cannot:

| Limitation | Example Impact |
|---|---|
| No lookahead | Discharges at noon when solar is about to cover demand anyway |
| No cost awareness | Doesn't distinguish between peak-tariff and off-peak grid imports |
| No blackout prevention | Doesn't preserve SOC to survive the coming night's demand |
| Cannot learn patterns | Repeats the same mistake every day |

### What RL Provides

An RL **agent** learns a *policy* — a mapping from the current grid state to an optimal battery action — by interacting with the **digital twin environment** (the simulation you already built) and maximising a cumulative reward signal.

```
Agent ──action──> Digital Twin Environment ──reward──> Agent
       <──state──                          <──next state──
```

The agent discovers: *"If I keep SOC above 40% before sunset, blackouts drop by 60%"* — something no hand-crafted rule can easily express.

---

## 2. RL Concepts Applied to the Microgrid

| RL Term | Microgrid Meaning |
|---|---|
| **Environment** | The digital twin (`digital_twin.py` + `grid_simulator.py`) |
| **Agent** | Trained neural network (policy network) |
| **State s(t)** | Snapshot of the grid at hour `t` |
| **Action a(t)** | How much power to charge / discharge the battery |
| **Reward r(t)** | Penalise blackouts, voltage violations, and grid cost |
| **Episode** | One full simulation run (e.g., 30 days = 720 timesteps) |
| **Policy pi(s)** | The learned decision function: `action = pi(state)` |

---

## 3. Architecture Overview

```
+-------------------------------------------------------------+
|                      TRAINING LOOP                         |
|                                                             |
|  +--------------+   action a(t)  +------------------+      |
|  |   RL Agent   |--------------->|  MicrogridEnv    |      |
|  |  (PPO / SAC) |                |  (wraps existing |      |
|  |              |<---------------|   simulation)    |      |
|  +--------------+  state, reward +------------------+      |
|                                        |                    |
|                               +--------v---------+         |
|                               |  grid_simulator  |         |
|                               |  battery_model   |         |
|                               |  label_generator |         |
|                               +------------------+         |
+-------------------------------------------------------------+
                        |
              After training
                        |
                        v
          +---------------------------+
          |  Trained Policy Model     |
          |  replaces rule-based      |
          |  battery dispatch in      |
          |  digital_twin.py          |
          +---------------------------+
                        |
                        v
          +---------------------------+
          |  ml_model.py              |
          |  uses RL-optimised data   |
          |  (fewer blackouts ->      |
          |   better classifier)      |
          +---------------------------+
```

---

## 4. The RL Environment

The environment is a **Gymnasium-compatible** class that wraps your existing simulation modules.  
It must implement three methods:

| Method | Purpose |
|---|---|
| `reset()` | Start a new episode, return initial state |
| `step(action)` | Apply action, run load flow, return `(next_state, reward, done, info)` |
| `render()` | Optional: visualise the grid state |

**File to create:** `rl_environment.py`

---

## 5. State Space

The state vector gives the agent everything it needs to make a decision at hour `t`.

### Recommended State Vector (12 dimensions)

| # | Feature | Source | Why it matters |
|---|---|---|---|
| 1 | `soc` | `battery_model.soc` | Current energy reserve |
| 2 | `p_load_mw` (normalised) | `data_load` | How much demand to meet |
| 3 | `p_solar_mw` (normalised) | `data_solar` | Free energy available |
| 4 | `v_min` | `grid_simulator` | Voltage health |
| 5 | `hour_of_day / 23` | timestep | Time-of-day pattern |
| 6 | `p_load_lag_1h` | `digital_twin` | Short-term load trend |
| 7 | `solar_rolling_avg_6h` | `digital_twin` | Solar forecast proxy |
| 8 | `load_rolling_avg_6h` | `digital_twin` | Load forecast proxy |
| 9 | `grid_stress_index` | `digital_twin` | Composite risk signal |
| 10 | `soc_rate_of_change` | `digital_twin` | Battery depletion speed |
| 11 | `load_solar_diff` (normalised) | `digital_twin` | Net power gap |
| 12 | `line_loading_max / 100` | `grid_simulator` | Thermal stress |

> **Note:** All 11 temporal features already computed by `add_temporal_features()` in `digital_twin.py` are directly reusable here without any extra work.

---

## 6. Action Space

### Option A — Continuous Action (Recommended for SAC)

```python
action in [-1.0, +1.0]
```
- `-1.0`  -> charge at full rate (0.25 MW)
- `+1.0`  -> discharge at full rate (0.25 MW)
- `0.0`   -> idle

The actual power is: `p_battery = action x discharge_rate`

### Option B — Discrete Action (Simpler, good for DQN)

```python
action in {0, 1, 2}
```
- `0` -> charge
- `1` -> idle
- `2` -> discharge

Start with Option B if you are new to RL; switch to Option A for finer control.

---

## 7. Reward Function Design

The reward is the most important design decision. It must encode what "good" operation means.

### Recommended Composite Reward

```
r(t) = r_blackout + r_voltage + r_cost + r_soc
```

| Component | Formula | Rationale |
|---|---|---|
| **Blackout penalty** | `-10.0 if blackout else 0` | Hard penalty for the primary failure event |
| **Voltage reward** | `+(v_min - 0.95) x 5` if v_min >= 0.95 else `-5` | Keep voltage in safe band |
| **Grid import cost** | `-p_grid_mw x 0.1` | Minimise expensive grid purchases |
| **SOC health bonus** | `+0.5 x soc` if `0.2 <= soc <= 0.8` else `-1.0` | Prevent deep discharge |

### Implementation

```python
def _compute_reward(self, p_load, p_solar, p_battery, v_min, blackout, soc):
    reward = 0.0

    # 1. Blackout — hard penalty
    if blackout:
        reward -= 10.0

    # 2. Voltage — reward safe voltage, penalise violation
    if v_min >= 0.95:
        reward += (v_min - 0.95) * 5.0
    else:
        reward -= 5.0

    # 3. Grid import cost — encourage solar + battery self-sufficiency
    p_grid = max(0, p_load - p_solar - p_battery)
    reward -= p_grid * 0.1

    # 4. SOC health — keep battery in a healthy operating range
    if 0.2 <= soc <= 0.8:
        reward += 0.5 * soc
    else:
        reward -= 1.0

    return reward
```

> **Tip:** Start with just the blackout penalty. Add more terms gradually to avoid reward shaping conflicts.

---

## 8. Algorithm Choice

| Algorithm | Type | Best For | Difficulty |
|---|---|---|---|
| **DQN** | Discrete actions | Getting started, Option B actions | Low |
| **PPO** | Continuous or discrete | Stable training, good first choice | Medium |
| **SAC** | Continuous actions | Best final performance, sample-efficient | Medium-High |
| **TD3** | Continuous actions | Alternative to SAC, lower variance | Medium-High |

### Recommendation for This Project

**Start with PPO** (Proximal Policy Optimisation):
- Works for both discrete and continuous actions
- Very stable to train
- Available out-of-the-box in `stable-baselines3`
- Battery dispatch with ~720 steps/episode converges well with PPO

**Upgrade to SAC** once PPO is working, for finer continuous dispatch control.

---

## 9. Step-by-Step Integration Plan

### Phase 1 — Environment Setup (Day 1-2)

- [ ] Create `rl_environment.py` with `MicrogridEnv(gym.Env)` class
- [ ] Connect it to `grid_simulator.py`, `battery_model.py`, `data_load.py`, `data_solar.py`
- [ ] Validate with `gym.utils.env_checker.check_env(env)`
- [ ] Run 1 random episode to confirm state/reward flow

### Phase 2 — Baseline Agent (Day 3-4)

- [ ] Install `stable-baselines3` and `gymnasium`
- [ ] Train a PPO agent for 50,000 steps (quick smoke test)
- [ ] Plot reward curve — should be rising
- [ ] Compare blackout count: RL agent vs. rule-based battery

### Phase 3 — Hyperparameter Tuning (Day 5-7)

- [ ] Tune reward weights (blackout penalty, SOC bonus)
- [ ] Tune PPO hyperparameters: `learning_rate`, `n_steps`, `ent_coef`
- [ ] Train for 200,000-500,000 steps with the best config
- [ ] Evaluate on held-out test data (last 7 days)

### Phase 4 — Integration (Day 8-10)

- [ ] Save the trained policy (`model.save("outputs/rl_policy")`)
- [ ] Modify `digital_twin.py` to use RL policy instead of rule-based dispatch
- [ ] Re-run full simulation -> export enriched CSV
- [ ] Feed the improved data back into `ml_model.py` -> blackout predictor improves too

### Phase 5 — Evaluation and Reporting (Day 11-14)

- [ ] Compute LOLP and EENS for: rule-based vs. RL agent
- [ ] Plot SOC profile, voltage profile, blackout events
- [ ] Document policy's learned behaviour (e.g., charges pre-emptively at dawn)

---

## 10. Code Skeleton

### `rl_environment.py` — Full Template

```python
"""
Gymnasium environment for Microgrid RL Battery Dispatch.

Wraps the existing digital twin simulation components to provide
a standard gym.Env interface for RL agents.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from battery_model import BatteryModel
from grid_simulator import create_ieee33_network, run_load_flow
from label_generator import generate_blackout_label
from data_load import prepare_load_series
from data_solar import prepare_solar_series


class MicrogridEnv(gym.Env):
    """
    Gymnasium environment for microgrid battery dispatch optimisation.

    Observation (12 features, all normalised to [0,1] or [-1,1]):
        soc, p_load_norm, p_solar_norm, v_min, hour_norm,
        p_load_lag1, solar_avg6h, load_avg6h,
        stress_index, soc_roc, load_solar_diff, line_loading_norm

    Action (continuous):
        Float in [-1, +1]  ->  fraction of max charge/discharge rate
        Negative = charge, Positive = discharge
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        load_path: str,
        solar_path: str,
        n_days: int = 30,
        battery_capacity: float = 0.5,
    ):
        super().__init__()

        self.p_load_series  = prepare_load_series(load_path, n_days)
        self.p_solar_series = prepare_solar_series(solar_path, n_days)
        T = min(len(self.p_load_series), len(self.p_solar_series))
        self.p_load_series  = self.p_load_series.iloc[:T].values
        self.p_solar_series = self.p_solar_series.iloc[:T].values
        self.T = T
        self.battery_capacity = battery_capacity
        self.net = create_ieee33_network()
        self.max_load  = float(self.p_load_series.max())  or 1.0
        self.max_solar = float(self.p_solar_series.max()) or 1.0

        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._load_history  = []
        self._solar_history = []
        self._soc_history   = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.battery = BatteryModel(
            capacity_mwh=self.battery_capacity, soc_init=0.5
        )
        self.t = 0
        self._load_history  = []
        self._solar_history = []
        self._soc_history   = []
        return self._get_obs(), {}

    def step(self, action):
        max_rate = self.battery.charge_rate
        p_cmd    = float(action[0]) * max_rate

        p_load  = float(self.p_load_series[self.t])
        p_solar = float(self.p_solar_series[self.t])

        p_battery, soc = self._apply_action(p_cmd, p_load, p_solar)
        lf = run_load_flow(self.net, p_load, p_solar)

        blackout = generate_blackout_label(
            p_load, p_solar, p_battery, lf['v_min'], lf['converged']
        )
        reward = self._compute_reward(
            p_load, p_solar, p_battery, lf['v_min'], blackout, soc
        )

        self._load_history.append(p_load)
        self._solar_history.append(p_solar)
        self._soc_history.append(soc)

        self.t += 1
        done = self.t >= self.T
        obs  = self._get_obs() if not done else np.zeros(12, dtype=np.float32)
        info = {'blackout': blackout, 'soc': soc, 'v_min': lf['v_min']}
        return obs, reward, done, False, info

    def _apply_action(self, p_cmd, p_load, p_solar):
        if p_cmd < 0:
            p_charge = min(abs(p_cmd), self.battery.charge_rate,
                           (self.battery.soc_max - self.battery.soc) * self.battery.capacity)
            self.battery.soc += (p_charge * self.battery.eta) / self.battery.capacity
            p_battery = -p_charge
        else:
            p_discharge = min(p_cmd, self.battery.discharge_rate,
                              (self.battery.soc - self.battery.soc_min) * self.battery.capacity)
            self.battery.soc -= p_discharge / (self.battery.capacity * self.battery.eta)
            p_battery = p_discharge
        self.battery.soc = np.clip(self.battery.soc, self.battery.soc_min, self.battery.soc_max)
        return p_battery, self.battery.soc

    def _compute_reward(self, p_load, p_solar, p_battery, v_min, blackout, soc):
        reward = 0.0
        if blackout:
            reward -= 10.0
        if v_min >= 0.95:
            reward += (v_min - 0.95) * 5.0
        else:
            reward -= 5.0
        p_grid = max(0.0, p_load - p_solar - p_battery)
        reward -= p_grid * 0.1
        if 0.2 <= soc <= 0.8:
            reward += 0.5 * soc
        else:
            reward -= 1.0
        return float(reward)

    def _get_obs(self):
        t       = self.t if self.t < self.T else self.T - 1
        p_load  = float(self.p_load_series[t])
        p_solar = float(self.p_solar_series[t])
        soc     = self.battery.soc

        hist_load  = self._load_history[-6:]  if self._load_history  else [p_load]
        hist_solar = self._solar_history[-6:] if self._solar_history else [p_solar]
        hist_soc   = self._soc_history[-1:]   if self._soc_history   else [soc]

        load_avg6  = float(np.mean(hist_load))
        solar_avg6 = float(np.mean(hist_solar))
        soc_roc    = float(soc - hist_soc[-1]) if hist_soc else 0.0

        lf = run_load_flow(self.net, p_load, p_solar)

        return np.array([
            soc,
            p_load  / self.max_load,
            p_solar / self.max_solar,
            lf['v_min'],
            (t % 24) / 23.0,
            self._load_history[-1] / self.max_load if self._load_history else 0.0,
            solar_avg6 / self.max_solar,
            load_avg6  / self.max_load,
            (p_load / self.max_load) + (1 - p_solar / self.max_solar) + (1 - soc),
            soc_roc,
            (p_load - p_solar) / self.max_load,
            lf['line_loading_max'] / 100.0,
        ], dtype=np.float32)
```

---

### `rl_train.py` — Training Script Template

```python
"""
Train a PPO agent to control microgrid battery dispatch.

Usage:
    python rl_train.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from rl_environment import MicrogridEnv

LOAD_PATH  = r'datasets\LD2011_2014.txt'
SOLAR_PATH = r'datasets\Plant_1_Generation_Data.csv'

def make_env():
    return MicrogridEnv(LOAD_PATH, SOLAR_PATH, n_days=30)

train_env = make_vec_env(make_env, n_envs=4)
eval_env  = make_env()

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    tensorboard_log="outputs/rl_logs/",
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="outputs/rl_policy/",
    log_path="outputs/rl_logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

print("Starting PPO training...")
model.learn(total_timesteps=300_000, callback=eval_callback)
model.save("outputs/rl_policy/final_ppo_model")
print("Training complete. Model saved.")
```

---

### Integrating the Trained Policy into `digital_twin.py`

Replace the rule-based battery step inside the simulation loop:

```python
from stable_baselines3 import PPO

# Load trained policy once (before the loop)
rl_policy = PPO.load("outputs/rl_policy/best_model")

# Inside the timestep loop — replaces: p_battery, soc = battery.step(p_load, p_solar)
obs = env._get_obs()
action, _ = rl_policy.predict(obs, deterministic=True)
p_battery, soc = env._apply_action(float(action[0]), p_load, p_solar)
```

---

## 11. Training Workflow

```
TRAINING PROGRESSION
-----------------------------------------
1. Random Policy (0 steps)
   -> High blackout rate, random charge/discharge

2. Early Learning (~50k steps)
   -> Agent learns "discharge at night"
   -> Blackout rate starts falling

3. Mid Training (~150k steps)
   -> Learns "pre-charge before sunset"
   -> Voltage violations minimised

4. Convergence (~300k steps)
   -> Stable policy: LOLP significantly lower
   -> SOC profile shows strategic behaviour
-----------------------------------------
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir outputs/rl_logs/
```

Watch these metrics:
- `rollout/ep_rew_mean` — average episode reward (should rise)
- `rollout/ep_len_mean` — should stay ~720 (full episode = no crashes)
- `train/entropy_loss` — should decrease as policy converges

---

## 12. Connecting to the Existing Pipeline

The RL agent fits **cleanly** into your existing architecture:

```
data_load.py  ---+
                 +---> MicrogridEnv (rl_environment.py)
data_solar.py ---+           |
                             | uses
                    +--------v--------------------+
                    | battery_model.py (physics)  |
                    | grid_simulator.py (load flow)|
                    | label_generator.py (labels) |
                    +-----------------------------+
                             |
                      RL-optimised data
                             |
                    digital_twin.py  (run_simulation with RL policy)
                             |
                    simulation_results.csv
                             |
                    ml_model.py  (blackout predictor — now sees fewer
                                  blackouts and smarter SOC patterns)
```

**The ML classifier in `ml_model.py` benefits directly:**
- Fewer blackouts -> predictor trained on more balanced data
- RL-dispatched battery creates more informative SOC patterns
- `grid_stress_index` more accurately reflects risk under smart dispatch

---

## 13. Expected Improvements

Based on similar microgrid RL studies, you can expect:

| Metric | Rule-Based | RL (PPO/SAC) | Improvement |
|---|---|---|---|
| **LOLP** | ~10-15% | ~3-6% | ~50-70% reduction |
| **EENS** | High | Significantly lower | ~40-60% reduction |
| **Battery cycles/day** | Uncontrolled | Optimised | Longer battery life |
| **Grid import cost** | Baseline | Lower (peak-shaving) | ~15-25% savings |
| **Voltage violations** | Reactive | Proactive | Fewer dips |

> **Note:** Actual results depend on your dataset characteristics. Run baseline vs. RL comparisons and report LOLP/EENS from `compute_reliability_metrics()` in `ml_model.py`.

---

## 14. Recommended Libraries

Install into your existing `venv`:

```bash
pip install gymnasium stable-baselines3[extra] tensorboard
```

| Library | Version | Purpose |
|---|---|---|
| `gymnasium` | >= 0.29 | Standard RL environment interface |
| `stable-baselines3` | >= 2.3 | PPO, SAC, DQN implementations |
| `tensorboard` | >= 2.15 | Training curve visualisation |
| `torch` | >= 2.0 | Already needed for LSTM in `ml_model.py` |

---

## Quick Start Checklist

```
[ ] pip install gymnasium stable-baselines3[extra] tensorboard
[ ] Create rl_environment.py  (copy skeleton from Section 10)
[ ] Create rl_train.py        (copy skeleton from Section 10)
[ ] Run: python rl_train.py   (first smoke test, 50k steps)
[ ] Plot reward curve in TensorBoard
[ ] Compare LOLP: rule-based vs. RL agent
[ ] Integrate trained policy into digital_twin.py
[ ] Re-run ml_model.py on RL-optimised simulation data
```

---

*Guide written for the Microgrid Digital Twin project — IEEE 33-bus, Battery 0.5 MWh, 30-day simulation horizon.*
