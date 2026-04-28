"""
rl_train.py — Train a PPO Agent for Microgrid Battery Dispatch.

Trains a PPO agent inside MicrogridEnv (lookup-table version).
Training completes in ~5 minutes using the pre-computed load flow table.

Usage
-----
    python rl_precompute.py          # step 1 — run once
    python rl_train.py               # step 2 — train agent
    python rl_evaluate.py            # step 3 — evaluate vs rule-based

TensorBoard
-----------
    tensorboard --logdir outputs/rl_logs/
"""

import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_environment import MicrogridEnv


# ── Paths ─────────────────────────────────────────────────────────────────────

LOAD_PATH  = 'datasets/electricityloaddiagrams20112014/LD2011_2014.txt'
SOLAR_PATH = 'datasets/Solar Power Generation Data/Plant_1_Generation_Data.csv'
POLICY_DIR = 'outputs/rl_policy'
LOG_DIR    = 'outputs/rl_logs'


# ── Episode stats callback ────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):

    def __init__(self, log_every: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.log_every   = log_every
        self._ep_count   = 0
        self._rewards:   list = []
        self._blackouts: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._ep_count += 1
                self._rewards.append(info['episode']['r'])
                self._blackouts.append(info.get('episode_blackouts', 0))

                if self._ep_count % self.log_every == 0 and self.verbose:
                    mean_r  = np.mean(self._rewards[-self.log_every:])
                    mean_bo = np.mean(self._blackouts[-self.log_every:])
                    print(
                        f"  [Ep {self._ep_count:5d}] "
                        f"mean_reward={mean_r:+8.2f}  "
                        f"mean_blackouts={mean_bo:.1f}"
                    )
        return True


# ── Environment factory ───────────────────────────────────────────────────────

def _make_env(load_path, solar_path, n_days):
    def _init():
        env = MicrogridEnv(load_path, solar_path, n_days=n_days)
        return Monitor(env)
    return _init


# ── Train ─────────────────────────────────────────────────────────────────────

def train(
    timesteps: int   = 300_000,
    n_days:    int   = 7,
    n_envs:    int   = 4,
    lr:        float = 3e-4,
    seed:      int   = 42,
) -> PPO:

    os.makedirs(POLICY_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    steps_per_ep   = n_days * 24
    total_episodes = timesteps // steps_per_ep

    print("=" * 60)
    print("  MICROGRID RL TRAINING — PPO")
    print("=" * 60)
    print(f"  Timesteps     : {timesteps:,}")
    print(f"  Days/episode  : {n_days}  ({steps_per_ep} steps/ep)")
    print(f"  Total episodes: ~{total_episodes}")
    print(f"  Parallel envs : {n_envs}")
    print(f"  Learning rate : {lr}")
    print(f"  Device        : cpu")
    print(f"  Policy dir    : {POLICY_DIR}")
    print(f"  Log dir       : {LOG_DIR}")
    print()

    train_env = make_vec_env(
        _make_env(LOAD_PATH, SOLAR_PATH, n_days),
        n_envs=n_envs,
        seed=seed,
    )
    eval_env = Monitor(MicrogridEnv(LOAD_PATH, SOLAR_PATH, n_days=n_days))

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = lr,
        n_steps         = 1024,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        ent_coef        = 0.005,
        clip_range      = 0.2,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        seed            = seed,
        tensorboard_log = LOG_DIR,
        verbose         = 0,
        device          = 'cpu',
        policy_kwargs   = dict(
            net_arch = dict(pi=[128, 128], vf=[128, 128]),
        ),
    )

    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path = POLICY_DIR,
        log_path             = LOG_DIR,
        eval_freq            = max(steps_per_ep, 500),
        n_eval_episodes      = 3,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )
    stats_cb = EpisodeStatsCallback(log_every=10, verbose=1)
    callback = CallbackList([eval_cb, stats_cb])

    print("Starting training...")
    print(f"Monitor: tensorboard --logdir {LOG_DIR}/\n")

    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)

    final_path = os.path.join(POLICY_DIR, 'final_ppo_model')
    model.save(final_path)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best model  : {POLICY_DIR}/best_model.zip")
    print(f"  Final model : {final_path}.zip")
    print(f"  Logs        : tensorboard --logdir {LOG_DIR}/")

    train_env.close()
    eval_env.close()
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO agent for microgrid dispatch')
    parser.add_argument('--timesteps', type=int,   default=300_000)
    parser.add_argument('--n_days',    type=int,   default=7)
    parser.add_argument('--n_envs',    type=int,   default=4)
    parser.add_argument('--lr',        type=float, default=3e-4)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    train(
        timesteps = args.timesteps,
        n_days    = args.n_days,
        n_envs    = args.n_envs,
        lr        = args.lr,
        seed      = args.seed,
    )