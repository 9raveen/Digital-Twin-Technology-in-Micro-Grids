"""
hyperparameter_optimization.py
=============================
Grid search and Bayesian optimization for battery and ML hyperparameters.
"""

import numpy as np
import pandas as pd
from itertools import product
import time

from battery_model import BatteryModel
from config import (
    BATTERY_CAPACITY_MWH, BATTERY_SOC_INIT, BATTERY_SOC_MIN, BATTERY_SOC_MAX,
    BATTERY_CHARGE_RATE_MW, BATTERY_DISCHARGE_RATE_MW, BATTERY_EFFICIENCY,
    BATTERY_RESPONSE_FACTOR, BATTERY_TARGET_SOC,
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    GB_N_ESTIMATORS, GB_LEARNING_RATE, GB_MAX_DEPTH
)


class BatteryHyperparameterOptimizer:
    """
    Grid search for optimal battery control parameters.

    Parameters to optimize:
    - response_factor: 0.3 - 0.8 (lower = slower response)
    - target_soc: 0.4 - 0.8 (where to keep battery)
    - charge/discharge rates: 0.15 - 0.35 MW
    """

    def __init__(self, simulation_func, test_scenarios: list):
        """
        Parameters
        ----------
        simulation_func : callable
            Function that runs simulation and returns metrics dict
        test_scenarios : list
            List of (p_load, p_solar) tuples to test
        """
        self.simulation_func = simulation_func
        self.test_scenarios = test_scenarios
        self.results = []

    def optimize(self, param_grid: dict, metric: str = 'blackout_rate',
                 minimize: bool = False) -> pd.DataFrame:
        """
        Run grid search over battery parameters.

        Parameters
        ----------
        param_grid : dict
            Grid of parameters to search, e.g.:
            {'response_factor': [0.3, 0.5, 0.7],
             'target_soc': [0.4, 0.6, 0.8],
             'charge_rate': [0.2, 0.25, 0.3]}
        metric : str
            Metric to optimize ('blackout_rate', 'avg_soc_volatility', 'grid_stress')
        minimize : bool
            If True, minimize metric; else maximize

        Returns
        -------
        pd.DataFrame
            Results of all evaluations sorted by metric
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"\n{'='*80}")
        print(f"BATTERY HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Grid search space: {len(combinations)} combinations")
        print(f"Parameters: {param_names}")
        print(f"Metric: {metric} ({'minimize' if minimize else 'maximize'})\n")

        results = []
        t_start = time.time()

        for i, combo in enumerate(combinations):
            param_dict = dict(zip(param_names, combo))

            # Create battery with these parameters
            battery = BatteryModel(
                capacity_mwh=BATTERY_CAPACITY_MWH,
                soc_init=BATTERY_SOC_INIT,
                soc_min=BATTERY_SOC_MIN,
                soc_max=BATTERY_SOC_MAX,
                charge_rate=param_dict.get('charge_rate', BATTERY_CHARGE_RATE_MW),
                discharge_rate=param_dict.get('discharge_rate', BATTERY_DISCHARGE_RATE_MW),
                efficiency=BATTERY_EFFICIENCY,
                response_factor=param_dict.get('response_factor', BATTERY_RESPONSE_FACTOR),
            )
            battery.target_soc = param_dict.get('target_soc', BATTERY_TARGET_SOC)

            # Run simulation
            metrics = self.simulation_func(battery, self.test_scenarios)

            # Record results
            result = {**param_dict, **metrics}
            results.append(result)

            # Progress update
            pct = (i + 1) / len(combinations) * 100
            metric_val = metrics.get(metric, 0)
            print(f"[{i+1:3d}/{len(combinations)}] {pct:5.1f}% | {metric}={metric_val:.4f} | {param_dict}")

        # Sort by metric
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by=metric, ascending=minimize)

        elapsed = time.time() - t_start
        print(f"\n✓ Optimization complete in {elapsed:.1f}s")
        print(f"\nBest parameters:")
        print(df_results.iloc[0])

        return df_results


class MLHyperparameterOptimizer:
    """
    Grid search for ML model hyperparameters.

    Parameters to optimize:
    - Random Forest: n_estimators, max_depth, min_samples_split
    - Gradient Boosting: n_estimators, learning_rate, max_depth
    - XGBoost: n_estimators, learning_rate, max_depth, subsample
    """

    @staticmethod
    def optimize_random_forest(X_train, X_test, y_train, y_test,
                              param_grid: dict = None) -> pd.DataFrame:
        """
        Grid search for Random Forest hyperparameters.

        Parameters
        ----------
        X_train, X_test : pd.DataFrame
            Training and test features
        y_train, y_test : pd.Series
            Training and test targets
        param_grid : dict, optional
            Parameters to search. Default: standard grid

        Returns
        -------
        pd.DataFrame
            Results sorted by test R² score
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 8, 12],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        # Generate combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"\n{'='*80}")
        print(f"RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Grid search space: {len(combinations)} combinations\n")

        results = []
        t_start = time.time()

        for i, combo in enumerate(combinations):
            param_dict = dict(zip(param_names, combo))

            # Train model
            rf = RandomForestRegressor(random_state=42, n_jobs=-1, **param_dict)
            rf.fit(X_train, y_train)

            # Evaluate
            y_pred_train = rf.predict(X_train)
            y_pred_test = rf.predict(X_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            result = {
                **param_dict,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_test': rmse_test,
                'overfit': r2_train - r2_test
            }
            results.append(result)

            pct = (i + 1) / len(combinations) * 100
            print(f"[{i+1:3d}/{len(combinations)}] {pct:5.1f}% | R²={r2_test:.4f} | {param_dict}")

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='r2_test', ascending=False)

        elapsed = time.time() - t_start
        print(f"\n✓ Optimization complete in {elapsed:.1f}s")
        print(f"\nBest hyperparameters:")
        print(df_results.iloc[0])

        return df_results

    @staticmethod
    def optimize_gradient_boosting(X_train, X_test, y_train, y_test,
                                  param_grid: dict = None) -> pd.DataFrame:
        """Grid search for Gradient Boosting hyperparameters."""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_squared_error

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 8],
                'min_samples_split': [2, 5, 10]
            }

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"\n{'='*80}")
        print(f"GRADIENT BOOSTING HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Grid search space: {len(combinations)} combinations\n")

        results = []
        t_start = time.time()

        for i, combo in enumerate(combinations):
            param_dict = dict(zip(param_names, combo))

            gb = GradientBoostingRegressor(random_state=42, **param_dict)
            gb.fit(X_train, y_train)

            y_pred_train = gb.predict(X_train)
            y_pred_test = gb.predict(X_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            result = {
                **param_dict,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_test': rmse_test,
                'overfit': r2_train - r2_test
            }
            results.append(result)

            pct = (i + 1) / len(combinations) * 100
            print(f"[{i+1:3d}/{len(combinations)}] {pct:5.1f}% | R²={r2_test:.4f} | {param_dict}")

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='r2_test', ascending=False)

        elapsed = time.time() - t_start
        print(f"\n✓ Optimization complete in {elapsed:.1f}s")
        print(f"\nBest hyperparameters:")
        print(df_results.iloc[0])

        return df_results


if __name__ == '__main__':
    print("Hyperparameter optimization module ready.")
    print("Use BatteryHyperparameterOptimizer or MLHyperparameterOptimizer in your pipeline.")
