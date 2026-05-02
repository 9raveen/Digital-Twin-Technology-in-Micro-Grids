"""
compare_zip_models.py
====================
Compare ML model performance between ZIP and Constant Power load models.

This script:
1. Generates simulation data with ZIP=False (constant power loads)
2. Generates simulation data with ZIP=True (voltage-dependent loads)
3. Trains regression models on both datasets
4. Compares accuracy, RMSE, R² across different load models

Output: Comparison table and analysis
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import pipeline modules
from digital_twin import run_simulation, print_summary
import ml_model


# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = 'outputs/zip_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_comparison_pipeline():
    """
    Run full comparison: constant power vs ZIP models.
    """
    print("\n" + "=" * 80)
    print("  ZIP vs Constant Power Load Model Comparison")
    print("=" * 80)

    results = {}

    # ── Run Simulation with Constant Power ────────────────────────────────────
    print("\n" + "-" * 80)
    print("  PHASE 1: Constant Power Load Model")
    print("-" * 80)

    df_constant = run_simulation(use_zip=False)
    print_summary(df_constant)

    # Save constant power dataset
    csv_constant = os.path.join(OUTPUT_DIR, 'simulation_constant_power.csv')
    df_constant.to_csv(csv_constant, index=False)
    print(f"\nSaved → {csv_constant}")

    results['constant'] = {
        'df': df_constant,
        'csv_path': csv_constant,
    }

    # ── Run Simulation with ZIP ────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  PHASE 2: ZIP Load Model (Voltage-Dependent)")
    print("-" * 80)

    df_zip = run_simulation(use_zip=True)
    print_summary(df_zip)

    # Save ZIP dataset
    csv_zip = os.path.join(OUTPUT_DIR, 'simulation_zip_loads.csv')
    df_zip.to_csv(csv_zip, index=False)
    print(f"\nSaved → {csv_zip}")

    results['zip'] = {
        'df': df_zip,
        'csv_path': csv_zip,
    }

    # ── Train Models on Both Datasets ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PHASE 3: Train ML Models")
    print("=" * 80)

    for mode_name, mode_data in results.items():
        print(f"\n  Training on {mode_name.upper()} dataset...")

        df = mode_data['df']

        # Prepare data
        X_train, X_test, y_train, y_test, _ = ml_model.load_and_prepare(mode_data['csv_path'])

        # Train models
        ml_results, scaler, best_name = ml_model.train_and_evaluate(
            X_train, X_test, y_train, y_test
        )

        mode_data['models'] = ml_results
        mode_data['scaler'] = scaler
        mode_data['best_model'] = best_name
        mode_data['X_test'] = X_test
        mode_data['y_test'] = y_test

    # ── Compare Results ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  COMPARISON: Constant Power vs ZIP Model")
    print("=" * 80)

    comparison_data = []
    for mode_name, mode_data in results.items():
        best_name = mode_data['best_model']
        best_result = mode_data['models'][best_name]

        comparison_data.append({
            'Load_Model': 'Constant Power' if mode_name == 'constant' else 'ZIP (V-dependent)',
            'Best_Regressor': best_name,
            'R²_Score': best_result['r2'],
            'RMSE': best_result['rmse'],
            'MAE': best_result['mae'],
            'CV_R²_Mean': best_result['cv_r2_mean'],
        })

    df_comparison = pd.DataFrame(comparison_data)

    print("\n" + df_comparison.to_string(index=False))

    # Save comparison table
    comparison_csv = os.path.join(OUTPUT_DIR, 'comparison_results.csv')
    df_comparison.to_csv(comparison_csv, index=False)
    print(f"\nSaved → {comparison_csv}")

    # ── Analysis ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ANALYSIS")
    print("=" * 80)

    constant_r2 = comparison_data[0]['R²_Score']
    zip_r2 = comparison_data[1]['R²_Score']
    r2_improvement = (zip_r2 - constant_r2) / constant_r2 * 100

    print(f"\n  R² Score Improvement (ZIP vs Constant):")
    print(f"    Constant Power R² : {constant_r2:.4f}")
    print(f"    ZIP Model R²      : {zip_r2:.4f}")
    print(f"    Improvement       : {r2_improvement:+.2f}%")

    if r2_improvement > 2:
        print(f"\n  ✓ ZIP model shows SIGNIFICANT improvement in accuracy")
        print(f"    → Voltage-dependent loads improve risk prediction")
    elif r2_improvement > 0:
        print(f"\n  ✓ ZIP model shows MODEST improvement")
        print(f"    → Voltage-dependent effect is present but small")
    else:
        print(f"\n  ✗ ZIP model shows NO improvement")
        print(f"    → Constant power model is sufficient for this dataset")

    # Analyze voltage behavior differences
    print(f"\n  Voltage Statistics Comparison:")
    print(f"\n    Constant Power Model:")
    print(f"      V_min mean   : {results['constant']['df']['v_min'].mean():.4f} pu")
    print(f"      V_min std    : {results['constant']['df']['v_min'].std():.4f} pu")
    print(f"      Risk mean    : {results['constant']['df']['risk_score'].mean():.4f}")

    print(f"\n    ZIP Load Model:")
    print(f"      V_min mean   : {results['zip']['df']['v_min'].mean():.4f} pu")
    print(f"      V_min std    : {results['zip']['df']['v_min'].std():.4f} pu")
    print(f"      Risk mean    : {results['zip']['df']['risk_score'].mean():.4f}")

    dv_min = results['zip']['df']['v_min'].mean() - results['constant']['df']['v_min'].mean()
    print(f"\n    Voltage Shift (ZIP - Constant): {dv_min:+.4f} pu")

    if abs(dv_min) > 0.01:
        print(f"    → Significant voltage behavior change with ZIP loads")
    else:
        print(f"    → Minimal voltage behavior change with ZIP loads")

    return results, df_comparison


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results, comparison_df = run_comparison_pipeline()

    print("\n" + "=" * 80)
    print("  Comparison Complete")
    print("=" * 80)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Files saved:")
    print(f"    - simulation_constant_power.csv")
    print(f"    - simulation_zip_loads.csv")
    print(f"    - comparison_results.csv")
