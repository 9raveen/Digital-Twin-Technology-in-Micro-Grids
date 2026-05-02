"""
generate_dataset.py
===================
Main entry point for dataset generation pipeline.

Usage:
    python generate_dataset.py --n-scenarios 50 --output-dir datasets/generated

This script:
  1. Generates scenario configurations
  2. Runs multi-scenario simulations
  3. Validates dataset quality
  4. Splits into train/val/test
  5. Generates analysis report
"""

import argparse
import sys
from pathlib import Path

from dataset_generator import DatasetGenerator
from dataset_validation import DatasetValidator
from dataset_analysis import DatasetAnalyzer


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate large-scale dataset for power system blackout prediction"
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=50,
        help="Number of scenarios to generate (default: 50, ~36k samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/generated",
        help="Output directory for generated datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--load-csv",
        type=str,
        default="datasets/processed/load_scaled.csv",
        help="Path to load CSV (optional)",
    )
    parser.add_argument(
        "--solar-csv",
        type=str,
        default="datasets/processed/solar_scaled.csv",
        help="Path to solar CSV (optional)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("POWER SYSTEM DATASET GENERATION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Scenarios: {args.n_scenarios}")
    print(f"  Output: {args.output_dir}")
    print(f"  Seed: {args.seed}\n")

    # Phase 1: Generate dataset
    print("[PHASE 1] Dataset Generation")
    print("-" * 80)

    generator = DatasetGenerator(
        n_scenarios=args.n_scenarios,
        load_csv_path=args.load_csv,
        solar_csv_path=args.solar_csv,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    df = generator.generate_dataset()

    # Phase 2: Validate
    print("\n[PHASE 2] Dataset Validation")
    print("-" * 80)

    validator = DatasetValidator(df, Path(args.output_dir))
    validator.run_full_validation()

    # Get splits
    df_train, df_val, df_test = validator.split_train_val_test(split_by_scenario=True)
    validator.export_dataset(df_train, df_val, df_test)

    # Phase 3: Analyze
    print("\n[PHASE 3] Dataset Analysis")
    print("-" * 80)

    analyzer = DatasetAnalyzer(df, Path(args.output_dir))
    analyzer.compute_feature_statistics()
    analyzer.compute_diversity_score()
    analyzer.detect_outliers()
    analyzer.suggest_overfitting_detection()
    analyzer.generate_report()

    # Summary
    print("\n" + "=" * 80)
    print("✓ DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated dataset statistics:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Train samples: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Val samples: {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Test samples: {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
    print(f"\n  Blackout rate (train): {df_train['blackout'].mean():.1%}")
    print(f"  Blackout rate (val): {df_val['blackout'].mean():.1%}")
    print(f"  Blackout rate (test): {df_test['blackout'].mean():.1%}")

    print(f"\nOutput files:")
    output_path = Path(args.output_dir)
    print(f"  - {output_path / 'train.csv'}")
    print(f"  - {output_path / 'val.csv'}")
    print(f"  - {output_path / 'test.csv'}")
    print(f"  - {output_path / 'dataset_info.txt'}")
    print(f"  - {output_path / 'feature_statistics.csv'}")
    print(f"  - {output_path / 'analysis_report.html'}")
    print(f"  - {output_path / 'scenario_configs.json'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
