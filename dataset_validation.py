"""
dataset_validation.py
====================
Quality assurance and export for generated datasets.

Functions:
  - Validate no data leakage
  - Check class balance
  - Check feature diversity
  - Split into train/val/test while avoiding temporal leakage
  - Export to CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DatasetValidator:
    """Verify dataset quality and export in train/val/test splits."""

    def __init__(self, df: pd.DataFrame, output_dir: Path = Path("datasets/generated")):
        """
        Initialize validator.

        Parameters
        ----------
        df : pd.DataFrame
            Combined dataset from DatasetGenerator
        output_dir : Path
            Output directory for splits
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_no_leakage(self) -> bool:
        """
        Check that features don't leak label information.

        Features that should NOT be in ML model:
          - v_min, v_max, v_mean (post-hoc system state)
          - converged (post-hoc load flow result)
        """
        print("\n[VALIDATION] Checking for data leakage...")

        leakage_features = ["v_min", "v_max", "v_mean", "converged"]
        has_leakage = False

        for feat in leakage_features:
            if feat not in self.df.columns:
                continue

            corr = self.df[feat].corr(self.df["blackout"])
            print(f"  {feat:20s} ↔ blackout: {corr:+.4f}")

            if abs(corr) > 0.80:
                print(f"    ⚠️  LEAKAGE WARNING: correlation too high!")
                has_leakage = True

        if not has_leakage:
            print("  ✓ No leakage detected")

        return not has_leakage

    def check_class_balance(self) -> dict:
        """Report blackout rate; warn if extreme."""
        print("\n[VALIDATION] Checking class balance...")

        blackout_count = self.df["blackout"].sum()
        total_count = len(self.df)
        rate = blackout_count / total_count

        print(f"  Blackout samples: {blackout_count:,} / {total_count:,}")
        print(f"  Blackout rate: {rate*100:.1f}%")

        if rate < 0.05:
            print(f"  ⚠️  Very low blackout rate (<5%) — model may not learn")
        elif rate > 0.50:
            print(f"  ⚠️  Very high blackout rate (>50%) — class imbalance")
        else:
            print(f"  ✓ Class balance reasonable")

        return {"blackout_rate": rate, "normal_rate": 1 - rate}

    def check_diversity(self) -> dict:
        """Measure coverage of feature space."""
        print("\n[VALIDATION] Checking dataset diversity...")

        diversity = {
            "v_min_range": (self.df["v_min"].min(), self.df["v_min"].max()),
            "load_range": (self.df["p_load_mw"].min(), self.df["p_load_mw"].max()),
            "solar_range": (self.df["p_solar_mw"].min(), self.df["p_solar_mw"].max()),
            "soc_range": (self.df["soc"].min(), self.df["soc"].max()),
            "risk_range": (self.df["risk_score"].min(), self.df["risk_score"].max()),
        }

        print(f"  Voltage: [{diversity['v_min_range'][0]:.4f}, {diversity['v_min_range'][1]:.4f}] pu")
        print(f"  Load:    [{diversity['load_range'][0]:.2f}, {diversity['load_range'][1]:.2f}] MW")
        print(f"  Solar:   [{diversity['solar_range'][0]:.2f}, {diversity['solar_range'][1]:.2f}] MW")
        print(f"  SOC:     [{diversity['soc_range'][0]:.3f}, {diversity['soc_range'][1]:.3f}]")
        print(f"  Risk:    [{diversity['risk_range'][0]:.3f}, {diversity['risk_range'][1]:.3f}]")

        if "scenario_type" in self.df.columns:
            print(f"\n  Scenario distribution:")
            for stype, count in self.df["scenario_type"].value_counts().items():
                pct = count / len(self.df) * 100
                print(f"    {stype:15s}: {count:6,} samples ({pct:5.1f}%)")

        print(f"  ✓ Diversity check complete")

        return diversity

    def verify_realism(self) -> dict:
        """Sanity checks on physical plausibility."""
        print("\n[VALIDATION] Verifying realism...")

        issues = []

        # Power balance check
        p_grid_calc = self.df["p_load_mw"] - self.df["p_solar_mw"] - self.df["p_battery_mw"]
        p_grid_diff = np.abs(p_grid_calc - self.df["p_grid_mw"]).max()

        if p_grid_diff > 0.01:
            issues.append(f"Power balance error: max diff = {p_grid_diff:.4f} MW")
        else:
            print(f"  ✓ Power balance OK (max diff = {p_grid_diff:.6f} MW)")

        # SOC bounds
        soc_valid = (self.df["soc"] >= 0.0) & (self.df["soc"] <= 1.0)
        if not soc_valid.all():
            issues.append(f"SOC out of bounds: {(~soc_valid).sum()} samples")
        else:
            print(f"  ✓ SOC bounds OK")

        # Voltage bounds (relaxed for measurement noise)
        v_valid = (self.df["v_min"] >= 0.7) & (self.df["v_min"] <= 1.15)
        if not v_valid.all():
            issues.append(f"Voltage out of physical bounds: {(~v_valid).sum()} samples")
        else:
            print(f"  ✓ Voltage bounds OK")

        # NaN check
        nan_count = self.df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")
        else:
            print(f"  ✓ No NaN values")

        # Infinity check
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")
        else:
            print(f"  ✓ No infinite values")

        if issues:
            print(f"\n  ⚠️  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            return {"status": "failed", "issues": issues}
        else:
            print(f"  ✓ All realism checks passed")
            return {"status": "passed"}

    def split_train_val_test(
        self, split_by_scenario: bool = True, train_frac: float = 0.5, val_frac: float = 0.175
    ) -> tuple:
        """
        Split data avoiding temporal leakage.

        Parameters
        ----------
        split_by_scenario : bool
            If True, split by scenario ID (recommended to avoid temporal leakage)
            If False, split by timestep within scenarios
        train_frac : float
            Fraction for training (default 0.5 = 50%)
        val_frac : float
            Fraction for validation (default 0.175 = 17.5%)

        Returns
        -------
        tuple (df_train, df_val, df_test)
        """
        print("\n[SPLIT] Creating train/val/test splits...")

        if split_by_scenario:
            # Scenario-based split (safer, recommended)
            unique_scenarios = sorted(self.df["scenario_id"].unique())
            n_scenarios = len(unique_scenarios)

            n_train = int(n_scenarios * train_frac)
            n_val = int(n_scenarios * val_frac)

            train_scenarios = unique_scenarios[:n_train]
            val_scenarios = unique_scenarios[n_train : n_train + n_val]
            test_scenarios = unique_scenarios[n_train + n_val :]

            df_train = self.df[self.df["scenario_id"].isin(train_scenarios)].reset_index(drop=True)
            df_val = self.df[self.df["scenario_id"].isin(val_scenarios)].reset_index(drop=True)
            df_test = self.df[self.df["scenario_id"].isin(test_scenarios)].reset_index(drop=True)

            print(f"  Split by scenario ID:")
            print(f"    Train: scenarios {train_scenarios[0]}-{train_scenarios[-1]} "
                  f"({len(df_train):,} samples, {len(df_train)/len(self.df)*100:.1f}%)")
            print(f"    Val:   scenarios {val_scenarios[0]}-{val_scenarios[-1]} "
                  f"({len(df_val):,} samples, {len(df_val)/len(self.df)*100:.1f}%)")
            print(f"    Test:  scenarios {test_scenarios[0]}-{test_scenarios[-1]} "
                  f"({len(df_test):,} samples, {len(df_test)/len(self.df)*100:.1f}%)")

        else:
            # Temporal split within scenarios (may have some leakage)
            n_total = len(self.df)
            train_end = int(n_total * train_frac)
            val_end = train_end + int(n_total * val_frac)

            df_train = self.df.iloc[:train_end].reset_index(drop=True)
            df_val = self.df.iloc[train_end:val_end].reset_index(drop=True)
            df_test = self.df.iloc[val_end:].reset_index(drop=True)

            print(f"  Split by timestep:")
            print(f"    Train: {len(df_train):,} samples ({len(df_train)/len(self.df)*100:.1f}%)")
            print(f"    Val:   {len(df_val):,} samples ({len(df_val)/len(self.df)*100:.1f}%)")
            print(f"    Test:  {len(df_test):,} samples ({len(df_test)/len(self.df)*100:.1f}%)")

        return df_train, df_val, df_test

    def export_dataset(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
    ) -> None:
        """Save splits and metadata."""
        print("\n[EXPORT] Saving splits to CSV...")

        df_train.to_csv(self.output_dir / "train.csv", index=False)
        df_val.to_csv(self.output_dir / "val.csv", index=False)
        df_test.to_csv(self.output_dir / "test.csv", index=False)

        print(f"  ✓ train.csv: {len(df_train):,} samples")
        print(f"  ✓ val.csv:   {len(df_val):,} samples")
        print(f"  ✓ test.csv:  {len(df_test):,} samples")

        # Save metadata
        with open(self.output_dir / "dataset_info.txt", "w") as f:
            f.write("DATASET SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples: {len(self.df):,}\n")
            f.write(f"  Train: {len(df_train):,} ({len(df_train)/len(self.df)*100:.1f}%)\n")
            f.write(f"  Val:   {len(df_val):,} ({len(df_val)/len(self.df)*100:.1f}%)\n")
            f.write(f"  Test:  {len(df_test):,} ({len(df_test)/len(self.df)*100:.1f}%)\n\n")

            f.write(f"Features: {len(self.df.columns)}\n")
            f.write(f"  Columns: {', '.join(self.df.columns)}\n\n")

            f.write(f"Class balance (blackout):\n")
            f.write(f"  Train: {df_train['blackout'].sum() / len(df_train)*100:.1f}%\n")
            f.write(f"  Val:   {df_val['blackout'].sum() / len(df_val)*100:.1f}%\n")
            f.write(f"  Test:  {df_test['blackout'].sum() / len(df_test)*100:.1f}%\n")

        print(f"  ✓ dataset_info.txt")

    def run_full_validation(self) -> bool:
        """Run all validation checks and export."""
        print("\n" + "=" * 80)
        print("DATASET VALIDATION")
        print("=" * 80)

        # Run checks
        no_leakage = self.validate_no_leakage()
        class_balance = self.check_class_balance()
        diversity = self.check_diversity()
        realism = self.verify_realism()

        # Split
        df_train, df_val, df_test = self.split_train_val_test(split_by_scenario=True)

        # Export
        self.export_dataset(df_train, df_val, df_test)

        print("\n" + "=" * 80)
        print("✓ VALIDATION COMPLETE")
        print("=" * 80)

        return no_leakage and realism["status"] == "passed"


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # This would be used after dataset generation
    print("dataset_validation.py — Use in pipeline after DatasetGenerator")
