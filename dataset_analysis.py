"""
dataset_analysis.py
===================
Dataset quality and diversity analysis.

Provides metrics to evaluate:
  - Feature diversity and coverage
  - Class balance
  - Scenario type distribution
  - Overfitting risk detection
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DatasetAnalyzer:
    """Analyze dataset quality and diversity."""

    def __init__(self, df: pd.DataFrame, output_dir: Path = Path("datasets/generated")):
        """
        Initialize analyzer.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze
        output_dir : Path
            Output directory for reports
        """
        self.df = df
        self.output_dir = Path(output_dir)

    def compute_feature_statistics(self) -> pd.DataFrame:
        """Compute statistics for all numeric features."""
        print("\n[ANALYSIS] Computing feature statistics...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        stats = pd.DataFrame(
            {
                "mean": self.df[numeric_cols].mean(),
                "std": self.df[numeric_cols].std(),
                "min": self.df[numeric_cols].min(),
                "max": self.df[numeric_cols].max(),
                "q25": self.df[numeric_cols].quantile(0.25),
                "q50": self.df[numeric_cols].quantile(0.50),
                "q75": self.df[numeric_cols].quantile(0.75),
            }
        )

        # Save
        stats.to_csv(self.output_dir / "feature_statistics.csv")
        print(f"  ✓ Feature statistics saved")

        return stats

    def compute_diversity_score(self) -> float:
        """
        Compute overall diversity metric (0-1).

        Based on:
          - Coverage of feature space
          - Scenario type balance
          - Blackout label distribution
        """
        print("\n[ANALYSIS] Computing diversity score...")

        scores = []

        # 1. Voltage coverage: how much of the operating range is covered?
        v_min_range = self.df["v_min"].max() - self.df["v_min"].min()
        v_coverage = min(v_min_range / 0.3, 1.0)  # Target: 0.3 pu range
        scores.append(v_coverage)
        print(f"  Voltage coverage: {v_coverage:.3f} (range: {v_min_range:.3f} pu)")

        # 2. Load coverage
        load_range = self.df["p_load_mw"].max() - self.df["p_load_mw"].min()
        load_coverage = min(load_range / 4.0, 1.0)  # Target: 4.0 MW range
        scores.append(load_coverage)
        print(f"  Load coverage: {load_coverage:.3f} (range: {load_range:.2f} MW)")

        # 3. Solar coverage
        solar_range = self.df["p_solar_mw"].max() - self.df["p_solar_mw"].min()
        solar_coverage = min(solar_range / 2.5, 1.0)  # Target: 2.5 MW range
        scores.append(solar_coverage)
        print(f"  Solar coverage: {solar_coverage:.3f} (range: {solar_range:.2f} MW)")

        # 4. Scenario type balance (entropy-based)
        if "scenario_type" in self.df.columns:
            type_dist = self.df["scenario_type"].value_counts(normalize=True)
            # Calculate entropy (normalized)
            entropy = -np.sum(type_dist * np.log2(type_dist + 1e-10))
            max_entropy = np.log2(len(type_dist))
            balance_score = entropy / max_entropy if max_entropy > 0 else 1.0
            scores.append(balance_score)
            print(f"  Scenario balance: {balance_score:.3f} (entropy: {entropy:.3f})")

        # 5. Class balance (not too skewed)
        blackout_rate = self.df["blackout"].mean()
        class_balance = 1.0 - abs(blackout_rate - 0.2)  # Target: ~20% blackout
        class_balance = max(0, class_balance)
        scores.append(class_balance)
        print(f"  Class balance: {class_balance:.3f} (blackout rate: {blackout_rate:.1%})")

        overall_score = np.mean(scores)
        print(f"\n  Overall diversity score: {overall_score:.3f}")

        return overall_score

    def detect_outliers(self, threshold: float = 3.0) -> dict:
        """Identify samples with extreme feature values."""
        print(f"\n[ANALYSIS] Detecting outliers (threshold: {threshold}σ)...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs((self.df[numeric_cols] - self.df[numeric_cols].mean()) / (self.df[numeric_cols].std() + 1e-10))

        outlier_samples = (z_scores > threshold).any(axis=1)
        n_outliers = outlier_samples.sum()

        print(f"  Found {n_outliers} outlier samples ({n_outliers/len(self.df)*100:.2f}%)")

        return {"count": n_outliers, "percentage": n_outliers / len(self.df) * 100}

    def suggest_overfitting_detection(self) -> dict:
        """Provide recommendations to detect ML overfitting to simulation artifacts."""
        print("\n[ANALYSIS] Overfitting detection recommendations...")

        recommendations = {
            "feature_importance": {
                "description": "Top 3 features should not exceed 50% importance",
                "why": "High importance concentration suggests memorization",
                "test": "Train RF model, check feature_importance_[:3].sum()",
                "threshold": 0.50,
            },
            "transfer_learning": {
                "description": "Train on 50% of scenarios, test on other 50%",
                "why": "Large train/test gap indicates scenario-specific memorization",
                "test": "AUC drop > 10% suggests overfitting",
                "threshold": 0.10,
            },
            "noise_sensitivity": {
                "description": "Increase measurement noise σ and retrain",
                "why": "Overfitted models degrade more with noise",
                "test": "Model performance with σ=0.02 vs σ=0.05",
                "threshold": 0.05,
            },
            "temporal_generalization": {
                "description": "Generate hold-out test with different season/pattern",
                "why": "Overfitting to seasonal patterns in training data",
                "test": "AUC on unseen temporal pattern",
                "threshold": 0.15,
            },
            "ablation_test": {
                "description": "Remove synthetic stress from test set",
                "why": "Model may memorize stress injection patterns",
                "test": "Compare AUC on normal vs synthetic-heavy test sets",
                "threshold": 0.08,
            },
        }

        for test_name, details in recommendations.items():
            print(f"\n  {test_name}:")
            print(f"    Description: {details['description']}")
            print(f"    Why: {details['why']}")
            print(f"    How: {details['test']}")

        return recommendations

    def generate_report(self) -> str:
        """Generate HTML report."""
        print("\n[ANALYSIS] Generating report...")

        stats = self.compute_feature_statistics()
        diversity_score = self.compute_diversity_score()
        outliers = self.detect_outliers()

        html = f"""
        <html>
        <head>
            <title>Dataset Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Dataset Analysis Report</h1>

            <h2>Summary</h2>
            <div class="summary">
                <p><strong>Total Samples:</strong> {len(self.df):,}</p>
                <p><strong>Features:</strong> {len(self.df.columns)}</p>
                <p><strong>Scenario Types:</strong> {self.df['scenario_type'].nunique() if 'scenario_type' in self.df.columns else 'N/A'}</p>
                <p><strong>Diversity Score:</strong> {diversity_score:.3f} / 1.0</p>
                <p><strong>Outliers:</strong> {outliers['count']} ({outliers['percentage']:.2f}%)</p>
            </div>

            <h2>Class Distribution</h2>
            <p>Blackout rate: {self.df['blackout'].mean():.1%}</p>
            <p>Normal rate: {(1 - self.df['blackout'].mean()):.1%}</p>

            <h2>Feature Statistics</h2>
            <table>
                <tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
        """

        for feature in stats.index[:10]:  # Show first 10 features
            html += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{stats.loc[feature, 'mean']:.4f}</td>
                    <td>{stats.loc[feature, 'std']:.4f}</td>
                    <td>{stats.loc[feature, 'min']:.4f}</td>
                    <td>{stats.loc[feature, 'max']:.4f}</td>
                </tr>
            """

        html += """
            </table>

            <h2>Recommendations</h2>
            <ul>
                <li>Use scenario-based train/val/test split to avoid temporal leakage</li>
                <li>Monitor feature importance for signs of overfitting</li>
                <li>Test transfer learning across scenario types</li>
                <li>Validate noise robustness with different measurement noise levels</li>
            </ul>
        </body>
        </html>
        """

        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, "w") as f:
            f.write(html)

        print(f"  ✓ Report saved to {report_path}")

        return html


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("dataset_analysis.py — Use in pipeline after DatasetValidator")
