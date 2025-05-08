"""Data validation for Credit Card Fraud Detection dataset."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class FraudDataValidator:
    """Validate fraud detection data quality."""

    def __init__(self):
        self.validation_results = {}
        self.expected_columns = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame has expected columns.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with validation results
        """
        missing_cols = set(self.expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(self.expected_columns)

        is_valid = len(missing_cols) == 0

        result = {
            "is_valid": is_valid,
            "missing_columns": list(missing_cols),
            "extra_columns": list(extra_cols),
            "expected_count": len(self.expected_columns),
            "actual_count": len(df.columns)
        }

        if not is_valid:
            logger.warning(f"Schema validation failed: missing {missing_cols}")
        else:
            logger.info("Schema validation passed")

        return result

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with quality metrics
        """
        total_rows = len(df)

        # Check for missing values
        missing_counts = df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0].to_dict()

        # Check for duplicates
        duplicate_count = df.duplicated().sum()

        quality_metrics = {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "missing_values_total": int(missing_counts.sum()),
            "columns_with_missing": columns_with_missing,
            "duplicate_rows": int(duplicate_count),
            "duplicate_percentage": round(duplicate_count / total_rows * 100, 4),
            "is_valid": missing_counts.sum() == 0
        }

        logger.info(f"Data quality: {total_rows:,} rows, "
                   f"{quality_metrics['missing_values_total']} missing values, "
                   f"{duplicate_count} duplicates")

        return quality_metrics

    def validate_target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate target class distribution.

        Args:
            df: DataFrame with 'Class' column

        Returns:
            Dict with class distribution info
        """
        if "Class" not in df.columns:
            return {"is_valid": False, "error": "Class column not found"}

        class_counts = df["Class"].value_counts()
        fraud_count = int(class_counts.get(1, 0))
        legit_count = int(class_counts.get(0, 0))
        total = fraud_count + legit_count

        # Validate expected imbalance (fraud should be < 1% typically)
        fraud_rate = fraud_count / total if total > 0 else 0
        expected_range = (0.001, 0.01)  # 0.1% to 1%
        is_valid = expected_range[0] <= fraud_rate <= expected_range[1]

        result = {
            "is_valid": is_valid,
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_rate": round(fraud_rate, 6),
            "fraud_percentage": round(fraud_rate * 100, 4),
            "imbalance_ratio": round(legit_count / fraud_count, 1) if fraud_count > 0 else None,
            "expected_fraud_rate_range": expected_range,
            "warning": None if is_valid else f"Fraud rate {fraud_rate:.4%} outside expected range"
        }

        if is_valid:
            logger.info(f"Target distribution valid: {fraud_rate:.4%} fraud rate")
        else:
            logger.warning(f"Target distribution unusual: {fraud_rate:.4%} fraud rate")

        return result

    def validate_numeric_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate numeric columns are within expected ranges.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with validation results
        """
        results = {}

        # Amount should be non-negative
        if "Amount" in df.columns:
            negative_amounts = (df["Amount"] < 0).sum()
            results["Amount"] = {
                "is_valid": negative_amounts == 0,
                "min": float(df["Amount"].min()),
                "max": float(df["Amount"].max()),
                "mean": float(df["Amount"].mean()),
                "negative_count": int(negative_amounts)
            }

        # Time should be non-negative
        if "Time" in df.columns:
            negative_times = (df["Time"] < 0).sum()
            results["Time"] = {
                "is_valid": negative_times == 0,
                "min": float(df["Time"].min()),
                "max": float(df["Time"].max()),
                "span_hours": round(df["Time"].max() / 3600, 1),
                "negative_count": int(negative_times)
            }

        # V1-V28 should have reasonable distributions (PCA components)
        pca_stats = {}
        for i in range(1, 29):
            col = f"V{i}"
            if col in df.columns:
                pca_stats[col] = {
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4)
                }

        results["pca_features"] = {
            "count": len(pca_stats),
            "stats": pca_stats
        }

        # Overall validation
        all_valid = all(
            results.get(key, {}).get("is_valid", True)
            for key in ["Amount", "Time"]
        )
        results["overall_valid"] = all_valid

        logger.info(f"Numeric range validation: {'PASSED' if all_valid else 'FAILED'}")
        return results

    def validate_for_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Check for data drift between reference and current data.

        Uses Population Stability Index (PSI) for drift detection.

        Args:
            reference_df: Reference/training data
            current_df: Current/production data
            threshold: PSI threshold for drift (0.1 = some drift, 0.2 = significant)

        Returns:
            Dict with drift metrics
        """
        drift_results = {}

        # Check Amount distribution drift
        if "Amount" in reference_df.columns and "Amount" in current_df.columns:
            psi = self._calculate_psi(reference_df["Amount"], current_df["Amount"])
            drift_results["Amount"] = {
                "psi": round(psi, 4),
                "has_drift": psi > threshold,
                "severity": "high" if psi > 0.2 else "medium" if psi > 0.1 else "low"
            }

        # Check PCA feature drift (sample a few key features)
        for col in ["V1", "V4", "V14", "V17"]:
            if col in reference_df.columns and col in current_df.columns:
                psi = self._calculate_psi(reference_df[col], current_df[col])
                drift_results[col] = {
                    "psi": round(psi, 4),
                    "has_drift": psi > threshold,
                    "severity": "high" if psi > 0.2 else "medium" if psi > 0.1 else "low"
                }

        # Overall drift assessment
        any_drift = any(r.get("has_drift", False) for r in drift_results.values())
        drift_results["overall"] = {
            "has_drift": any_drift,
            "features_with_drift": [k for k, v in drift_results.items() if v.get("has_drift")]
        }

        logger.info(f"Drift validation: {'DRIFT DETECTED' if any_drift else 'No drift'}")
        return drift_results

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)

        # Calculate proportions in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        ref_props = (ref_counts + epsilon) / (len(reference) + epsilon * bins)
        curr_props = (curr_counts + epsilon) / (len(current) + epsilon * bins)

        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))

        return psi

    def run_all_validations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with all validation results
        """
        results = {
            "schema": self.validate_schema(df),
            "data_quality": self.validate_data_quality(df),
            "target_distribution": self.validate_target_distribution(df),
            "numeric_ranges": self.validate_numeric_ranges(df)
        }

        # Overall validation status
        all_valid = (
            results["schema"]["is_valid"] and
            results["data_quality"]["is_valid"] and
            results["numeric_ranges"]["overall_valid"]
        )

        results["overall_valid"] = all_valid
        results["summary"] = {
            "total_rows": results["data_quality"]["total_rows"],
            "fraud_rate": results["target_distribution"].get("fraud_percentage"),
            "all_checks_passed": all_valid
        }

        self.validation_results = results

        logger.info(f"Overall validation: {'PASSED' if all_valid else 'FAILED'}")
        return results
