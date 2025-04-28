"""Data validation for bikeshare dataset."""

import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate bikeshare data quality."""

    def __init__(self):
        self.validation_results = {}

    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
        """
        Validate DataFrame has expected columns.

        Args:
            df: DataFrame to validate
            expected_columns: List of required column names

        Returns:
            Dict with validation results
        """
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)

        is_valid = len(missing_cols) == 0

        result = {
            "is_valid": is_valid,
            "missing_columns": list(missing_cols),
            "extra_columns": list(extra_cols),
            "actual_columns": list(df.columns)
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

        quality_metrics = {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "missing_values": {
                col: {
                    "count": int(df[col].isnull().sum()),
                    "percentage": round(df[col].isnull().sum() / total_rows * 100, 2)
                }
                for col in df.columns
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": round(df.duplicated().sum() / total_rows * 100, 2)
        }

        logger.info(f"Data quality check: {total_rows} rows, "
                   f"{quality_metrics['duplicate_percentage']}% duplicates")

        return quality_metrics

    def validate_numeric_ranges(
        self,
        df: pd.DataFrame,
        column_ranges: Dict[str, tuple]
    ) -> Dict[str, Any]:
        """
        Validate numeric columns are within expected ranges.

        Args:
            df: DataFrame to validate
            column_ranges: Dict mapping column names to (min, max) tuples

        Returns:
            Dict with validation results per column
        """
        results = {}

        for col, (min_val, max_val) in column_ranges.items():
            if col not in df.columns:
                results[col] = {"status": "column_not_found"}
                continue

            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            total = len(df[col].dropna())

            results[col] = {
                "is_valid": out_of_range == 0,
                "out_of_range_count": int(out_of_range),
                "out_of_range_percentage": round(out_of_range / total * 100, 2) if total > 0 else 0,
                "min_expected": min_val,
                "max_expected": max_val,
                "min_actual": float(df[col].min()),
                "max_actual": float(df[col].max())
            }

            if not results[col]["is_valid"]:
                logger.warning(f"{col}: {out_of_range} values out of range [{min_val}, {max_val}]")

        return results

    def validate_timestamps(self, df: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """
        Validate timestamp column.

        Args:
            df: DataFrame to validate
            timestamp_col: Name of timestamp column

        Returns:
            Dict with validation results
        """
        if timestamp_col not in df.columns:
            return {"is_valid": False, "error": f"Column {timestamp_col} not found"}

        try:
            timestamps = pd.to_datetime(df[timestamp_col])

            result = {
                "is_valid": True,
                "date_range": {
                    "start": str(timestamps.min()),
                    "end": str(timestamps.max())
                },
                "null_count": int(df[timestamp_col].isnull().sum()),
                "invalid_dates": 0
            }

            logger.info(f"Timestamp validation passed for {timestamp_col}")
            return result

        except Exception as e:
            logger.error(f"Timestamp validation failed: {e}")
            return {"is_valid": False, "error": str(e)}

    def run_all_validations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with all validation results
        """
        # Expected columns for Capital Bikeshare data
        expected_cols = [
            'started_at', 'ended_at', 'start_station_name', 'end_station_name',
            'start_lat', 'start_lng', 'end_lat', 'end_lng',
            'member_casual'
        ]

        results = {
            "schema": self.validate_schema(df, expected_cols),
            "data_quality": self.validate_data_quality(df),
            "timestamp_validation": self.validate_timestamps(df, 'started_at'),
            "numeric_ranges": self.validate_numeric_ranges(df, {
                'start_lat': (38.0, 40.0),  # DC area latitude
                'start_lng': (-78.0, -76.0),  # DC area longitude
                'end_lat': (38.0, 40.0),
                'end_lng': (-78.0, -76.0)
            })
        }

        # Overall validation status
        all_valid = (
            results["schema"]["is_valid"] and
            results["timestamp_validation"]["is_valid"] and
            all(v["is_valid"] for v in results["numeric_ranges"].values())
        )

        results["overall_valid"] = all_valid
        self.validation_results = results

        logger.info(f"Overall validation: {'PASSED' if all_valid else 'FAILED'}")
        return results
