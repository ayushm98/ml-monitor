"""Data ingestion module for Credit Card Fraud Detection dataset."""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FraudDataLoader:
    """Load and prepare Credit Card Fraud Detection data."""

    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.target_col = "Class"
        self.feature_cols = None

    def load_raw_data(self, filename: str = "creditcard.csv") -> pd.DataFrame:
        """
        Load raw fraud detection data from CSV.

        Args:
            filename: CSV file to load

        Returns:
            DataFrame with raw transaction data
        """
        filepath = self.data_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading data from {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} transactions with {len(df.columns)} columns")

        # Store feature columns (exclude target)
        self.feature_cols = [col for col in df.columns if col != self.target_col]

        return df

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get summary information about the dataset."""
        fraud_count = df[self.target_col].sum()
        total_count = len(df)

        return {
            "num_transactions": total_count,
            "num_features": len(df.columns) - 1,
            "columns": list(df.columns),
            "fraud_count": int(fraud_count),
            "legit_count": int(total_count - fraud_count),
            "fraud_percentage": round(fraud_count / total_count * 100, 4),
            "class_imbalance_ratio": round((total_count - fraud_count) / fraud_count, 1),
            "missing_values": df.isnull().sum().sum(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "amount_stats": {
                "min": float(df["Amount"].min()),
                "max": float(df["Amount"].max()),
                "mean": float(df["Amount"].mean()),
                "median": float(df["Amount"].median())
            },
            "time_range_hours": round(df["Time"].max() / 3600, 1)
        }

    def get_class_distribution(self, df: pd.DataFrame) -> dict:
        """Get class distribution for imbalance analysis."""
        counts = df[self.target_col].value_counts()
        return {
            "legit": int(counts.get(0, 0)),
            "fraud": int(counts.get(1, 0)),
            "ratio": round(counts.get(0, 1) / counts.get(1, 1), 1)
        }

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/test split.

        Args:
            df: Full dataset
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by target class

        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        stratify_col = df[self.target_col] if stratify else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        logger.info(f"Train set: {len(train_df):,} samples")
        logger.info(f"Test set: {len(test_df):,} samples")
        logger.info(f"Train fraud rate: {train_df[self.target_col].mean():.4%}")
        logger.info(f"Test fraud rate: {test_df[self.target_col].mean():.4%}")

        return train_df, test_df
