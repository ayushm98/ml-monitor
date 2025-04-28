"""Data ingestion module for Capital Bikeshare dataset."""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BikeShareDataLoader:
    """Load and prepare Capital Bikeshare trip data."""

    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)

    def load_raw_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw bikeshare data from CSV.

        Args:
            filename: Specific CSV file to load. If None, loads first CSV found.

        Returns:
            DataFrame with raw trip data
        """
        if filename:
            filepath = self.data_path / filename
        else:
            # Find first CSV file
            csv_files = list(self.data_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_path}")
            filepath = csv_files[0]

        logger.info(f"Loading data from {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        return df

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get summary information about the dataset."""
        return {
            "num_records": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "date_range": {
                "start": df['started_at'].min() if 'started_at' in df.columns else None,
                "end": df['started_at'].max() if 'started_at' in df.columns else None
            },
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
