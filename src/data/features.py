"""Feature engineering for bike demand forecasting."""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class BikeShareFeatureEngineer:
    """Create features for bike demand prediction."""

    def __init__(self):
        self.feature_columns = []

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamps.

        Args:
            df: DataFrame with 'started_at' timestamp column

        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()

        # Convert to datetime if not already
        df['started_at'] = pd.to_datetime(df['started_at'])

        # Extract temporal components
        df['hour'] = df['started_at'].dt.hour
        df['day_of_week'] = df['started_at'].dt.dayofweek
        df['day_of_month'] = df['started_at'].dt.day
        df['month'] = df['started_at'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )

        # Rush hour indicator
        df['is_rush_hour'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19])) &
                               (df['is_weekend'] == 0)).astype(int)

        logger.info("Created temporal features")
        return df

    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demand aggregation features.

        Args:
            df: DataFrame with trip data

        Returns:
            DataFrame aggregated by hour with demand features
        """
        df = df.copy()

        # Ensure started_at is datetime
        df['started_at'] = pd.to_datetime(df['started_at'])

        # Aggregate by hour
        df['datetime_hour'] = df['started_at'].dt.floor('H')

        # Count trips per hour (this is our target variable)
        hourly_demand = df.groupby('datetime_hour').size().reset_index(name='trip_count')

        # Add temporal features to aggregated data
        hourly_demand['started_at'] = hourly_demand['datetime_hour']
        hourly_demand = self.create_temporal_features(hourly_demand)

        # Calculate rolling statistics (lag features)
        hourly_demand = hourly_demand.sort_values('datetime_hour')
        hourly_demand['trip_count_lag_1h'] = hourly_demand['trip_count'].shift(1)
        hourly_demand['trip_count_lag_24h'] = hourly_demand['trip_count'].shift(24)
        hourly_demand['trip_count_rolling_3h'] = hourly_demand['trip_count'].rolling(3, min_periods=1).mean()
        hourly_demand['trip_count_rolling_24h'] = hourly_demand['trip_count'].rolling(24, min_periods=1).mean()

        logger.info(f"Created aggregated features, shape: {hourly_demand.shape}")
        return hourly_demand

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'trip_count',
        drop_na: bool = True
    ) -> tuple:
        """
        Prepare final training dataset.

        Args:
            df: DataFrame with all features
            target_col: Name of target column
            drop_na: Whether to drop rows with missing values

        Returns:
            Tuple of (features DataFrame, target Series, feature names list)
        """
        df = df.copy()

        # Define feature columns
        feature_cols = [
            'hour', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_rush_hour',
            'trip_count_lag_1h', 'trip_count_lag_24h',
            'trip_count_rolling_3h', 'trip_count_rolling_24h'
        ]

        # One-hot encode time_of_day
        if 'time_of_day' in df.columns:
            time_dummies = pd.get_dummies(df['time_of_day'], prefix='time')
            df = pd.concat([df, time_dummies], axis=1)
            feature_cols.extend(time_dummies.columns.tolist())

        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]

        X = df[available_features]
        y = df[target_col] if target_col in df.columns else None

        if drop_na:
            if y is not None:
                mask = ~(X.isna().any(axis=1) | y.isna())
                X = X[mask]
                y = y[mask]
            else:
                X = X.dropna()

        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")

        self.feature_columns = available_features
        return X, y, available_features


def create_ml_dataset(raw_data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    End-to-end pipeline to create ML-ready dataset.

    Args:
        raw_data_path: Path to raw CSV file
        output_path: Optional path to save processed data

    Returns:
        ML-ready DataFrame
    """
    from .ingestion import BikeShareDataLoader

    # Load data
    loader = BikeShareDataLoader()
    df = loader.load_raw_data()

    # Engineer features
    engineer = BikeShareFeatureEngineer()
    df_features = engineer.create_aggregated_features(df)

    # Prepare for ML
    X, y, feature_names = engineer.prepare_training_data(df_features)

    # Combine X and y for saving
    ml_data = X.copy()
    if y is not None:
        ml_data['target'] = y

    if output_path:
        ml_data.to_csv(output_path, index=False)
        logger.info(f"Saved ML dataset to {output_path}")

    return ml_data
