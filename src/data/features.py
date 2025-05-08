"""Feature engineering for fraud detection."""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class FraudFeatureEngineer:
    """Create and transform features for fraud detection."""

    def __init__(self):
        self.scaler = None
        self.feature_columns = []
        self.pca_features = [f"V{i}" for i in range(1, 29)]  # V1-V28

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from the Time column.

        The Time column represents seconds elapsed from first transaction.

        Args:
            df: DataFrame with 'Time' column

        Returns:
            DataFrame with added time features
        """
        df = df.copy()

        # Time is seconds from first transaction in dataset
        # Convert to hour of day (assuming 48 hours of data)
        df["hour_of_day"] = (df["Time"] / 3600) % 24

        # Time bins for different periods
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)
        df["is_morning"] = ((df["hour_of_day"] > 6) & (df["hour_of_day"] <= 12)).astype(int)
        df["is_afternoon"] = ((df["hour_of_day"] > 12) & (df["hour_of_day"] <= 18)).astype(int)
        df["is_evening"] = ((df["hour_of_day"] > 18) & (df["hour_of_day"] < 22)).astype(int)

        logger.info("Created time-based features")
        return df

    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based features.

        Args:
            df: DataFrame with 'Amount' column

        Returns:
            DataFrame with added amount features
        """
        df = df.copy()

        # Log transform amount (handle zeros)
        df["log_amount"] = np.log1p(df["Amount"])

        # Amount bins
        df["is_small_amount"] = (df["Amount"] <= 10).astype(int)
        df["is_medium_amount"] = ((df["Amount"] > 10) & (df["Amount"] <= 100)).astype(int)
        df["is_large_amount"] = ((df["Amount"] > 100) & (df["Amount"] <= 1000)).astype(int)
        df["is_very_large_amount"] = (df["Amount"] > 1000).astype(int)

        logger.info("Created amount-based features")
        return df

    def create_pca_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features from PCA components.

        V1-V28 are already PCA-transformed, but we can create
        additional features from their interactions.

        Args:
            df: DataFrame with V1-V28 columns

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        # Magnitude of PCA vector (Euclidean norm)
        pca_cols = [f"V{i}" for i in range(1, 29)]
        df["pca_magnitude"] = np.sqrt((df[pca_cols] ** 2).sum(axis=1))

        # Key component interactions (based on feature importance in fraud)
        # V1-V4 and V10-V14 are typically most important
        df["v1_v2_interaction"] = df["V1"] * df["V2"]
        df["v3_v4_interaction"] = df["V3"] * df["V4"]
        df["v14_v17_interaction"] = df["V14"] * df["V17"]

        # Ratio features
        df["v1_v3_ratio"] = df["V1"] / (df["V3"] + 1e-8)
        df["v4_v12_ratio"] = df["V4"] / (df["V12"] + 1e-8)

        logger.info("Created PCA interaction features")
        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns_to_scale: List[str] = None,
        fit: bool = True,
        scaler_type: str = "robust"
    ) -> pd.DataFrame:
        """
        Scale specified features.

        Args:
            df: DataFrame to scale
            columns_to_scale: Columns to scale (default: Amount, Time)
            fit: Whether to fit the scaler (True for training data)
            scaler_type: 'standard' or 'robust' (robust better for outliers)

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        if columns_to_scale is None:
            columns_to_scale = ["Amount", "Time"]

        # Filter to existing columns
        columns_to_scale = [c for c in columns_to_scale if c in df.columns]

        if not columns_to_scale:
            return df

        if fit:
            if scaler_type == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
            logger.info(f"Fitted and transformed {len(columns_to_scale)} columns with {scaler_type} scaler")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
            logger.info(f"Transformed {len(columns_to_scale)} columns")

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        include_engineered: bool = True,
        scale: bool = True,
        fit_scaler: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Full feature preparation pipeline.

        Args:
            df: Raw DataFrame
            include_engineered: Whether to add engineered features
            scale: Whether to scale Amount and Time
            fit_scaler: Whether to fit scaler (True for train, False for test)

        Returns:
            Tuple of (prepared DataFrame, list of feature names)
        """
        df = df.copy()

        # Create engineered features
        if include_engineered:
            df = self.create_time_features(df)
            df = self.create_amount_features(df)
            df = self.create_pca_interaction_features(df)

        # Scale Amount and Time
        if scale:
            df = self.scale_features(df, ["Amount", "Time"], fit=fit_scaler)

        # Define feature columns
        base_features = ["Time", "Amount"] + self.pca_features

        engineered_features = [
            "hour_of_day", "is_night", "is_morning", "is_afternoon", "is_evening",
            "log_amount", "is_small_amount", "is_medium_amount", "is_large_amount", "is_very_large_amount",
            "pca_magnitude", "v1_v2_interaction", "v3_v4_interaction", "v14_v17_interaction",
            "v1_v3_ratio", "v4_v12_ratio"
        ]

        if include_engineered:
            self.feature_columns = base_features + engineered_features
        else:
            self.feature_columns = base_features

        # Filter to available columns
        self.feature_columns = [c for c in self.feature_columns if c in df.columns]

        logger.info(f"Prepared {len(self.feature_columns)} features")
        return df, self.feature_columns

    def get_X_y(
        self,
        df: pd.DataFrame,
        target_col: str = "Class"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target from DataFrame.

        Args:
            df: Prepared DataFrame
            target_col: Name of target column

        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        if not self.feature_columns:
            raise ValueError("No feature columns defined. Call prepare_features first.")

        X = df[self.feature_columns]
        y = df[target_col] if target_col in df.columns else None

        if y is not None:
            logger.info(f"X shape: {X.shape}, y fraud rate: {y.mean():.4%}")
        else:
            logger.info(f"X shape: {X.shape}")

        return X, y


def create_ml_dataset(
    raw_data_path: str = "data/raw",
    output_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    End-to-end pipeline to create ML-ready dataset.

    Args:
        raw_data_path: Path to raw data directory
        output_path: Optional path to save processed data

    Returns:
        Tuple of (X features, y target, feature names)
    """
    from .ingestion import FraudDataLoader

    # Load data
    loader = FraudDataLoader(raw_data_path)
    df = loader.load_raw_data()

    # Print dataset info
    info = loader.get_data_info(df)
    logger.info(f"Dataset: {info['num_transactions']:,} transactions, "
                f"{info['fraud_percentage']:.2%} fraud rate")

    # Engineer features
    engineer = FraudFeatureEngineer()
    df_prepared, feature_names = engineer.prepare_features(df)

    # Get X and y
    X, y = engineer.get_X_y(df_prepared)

    if output_path:
        output_df = X.copy()
        output_df["Class"] = y
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved ML dataset to {output_path}")

    return X, y, feature_names
