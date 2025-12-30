"""Model loading and prediction service."""

import os
import time
import logging
from datetime import datetime
from typing import Tuple, Optional
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and serving ML models."""

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        """
        Initialize model service.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_uri = mlflow_tracking_uri
        self.model = None
        self.model_version = None
        self.model_name = "fraud-detector"
        self.loaded_at = None
        self.feature_columns = None

        # Set MLflow tracking URI
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.client = MlflowClient()
            logger.info(f"Connected to MLflow at {mlflow_tracking_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow: {e}")
            self.client = None

    def load_model(self, version: Optional[str] = None, stage: str = "None") -> bool:
        """
        Load model from MLflow registry.

        Args:
            version: Specific version to load, or None for latest
            stage: Model stage (None, Staging, Production)

        Returns:
            True if model loaded successfully
        """
        try:
            if version:
                model_uri = f"models:/{self.model_name}/{version}"
            else:
                # Load latest version from specified stage
                model_uri = f"models:/{self.model_name}/{stage}"

            logger.info(f"Loading model from {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)

            # Get model version info
            if version:
                self.model_version = version
            else:
                # Get latest version for the stage
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
                if versions:
                    self.model_version = versions[0].version
                else:
                    self.model_version = "unknown"

            self.loaded_at = datetime.utcnow()
            logger.info(f"Model {self.model_name} v{self.model_version} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fall back to file-based model if MLflow not available
            try:
                return self._load_from_mlruns()
            except Exception as e2:
                logger.error(f"Failed to load from mlruns: {e2}")
                return False

    def _load_from_mlruns(self) -> bool:
        """Load model from local mlruns directory (fallback)."""
        try:
            # Load from the most recent run
            import glob
            model_paths = glob.glob("mlruns/1/*/artifacts/model")
            if not model_paths:
                return False

            # Sort by modification time, get most recent
            latest_model = max(model_paths, key=lambda p: os.path.getmtime(p))
            logger.info(f"Loading model from {latest_model}")

            self.model = mlflow.sklearn.load_model(latest_model)
            self.model_version = "latest-local"
            self.loaded_at = datetime.utcnow()
            return True
        except Exception as e:
            logger.error(f"Failed to load from mlruns: {e}")
            return False

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            features: DataFrame with raw transaction features

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Apply feature engineering
        features_engineered = self._engineer_features(features)

        # Make predictions
        predictions = self.model.predict(features_engineered)
        probabilities = self.model.predict_proba(features_engineered)[:, 1]

        return predictions, probabilities

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to raw features.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with engineered features
        """
        from src.data.features import FraudFeatureEngineer

        engineer = FraudFeatureEngineer()
        df_prepared, _ = engineer.prepare_features(df, include_engineered=True, scale=False, fit_scaler=False)
        X, _ = engineer.get_X_y(df_prepared)

        return X

    def get_confidence_level(self, probability: float) -> str:
        """
        Determine confidence level based on probability.

        Args:
            probability: Fraud probability

        Returns:
            Confidence level string
        """
        if probability < 0.3 or probability > 0.7:
            return "high"
        elif probability < 0.4 or probability > 0.6:
            return "medium"
        else:
            return "low"

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def get_model_info(self) -> dict:
        """Get information about loaded model."""
        return {
            "name": self.model_name,
            "version": self.model_version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "is_loaded": self.is_loaded()
        }


# Global model service instance
model_service = ModelService(mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
