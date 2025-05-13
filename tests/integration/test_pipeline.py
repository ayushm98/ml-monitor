"""Integration tests for ML pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDataPipeline:
    """Test data loading and feature engineering pipeline."""

    def test_data_loader(self):
        """Test fraud data loader."""
        try:
            from src.data.ingestion import FraudDataLoader
            
            loader = FraudDataLoader()
            train_df, test_df = loader.load_and_split()
            
            assert isinstance(train_df, pd.DataFrame)
            assert isinstance(test_df, pd.DataFrame)
            assert len(train_df) > 0
            assert len(test_df) > 0
            
            # Check expected columns
            assert 'Class' in train_df.columns
            assert 'Amount' in train_df.columns
        except FileNotFoundError:
            pytest.skip("Dataset not available")

    def test_feature_engineering(self):
        """Test feature engineering pipeline."""
        try:
            from src.data.features import FraudFeatureEngineer
            
            # Create sample data
            sample_data = pd.DataFrame({
                'Time': [0, 100, 200],
                'Amount': [149.62, 2.69, 378.66],
                'V1': [-1.35, -0.96, 1.45],
                'V2': [-0.07, -0.18, 0.52],
                'Class': [0, 0, 1]
            })
            
            # Add other V columns
            for i in range(3, 29):
                sample_data[f'V{i}'] = np.random.randn(3)
            
            engineer = FraudFeatureEngineer()
            df_prepared, _ = engineer.prepare_features(sample_data, include_engineered=True)
            
            # Should have engineered features
            assert 'hour_of_day' in df_prepared.columns
            assert 'log_amount' in df_prepared.columns
            assert 'pca_magnitude' in df_prepared.columns
            
        except ImportError:
            pytest.skip("Feature engineering modules not available")


class TestModelService:
    """Test model loading and prediction."""

    def test_model_service_initialization(self):
        """Test model service can be initialized."""
        try:
            from src.api.model_service import ModelService
            
            service = ModelService(mlflow_tracking_uri="http://localhost:5000")
            assert service is not None
            assert service.model_name == "fraud-detector"
        except ImportError:
            pytest.skip("Model service not available")


class TestDriftDetector:
    """Test drift detection service."""

    def test_drift_detector_initialization(self):
        """Test drift detector initialization."""
        try:
            from src.monitoring.drift_detector import DriftDetector
            
            # Create sample reference data
            reference_data = pd.DataFrame({
                'Amount': np.random.exponential(50, 1000),
                'V1': np.random.randn(1000),
                'V2': np.random.randn(1000)
            })
            
            detector = DriftDetector(
                reference_data=reference_data,
                window_size=100,
                psi_threshold=0.1
            )
            
            assert detector is not None
            assert detector.window_size == 100
            assert detector.psi_threshold == 0.1
        except ImportError:
            pytest.skip("Drift detector not available")

    def test_psi_calculation(self):
        """Test PSI calculation."""
        try:
            from src.monitoring.drift_detector import DriftDetector
            
            reference_data = pd.DataFrame({'x': np.random.randn(1000)})
            detector = DriftDetector(reference_data, window_size=100)
            
            # Same distribution should have low PSI
            same_dist = pd.Series(np.random.randn(1000))
            psi_same = detector.calculate_psi(reference_data['x'], same_dist)
            assert psi_same < 0.1
            
            # Different distribution should have higher PSI
            diff_dist = pd.Series(np.random.randn(1000) + 3)  # Shifted
            psi_diff = detector.calculate_psi(reference_data['x'], diff_dist)
            assert psi_diff > 0.2
            
        except ImportError:
            pytest.skip("Drift detector not available")
