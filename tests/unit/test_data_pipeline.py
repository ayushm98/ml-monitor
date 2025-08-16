"""Unit tests for data pipeline."""
import pytest
import pandas as pd
from src.data.features import FraudFeatureEngineer

def test_feature_engineer_initialization():
    """Test feature engineer can be initialized."""
    engineer = FraudFeatureEngineer()
    assert engineer is not None

def test_feature_engineering():
    """Test feature engineering creates expected features."""
    engineer = FraudFeatureEngineer()
    sample_df = pd.DataFrame({
        'Time': [0, 100],
        'Amount': [150.0, 2.5],
        'V1': [-1.0, 0.5],
        'V2': [0.2, -0.3],
        'Class': [0, 1]
    })
    for i in range(3, 29):
        sample_df[f'V{i}'] = [0.0, 0.1]
    
    df_prep, _ = engineer.prepare_features(sample_df, include_engineered=True)
    assert 'hour_of_day' in df_prep.columns
    assert 'log_amount' in df_prep.columns
