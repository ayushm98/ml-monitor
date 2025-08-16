"""Unit tests for drift detection."""
import pytest
import pandas as pd
import numpy as np
from src.monitoring.drift_detector import DriftDetector

def test_drift_detector_init():
    """Test drift detector initialization."""
    ref_data = pd.DataFrame({'x': np.random.randn(100)})
    detector = DriftDetector(ref_data, window_size=50)
    assert detector.window_size == 50

def test_psi_calculation():
    """Test PSI calculation."""
    ref_data = pd.DataFrame({'x': np.random.randn(1000)})
    detector = DriftDetector(ref_data)
    same_dist = pd.Series(np.random.randn(1000))
    psi = detector.calculate_psi(ref_data['x'], same_dist)
    assert psi < 0.1
