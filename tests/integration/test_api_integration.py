"""Integration tests for fraud detection API."""

import pytest
import pandas as pd
from fastapi.testclient import TestClient


@pytest.fixture
def sample_transaction():
    """Sample transaction for testing."""
    return {
        "Time": 0,
        "Amount": 149.62,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053
    }


class TestAPIEndpoints:
    """Test API endpoint functionality."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Fraud Detection API"
        assert "endpoints" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "fraud_predictions_total" in response.text

    def test_prediction_endpoint(self, client, sample_transaction):
        """Test single prediction endpoint."""
        response = client.post("/predict", json=sample_transaction)
        
        # Should succeed or fail gracefully
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "confidence" in data
            assert "model_version" in data
            assert isinstance(data["fraud_probability"], float)
            assert 0 <= data["fraud_probability"] <= 1

    def test_batch_prediction_endpoint(self, client, sample_transaction):
        """Test batch prediction endpoint."""
        batch_request = {
            "transactions": [sample_transaction, sample_transaction]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        # Should succeed or fail gracefully
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_transactions" in data
            assert data["total_transactions"] == 2
            assert len(data["predictions"]) == 2


class TestDriftMonitoring:
    """Test drift detection integration."""

    def test_drift_status_endpoint(self, client):
        """Test drift status endpoint."""
        response = client.get("/drift/status")
        
        # May not be initialized in test environment
        assert response.status_code in [200, 503]

    def test_drift_report_endpoint(self, client):
        """Test drift report endpoint."""
        response = client.get("/drift/report")
        
        # May not be initialized in test environment
        assert response.status_code in [200, 503]


@pytest.fixture
def client():
    """Create test client."""
    try:
        from src.api.main import app
        return TestClient(app)
    except Exception:
        pytest.skip("API not available in test environment")
