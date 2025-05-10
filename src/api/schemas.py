"""Pydantic schemas for API request/response validation."""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class TransactionFeatures(BaseModel):
    """Input features for fraud prediction."""

    Time: float = Field(..., description="Seconds elapsed from first transaction in dataset")
    Amount: float = Field(..., ge=0, description="Transaction amount in dollars")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")

    @field_validator('Amount')
    @classmethod
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response from fraud prediction."""

    is_fraud: bool = Field(..., description="Whether transaction is predicted as fraudulent")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    confidence: str = Field(..., description="Confidence level: low, medium, high")
    model_version: str = Field(..., description="Version of model used")
    prediction_time_ms: float = Field(..., description="Time taken for prediction in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.05,
                "confidence": "high",
                "model_version": "v1",
                "prediction_time_ms": 12.5
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    transactions: List[TransactionFeatures] = Field(..., max_length=100, description="List of transactions to predict")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Current model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    version: str
    stage: str
    metrics: dict
    loaded_at: str
