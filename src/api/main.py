"""FastAPI application for fraud detection."""

import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .schemas import (
    TransactionFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo
)
from .model_service import model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions',
    ['prediction']  # fraud or legitimate
)
prediction_latency = Histogram(
    'fraud_prediction_duration_seconds',
    'Prediction latency in seconds'
)
fraud_probability_gauge = Gauge(
    'fraud_probability',
    'Last fraud probability prediction'
)
model_version_info = Gauge(
    'model_version_info',
    'Model version information',
    ['version']
)

# Track service start time
SERVICE_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting Fraud Detection API...")

    # Load model
    success = model_service.load_model(version="1")
    if not success:
        logger.warning("Failed to load model from MLflow, will retry on first request")
    else:
        model_version_info.labels(version=model_service.model_version).set(1)

    yield

    # Shutdown
    logger.info("Shutting down Fraud Detection API...")


# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with ML",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionFeatures):
    """
    Predict whether a transaction is fraudulent.

    Args:
        transaction: Transaction features

    Returns:
        Prediction response with fraud probability
    """
    start_time = time.time()

    try:
        # Ensure model is loaded
        if not model_service.is_loaded():
            success = model_service.load_model(version="1")
            if not success:
                raise HTTPException(status_code=503, detail="Model not available")

        # Convert to DataFrame
        features_dict = transaction.model_dump()
        features_df = pd.DataFrame([features_dict])

        # Make prediction
        predictions, probabilities = model_service.predict(features_df)

        is_fraud = bool(predictions[0])
        fraud_prob = float(probabilities[0])

        # Update metrics
        prediction_counter.labels(prediction='fraud' if is_fraud else 'legitimate').inc()
        fraud_probability_gauge.set(fraud_prob)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        prediction_latency.observe(time.time() - start_time)

        # Determine confidence
        confidence = model_service.get_confidence_level(fraud_prob)

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_prob, 4),
            confidence=confidence,
            model_version=f"v{model_service.model_version}",
            prediction_time_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions.

    Args:
        request: Batch prediction request

    Returns:
        Batch prediction response
    """
    start_time = time.time()

    try:
        # Ensure model is loaded
        if not model_service.is_loaded():
            success = model_service.load_model(version="1")
            if not success:
                raise HTTPException(status_code=503, detail="Model not available")

        # Convert all transactions to DataFrame
        transactions_data = [t.model_dump() for t in request.transactions]
        features_df = pd.DataFrame(transactions_data)

        # Make predictions
        predictions, probabilities = model_service.predict(features_df)

        # Create response for each transaction
        responses = []
        fraud_count = 0

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            is_fraud = bool(pred)
            fraud_prob = float(prob)

            if is_fraud:
                fraud_count += 1

            prediction_counter.labels(prediction='fraud' if is_fraud else 'legitimate').inc()

            responses.append(PredictionResponse(
                is_fraud=is_fraud,
                fraud_probability=round(fraud_prob, 4),
                confidence=model_service.get_confidence_level(fraud_prob),
                model_version=f"v{model_service.model_version}",
                prediction_time_ms=0  # Individual timing not tracked in batch
            ))

        processing_time_ms = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=responses,
            total_transactions=len(request.transactions),
            fraud_count=fraud_count,
            processing_time_ms=round(processing_time_ms, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    uptime = time.time() - SERVICE_START_TIME

    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded(),
        model_version=f"v{model_service.model_version}" if model_service.model_version else None,
        uptime_seconds=round(uptime, 2)
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        Model information
    """
    if not model_service.is_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")

    info = model_service.get_model_info()

    return ModelInfo(
        name=info["name"],
        version=f"v{info['version']}",
        stage="Production",
        metrics={},  # Could load from MLflow
        loaded_at=info["loaded_at"]
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus metrics in text format
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
