"""FastAPI application exposing the sentiment analysis model.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health          - service + model status
    POST /predict         - classify a single text
    POST /predict/batch   - classify a list of texts
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import get_load_error, get_model_type, get_predictor, load_predictor
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading sentiment model...")
    load_predictor()
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Sentiment Analysis API",
    description="Classify text as positive, negative, or neutral user sentiment",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS is permissive by default for local/demo use; restrict allow_origins in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = get_predictor()
    status = "ok" if predictor is not None else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=predictor is not None,
        model_type=get_model_type(),
        load_error=get_load_error(),
        error=get_load_error(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train and save a model, then restart API.")

    prediction = predictor.predict(request.text)[0]
    return PredictResponse(model_used=get_model_type(), prediction=prediction)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train and save a model, then restart API.")
        
    predictions = predictor.predict(request.texts)
    return BatchPredictResponse(model_used=get_model_type(), predictions=predictions)