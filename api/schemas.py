"""Pydantic request/response schemas for the sentiment analysis API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Raw text to classify.")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ..., min_length=1, max_length=64, description="List of raw texts to classify."
    )


class PredictionResult(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]


class PredictResponse(BaseModel):
    model_used: str
    prediction: PredictionResult


class BatchPredictResponse(BaseModel):
    model_used: str
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    load_error: Optional[str] = None
    error: Optional[str] = None