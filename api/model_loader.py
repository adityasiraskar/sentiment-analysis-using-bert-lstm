"""
Singleton wrapper around `class SentimentPredictor` for the API.

The model type and location are configurable via environment variables
so the same Docker image can serve either the 'bert' or 'bert_lstm' model.

MODEL_TYPE="bert" or "bert_lstm"
MODEL_DIR=/path/to/model/dir  # optional explicit override
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.config import config as project_config
from src.config import resolve_path
from src.inference.predict import SentimentPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

_predictor: Optional[SentimentPredictor] = None
_load_error: Optional[str] = None
_model_type: str = os.environ.get("MODEL_TYPE", "bert")


def _default_model_dir() -> str:
    return str(resolve_path(project_config.training.output_dir) / _model_type)


def load_predictor() -> None:
    """Load the configured model into memory. Safe to call once at startup."""
    global _predictor, _load_error

    model_dir = os.environ.get("MODEL_DIR", _default_model_dir())
    logger.info(f"Loading model '{_model_type}' from {model_dir}")

    model_path = Path(model_dir)
    if not model_path.exists():
        _predictor = None
        _load_error = f"Model directory not found: {model_dir}"
        logger.warning(_load_error)
        return

    if not (model_path / "pytorch_model.bin").exists():
        _predictor = None
        _load_error = f"Model weights not found in {model_dir}"
        logger.warning(_load_error)
        return

    try:
        _predictor = SentimentPredictor(model_dir)
        _load_error = None
        logger.info(f"Successfully loaded model from {model_dir}")
    except Exception as e:
        _predictor = None
        _load_error = str(e)
        logger.error(f"Failed to load model from {model_dir}: {e}")


def get_predictor() -> Optional[SentimentPredictor]:
    return _predictor


def get_model_type() -> str:
    return _model_type


def get_load_error() -> Optional[str]:
    return _load_error