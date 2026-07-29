"""Factory functions to construct models by name and persist/load checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoTokenizer

from src.models.bert_classifier import BertSentimentClassifier
from src.models.bert_lstm_classifier import BertLstmClassifier
from src.config import config as project_config


ModelName = Literal["bert", "bert_lstm"]
MODEL_FILENAME = "pytorch_model.bin"
MODEL_CONFIG_FILENAME = "model_config.json"


def build_model(model_name: ModelName, cfg=None):
    """Instantiate a model architecture by name using project config values."""
    if cfg is None:
        cfg = project_config

    if model_name == "bert":
        return BertSentimentClassifier(
            bert_model_name=cfg.model.bert_model_name,
            num_labels=cfg.model.num_labels,
            dropout=cfg.model.dropout,
            freeze_bert=getattr(cfg.model, "freeze_bert", False),
            use_lora=getattr(cfg.model, "use_lora", False),
            lora_r=getattr(cfg.model, "lora_r", 8),
            lora_alpha=getattr(cfg.model, "lora_alpha", 32),
            lora_dropout=getattr(cfg.model, "lora_dropout", 0.1),
        )
    elif model_name == "bert_lstm":
        return BertLstmClassifier(
            bert_model_name=cfg.model.bert_model_name,
            num_labels=cfg.model.num_labels,
            lstm_hidden_size=cfg.model.lstm_hidden_size,
            lstm_num_layers=cfg.model.lstm_num_layers,
            lstm_bidirectional=cfg.model.lstm_bidirectional,
            dropout=cfg.model.dropout,
            # For the hybrid BERT+LSTM model we default to using frozen BERT
            # embeddings and training only the BiLSTM/classifier. To override,
            # set `model.freeze_bert: false` in your config.
            freeze_bert=getattr(cfg.model, "freeze_bert", True),
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Expected 'bert' or 'bert_lstm'.")


def save_model(model: torch.nn.Module, model_name: str, cfg, output_dir: str | Path) -> None:
    """Save model weights, architecture config, and tokenizer to `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / MODEL_FILENAME)

    model_config = {
        "model_name": model_name,
        "bert_model_name": cfg.model.bert_model_name,
        "num_labels": cfg.model.num_labels,
        "dropout": cfg.model.dropout,
        "max_seq_length": cfg.model.max_seq_length,
        "freeze_bert": getattr(cfg.model, "freeze_bert", False),
        "use_lora": getattr(cfg.model, "use_lora", False),
        "lora_r": getattr(cfg.model, "lora_r", None),
        "lora_alpha": getattr(cfg.model, "lora_alpha", None),
        "lora_dropout": getattr(cfg.model, "lora_dropout", None),
    }

    if model_name == "bert_lstm":
        model_config.update({
            "lstm_hidden_size": cfg.model.lstm_hidden_size,
            "lstm_num_layers": cfg.model.lstm_num_layers,
            "lstm_bidirectional": cfg.model.lstm_bidirectional,
        })

    with open(output_dir / MODEL_CONFIG_FILENAME, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model_name)
    tokenizer.save_pretrained(output_dir)


def load_model(model_dir: str | Path, device: str | torch.device = "cpu"):
    """
    Load a previously saved model from `model_dir`.
    Returns (model, tokenizer, config_dict)
    """
    model_dir = Path(model_dir)

    with open(model_dir / MODEL_CONFIG_FILENAME, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model_name = model_config["model_name"]

    if model_name == "bert":
        model = BertSentimentClassifier(
            bert_model_name=model_config["bert_model_name"],
            num_labels=model_config["num_labels"],
            dropout=model_config.get("dropout", 0.3),
            freeze_bert=model_config.get("freeze_bert", False),
            use_lora=model_config.get("use_lora", False),
            lora_r=model_config.get("lora_r", 8),
            lora_alpha=model_config.get("lora_alpha", 32),
            lora_dropout=model_config.get("lora_dropout", 0.1),
        )
    elif model_name == "bert_lstm":
        model = BertLstmClassifier(
            bert_model_name=model_config["bert_model_name"],
            num_labels=model_config["num_labels"],
            lstm_hidden_size=model_config["lstm_hidden_size"],
            lstm_num_layers=model_config["lstm_num_layers"],
            lstm_bidirectional=model_config["lstm_bidirectional"],
            dropout=model_config.get("dropout", 0.3),
            freeze_bert=model_config.get("freeze_bert", False),
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}' in saved config.")

    state_dict = torch.load(model_dir / MODEL_FILENAME, map_location=device)
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer, model_config