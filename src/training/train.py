"""
Fine-tune training script for the BERT and BERT+LSTM sentiment classifiers.

Usage:
    python -m src.training.train --model bert
    python -m src.training.train --model bert_lstm
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import load_config, resolve_path
from src.data.dataset import build_datasets_from_config
from src.models.factory import build_model, save_model
from src.training.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import accuracy_score, f1_score
    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    return metrics


def train_one_model(model_name: str, cfg, device, torch_device):
    logger.info(f"--- Training model: {model_name} ---")

    set_seed(cfg.training.seed)

    # Apply per-model overrides from config.training.per_model if present.
    import copy

    per_model_cfg = {}
    try:
        per_model_cfg = dict(cfg.training.get("per_model", {}))
    except Exception:
        # AttrDict may not support get in older layouts; fall back to attribute access
        per_model_cfg = getattr(cfg.training, "per_model", {}) or {}

    model_overrides = per_model_cfg.get(model_name, {}) if isinstance(per_model_cfg, dict) else {}

    # Create a deepcopy of cfg to safely apply per-model overrides without mutating global config
    local_cfg = copy.deepcopy(cfg)

    # Training-level overrides
    if isinstance(model_overrides, dict):
        if "batch_size" in model_overrides:
            local_cfg.training.batch_size = model_overrides["batch_size"]
        if "eval_batch_size" in model_overrides:
            local_cfg.training.eval_batch_size = model_overrides["eval_batch_size"]
        if "num_epochs" in model_overrides:
            local_cfg.training.num_epochs = model_overrides["num_epochs"]
        if "learning_rate" in model_overrides:
            local_cfg.training.learning_rate = model_overrides["learning_rate"]
        if "gradient_accumulation_steps" in model_overrides:
            local_cfg.training.gradient_accumulation_steps = model_overrides["gradient_accumulation_steps"]
        if "use_amp" in model_overrides:
            local_cfg.training.use_amp = model_overrides["use_amp"]

        # Model-level overrides
        if "freeze_bert" in model_overrides:
            local_cfg.model.freeze_bert = model_overrides["freeze_bert"]
        if "use_lora" in model_overrides:
            local_cfg.model.use_lora = model_overrides["use_lora"]
        if "lora_r" in model_overrides:
            local_cfg.model.lora_r = model_overrides["lora_r"]
        if "lora_alpha" in model_overrides:
            local_cfg.model.lora_alpha = model_overrides["lora_alpha"]
        if "lora_dropout" in model_overrides:
            local_cfg.model.lora_dropout = model_overrides["lora_dropout"]

    tokenizer = AutoTokenizer.from_pretrained(local_cfg.model.bert_model_name)
    train_ds, val_ds, test_ds = build_datasets_from_config(local_cfg, tokenizer)

    logger.info(
        f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=local_cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=local_cfg.training.eval_batch_size)

    # Build model using the per-model-local config so overrides take effect
    model = build_model(model_name, local_cfg)
    model.to(torch_device)

    # Only parameters with requires_grad=True will be passed to the optimizer,
    # which allows LoRA adapters (or frozen base models) to work out of the box.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=local_cfg.training.learning_rate,
        weight_decay=getattr(local_cfg.training, "weight_decay", 0.0),
    )

    num_training_steps = len(train_loader) * local_cfg.training.num_epochs
    num_warmup_steps = int(num_training_steps * getattr(local_cfg.training, "warmup_ratio", 0.1))

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    best_f1 = -1.0
    output_dir = resolve_path(local_cfg.training.output_dir) / model_name
    # Optional gradient accumulation and mixed precision (use per-model overrides)
    accumulate_steps = int(getattr(local_cfg.training, "gradient_accumulation_steps", 1))
    use_amp = bool(getattr(local_cfg.training, "use_amp", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(local_cfg.training.num_epochs):
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(torch_device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(torch_device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs, labels=labels)
                    loss = outputs["loss"]
                loss = loss / accumulate_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(**inputs, labels=labels)
                loss = outputs["loss"]
                loss = loss / accumulate_steps
                loss.backward()

            if (step + 1) % accumulate_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, local_cfg.training.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, local_cfg.training.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulate_steps

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_loader(model, val_loader, torch_device)

        logger.info(
            f"Epoch {epoch+1}/{local_cfg.training.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f}"
        )
        
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            logger.info(f"  -> Saved new best checkpoint to {output_dir}")
            # Save using the local config that reflects overrides
            save_model(model, model_name, local_cfg, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentiment classification models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "bert_lstm", "both"],
        default="both",
        help="Which model architecture to train.",
    )
    # Runtime overrides (override config values)
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument("--eval-batch-size", type=int, help="Override training.eval_batch_size")
    parser.add_argument("--num-epochs", type=int, help="Override training.num_epochs")
    parser.add_argument("--learning-rate", type=float, help="Override training.learning_rate")
    parser.add_argument("--freeze-bert", action="store_true", help="Freeze BERT encoder for this run")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapters for this run (bert only)")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Number of steps to accumulate gradients")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision if available")
    args = parser.parse_args()

    cfg = load_config()

    # Apply CLI overrides into config
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.eval_batch_size:
        cfg.training.eval_batch_size = args.eval_batch_size
    if args.num_epochs:
        cfg.training.num_epochs = args.num_epochs
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
    if args.freeze_bert:
        cfg.model.freeze_bert = True
    if args.use_lora:
        cfg.model.use_lora = True
    if args.gradient_accumulation_steps:
        cfg.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.use_amp:
        cfg.training.use_amp = True

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device_str)
    logger.info(f"Using device: {device_str}")
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

    models_to_train = ["bert", "bert_lstm"] if args.model == "both" else [args.model]

    for m_name in models_to_train:
        train_one_model(m_name, cfg, device_str, torch_device)

    logger.info("Training complete. Summary:")
    for m_name in models_to_train:
        logger.info(f"Evaluating best {m_name} on test set...")
        evaluate_model(m_name, cfg, device_str, torch_device)


if __name__ == "__main__":
    main()