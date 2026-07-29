"""
Evaluate trained sentiment models on the held-out test split.

Usage:
    python -m src.training.evaluate --model bert
    python -m src.training.evaluate --model bert_lstm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_path
from src.data.dataset import build_datasets_from_config
from src.data.preprocessing import ID2LABEL
from src.models.factory import load_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        
        outputs = model(**inputs)
        preds = torch.argmax(outputs["logits"], dim=1)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
    return all_labels, all_preds


def evaluate_model(model_name: str, cfg, device, torch_device):
    model_dir = resolve_path(cfg.training.output_dir) / model_name
    
    try:
        model, tokenizer, _ = load_model(model_dir, device=torch_device)
    except FileNotFoundError:
        logger.error(f"No trained model found at {model_dir}. Run training first:")
        logger.error(f"  python -m src.training.train --model {model_name}")
        return None
        
    _, _, test_ds = build_datasets_from_config(cfg, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.eval_batch_size)
    
    y_true, y_pred = predict_loader(model, test_loader, torch_device)
    
    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True
        ),
        "num_test_samples": len(y_true),
    }
    
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "bert_lstm", "both"],
        default="both",
        help="Which model architecture to evaluate.",
    )
    args = parser.parse_args()
    
    cfg = load_config()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    models_to_evaluate = ["bert", "bert_lstm"] if args.model == "both" else [args.model]
    
    all_metrics = {}
    for m_name in models_to_evaluate:
        logger.info(f"Evaluating {m_name}...")
        metrics = evaluate_model(m_name, cfg, device, torch_device)
        if metrics:
            all_metrics[m_name] = metrics
            
    if not all_metrics:
        logger.warning("No models were evaluated.")
        return
        
    report_path = resolve_path("reports") / "evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
        
    logger.info(f"Saved detailed report to {report_path}")
    
    if len(all_metrics) > 1:
        logger.info("\n--- Model Comparison (Test Set) ---")
        logger.info(f"{'Model':<15} | {'Accuracy':<10} | {'F1 (Macro)':<10}")
        logger.info("-" * 40)
        for name, res in all_metrics.items():
            logger.info(f"{name:<15} | {res['accuracy']:.4f}     | {res['f1_macro']:.4f}")
    else:
        name = list(all_metrics.keys())[0]
        res = all_metrics[name]
        logger.info(f"\n--- {name} Results ---")
        logger.info(f"Accuracy:  {res['accuracy']:.4f}")
        logger.info(f"Precision: {res['precision_macro']:.4f}")
        logger.info(f"Recall:    {res['recall_macro']:.4f}")
        logger.info(f"F1 Macro:  {res['f1_macro']:.4f}")


if __name__ == "__main__":
    main()