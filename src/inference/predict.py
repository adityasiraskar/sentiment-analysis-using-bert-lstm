"""Reusable inference helper for running trained sentiment models on raw text."""
from __future__ import annotations

from typing import Dict, List, Union

import torch

from src.data.preprocessing import ID2LABEL, clean_text
from src.models.factory import load_model


class SentimentPredictor:
    """Loads a trained model checkpoint and exposes a simple `predict` API."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        self.model, self.tokenizer, self.config = load_model(model_dir, device=self.device)
        self.model.eval()

        if hasattr(self.config, "get"):
            self.max_length = int(self.config.get("max_seq_length", 128))
        else:
            self.max_length = int(getattr(self.config, "max_seq_length", 128))

    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Predict sentiment for one or more raw text strings.

        Returns a list of dicts with keys: 'text', 'label', 'confidence',
        and 'probabilities' (probs for each class).
        """
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return []

        clean_texts = [clean_text(t) for t in texts]

        encoding = self.tokenizer(
            clean_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        results = []
        for i, original_text in enumerate(texts):
            label_id = int(torch.argmax(logits[i]).item())
            label = ID2LABEL[label_id]

            probabilities = {
                ID2LABEL[j]: round(float(probs[i, j]), 4) for j in range(probs.shape[1])
            }

            results.append(
                {
                    "text": original_text,
                    "label": label,
                    "confidence": round(float(probs[i, label_id]), 4),
                    "probabilities": probabilities,
                }
            )

        return results