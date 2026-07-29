"""Model architecture tests.

These instantiate the real BERT backbone ("prajjwal1/bert-tiny" by default)
and pass dummy tensors through the models to ensure shapes are correct.
Download the tiny model on first run. They require network access to
download the tiny model on first run, so they are skipped automatically if
network access is unavailable, unless RUN_MODEL_TESTS=1 to force-enable and fail loudly
instead of skipping.
"""
from __future__ import annotations

import os
import pytest
import torch
from transformers import AutoModel

# We use a tiny BERT model for testing to keep things fast
TINY_MODEL = "prajjwal1/bert-tiny"


def tiny_model_available() -> bool:
    if os.environ.get("RUN_MODEL_TESTS") == "1":
        return True
    
    try:
        # Test if we can fetch the model (or if it's cached)
        AutoModel.from_pretrained(TINY_MODEL)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not tiny_model_available(),
    reason="Requires network access to download a tiny BERT model for testing.",
)


@pytest.fixture
def dummy_batch(vocab_size: int = 30522, batch_size: int = 2, seq_len: int = 16):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.tensor([0, 2])
    
    return input_ids, attention_mask, labels


def test_bert_classifier_forward_shapes(dummy_batch):
    from src.models.bert_classifier import BertSentimentClassifier
    
    model = BertSentimentClassifier(
        bert_model_name=TINY_MODEL,
        num_labels=3,
        dropout=0.1,
    )
    
    input_ids, attention_mask, labels = dummy_batch
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    assert outputs["logits"].shape == (2, 3)
    assert outputs["loss"] is not None


def test_bert_lstm_classifier_forward_shapes(dummy_batch):
    from src.models.bert_lstm_classifier import BertLstmClassifier
    
    model = BertLstmClassifier(
        bert_model_name=TINY_MODEL,
        num_labels=3,
        lstm_hidden_size=32,
        lstm_num_layers=1,
        lstm_bidirectional=True,
        dropout=0.1,
    )
    
    input_ids, attention_mask, labels = dummy_batch
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    assert outputs["logits"].shape == (2, 3)
    assert outputs["loss"] is not None