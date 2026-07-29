"""Fine-tuned BERT sequence classification model for sentiment analysis."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BertSentimentClassifier(nn.Module):
    """
    BERT encoder -> dropout -> linear classification head.
    
    The [CLS] token's pooled representation is used for classification, and 
    the entire BERT encoder is fine-tuned end-to-end along with the head.
    """
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_labels: int = 3,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # Optionally freeze BERT for static embeddings
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Optionally apply LoRA adapters (requires `peft` package)
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["query", "key", "value"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                )

                # wrap the BERT encoder with LoRA adapters
                self.bert = get_peft_model(self.bert, lora_config)

                # Explicitly freeze base model parameters and leave only LoRA params trainable
                for name, param in self.bert.named_parameters():
                    if any(k in name for k in ("lora", "adapter", "peft")):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "LoRA requested but `peft` is not available. Install with `pip install peft`."
                ) from exc
        hidden_size = getattr(self.bert.config, "hidden_size", self.bert.config.hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Pooler output: [CLS] hidden state passed through a tanh dense layer.
        if outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            
        return {"loss": loss, "logits": logits}