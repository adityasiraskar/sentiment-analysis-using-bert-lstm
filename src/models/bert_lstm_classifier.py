"""Hybrid BERT + LSTM sentiment classification model."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BertLstmClassifier(nn.Module):
    """
    BERT -> BiLSTM -> Linear Head
    
    Extracts contextual token embeddings which are fed into a BiLSTM layer.
    The final LSTM hidden state is used for classification. This lets the model
    combine BERT's rich representations with an additional sequential encoder.
    """
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_labels: int = 3,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = True,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_hidden_size = getattr(self.bert.config, "hidden_size", self.bert.config.hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            batch_first=True,
        )
        
        lstm_out_size = lstm_hidden_size * (2 if lstm_bidirectional else 1)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_size, num_labels)
        
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
        
        # (batch, seq_len, hidden) contextual token embeddings from BERT
        sequence_output = outputs.last_hidden_state
        
        # Pass through LSTM
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        
        # Extract the final hidden state to use for classification
        if self.lstm.bidirectional:
            # Concat the final forward and backward hidden states
            final_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            final_hidden = h_n[-1, :, :]
            
        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            
        return {"loss": loss, "logits": logits}