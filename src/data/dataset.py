from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config import config as project_config
from src.config import resolve_path


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _resolve_column_name(df: pd.DataFrame, requested_name: str, aliases: list[str]) -> str:
    """Resolve a requested column name against the dataframe headers and common aliases."""
    if requested_name in df.columns:
        return requested_name

    normalized_columns = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        normalized_alias = str(alias).strip().lower()
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    available = ", ".join(map(str, df.columns))
    raise KeyError(
        f"Could not find column '{requested_name}'. Available columns: {available}."
    )


def load_raw_dataframe(csv_path: str, text_column: str, label_column: str) -> pd.DataFrame:
    """Load the raw CSV, normalize common schema variations, and clean the text column."""
    path = resolve_path(csv_path)
    df = pd.read_csv(path)

    resolved_text_col = _resolve_column_name(
        df,
        text_column,
        ["text", "comment", "content", "tweet", "review", "sentence"],
    )
    resolved_label_col = _resolve_column_name(
        df,
        label_column,
        ["sentiment", "label", "target", "class", "sentiment_label"],
    )

    if resolved_text_col != text_column:
        df = df.rename(columns={resolved_text_col: text_column})
    if resolved_label_col != label_column:
        df = df.rename(columns={resolved_label_col: label_column})

    df = df[[text_column, label_column]].copy()
    df = df.dropna(subset=[text_column, label_column]).reset_index(drop=True)
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column].str.len() > 0].reset_index(drop=True)
    return df


def split_dataframe(
    df: pd.DataFrame,
    label_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DataSplits:
    """Split a dataframe into train/val/test, stratifying by label when possible.

    Falls back to a non-stratified split (and to trivial splits) when the
    dataset is too small per-class for stratification, which happens with
    tiny demo datasets.
    """

    def _safe_split(data: pd.DataFrame, size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(data) < 2 or size <= 0:
            return data, data.iloc[0:0]
        try:
            return train_test_split(
                data, test_size=size, random_state=random_state, stratify=data[label_column]
            )
        except ValueError:
            return train_test_split(data, test_size=size, random_state=random_state)

    train_val, test = _safe_split(df, test_size)
    relative_val_size = val_size / (1.0 - test_size) if test_size < 1.0 else 0
    train, val = _safe_split(train_val, relative_val_size)

    return DataSplits(train, val, test)


class SentimentDataset(Dataset):
    """Tokenizes text on the fly and returns model-ready tensors."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def build_datasets_from_config(cfg, tokenizer: PreTrainedTokenizerBase) -> Tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """Convenience helper: load, split, and wrap into `SentimentDataset`s using project config."""
    from src.data.preprocessing import clean_text, label_to_id
    
    df = load_raw_dataframe(
        cfg.data.raw_path, cfg.data.text_column, cfg.data.label_column
    )
    df[cfg.data.text_column] = df[cfg.data.text_column].apply(clean_text)
    df = df[df[cfg.data.text_column].str.len() > 0].reset_index(drop=True)
    
    df[cfg.data.label_column] = df[cfg.data.label_column].apply(label_to_id)
    
    splits = split_dataframe(
        df,
        cfg.data.label_column,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_state=cfg.data.random_state,
    )
    
    max_len = cfg.model.max_seq_length
    
    train_ds = SentimentDataset(
        texts=splits.train[cfg.data.text_column].tolist(),
        labels=splits.train[cfg.data.label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_len,
    )
    
    val_ds = SentimentDataset(
        texts=splits.val[cfg.data.text_column].tolist(),
        labels=splits.val[cfg.data.label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_len,
    )
    
    test_ds = SentimentDataset(
        texts=splits.test[cfg.data.text_column].tolist(),
        labels=splits.test[cfg.data.label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_len,
    )
    
    return train_ds, val_ds, test_ds