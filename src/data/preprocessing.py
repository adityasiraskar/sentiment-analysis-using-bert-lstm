"""Text cleaning and label encoding utilities for sentiment analysis.

Text Normalization uses NLTK for tokenization, stopword removal,
and lemmatization. This is often better than plain regex/string ops, since it gives more
linguistically-aware cleaning (e.g. handling contractions and word forms).
"""
from __future__ import annotations

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Basic text cleaning regexes
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
MULTI_SPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z0-9]+)\s*>")
PUNCTUATION_RE = re.compile(r"[^\w\s<>/]+")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# NLTK's default English stopword list includes negation words like "not"
# "no". Dropping these would flip sentiment meaning (e.g. "not good" ->
# "good"), so they are explicitly preserved during stopword removal.
NEGATION_WORDS = {
    "not", "no", "never", "none", "don't", "doesn't", "didn't",
    "can't", "couldn't", "won't", "wouldn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't"
}

def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora/models on first use (no-op once cached)."""
    required = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4"
    }
    for resource_path, package_name in required.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)

_ensure_nltk_resources()


def tokenize_text(text: str) -> List[str]:
    """Tokenize a string into words using NLTK's Punkt tokenizer."""
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    stop_words = set(stopwords.words("english"))
    filtered_stopwords = stop_words - NEGATION_WORDS
    return [tok for tok in tokens if tok.lower() not in filtered_stopwords]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using the verb part-of-speech (e.g. 'loved' -> 'love')."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok, pos="v") for tok in tokens]


def clean_text(text: str) -> str:
    """Clean a single raw text string for downstream tasks."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)

    def _normalize_tag(match: re.Match[str]) -> str:
        slash = match.group(1)
        name = match.group(2).lower()
        return f"<{slash}{name}>"

    text = HTML_TAG_RE.sub(_normalize_tag, text)
    text = PUNCTUATION_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()

    return text


def label_to_id(label: str) -> int:
    """Map a string sentiment label to its integer ID."""
    key = str(label).strip().lower()
    if key not in LABEL2ID:
        raise ValueError(
            f"Unknown label '{label}'. Expected one of {list(LABEL2ID.keys())}"
        )
    return LABEL2ID[key]


def id_to_label(id_: int) -> str:
    """Map an integer ID back to its sentiment label string."""
    if id_ not in ID2LABEL:
        raise ValueError(
            f"Unknown label ID {id_}. Expected one of {list(ID2LABEL.keys())}"
        )
    return ID2LABEL[id_]