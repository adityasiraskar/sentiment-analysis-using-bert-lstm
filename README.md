# Sentiment Analysis: BERT & BERT+LSTM

End-to-end sentiment analysis project that classifies text as **positive**, **negative**, or **neutral** using transformer-based deep learning. Fine-tunes two model architectures for comparison:

1. **BERT classifier** — a pre-trained BERT encoder with a linear classification head. Supports either full fine-tuning or parameter-efficient tuning using LoRA adapters (recommended for large datasets).
2. **BERT+LSTM classifier** — uses BERT token embeddings (frozen by default) which are fed into a BiLSTM layer and a classification head. This hybrid reduces compute by training only the LSTM and head unless you explicitly unfreeze BERT.

The project covers data cleaning, tokenization, training, evaluation, inference, a FastAPI serving layer, and Docker containerization. Configuration is driven by YAML files and the training script supports mixed precision (AMP), gradient accumulation, and per-model hyperparameters for flexible, efficient training on large datasets.

The project covers data cleaning, tokenization, training, evaluation, inference, a FastAPI serving layer, and Docker containerization.

## Project Structure

```
sentiment/
├── api/
│   ├── main.py                # App + endpoints (/health, /predict, /predict/batch)
│   ├── model_loader.py        # Lazy singleton model loader (env-configurable)
│   └── schemas.py             # Pydantic request/response models
├── config/
│   └── config.yaml            # Data, model, and training hyperparameters (defaults)
├── config.yaml                # Optional legacy/override config (overrides defaults)
├── data/
│   ├── raw/dataset.csv        # Sample labeled dataset (text, sentiment) — 10 rows
│   └── processed/             # Generated artifacts (gitignored)
├── src/
│   ├── config.py               # YAML config loader (AttrDict-style access)
│   ├── data/
│   │   ├── preprocessing.py    # Text cleaning + label <-> id mapping
│   │   └── dataset.py          # Load/split CSV, PyTorch Dataset wrapper
│   ├── models/
│   │   ├── bert_classifier.py       # BERT + linear head
│   │   ├── bert_lstm_classifier.py  # BERT + BiLSTM + linear head
│   │   └── factory.py               # build/save/load model checkpoints
│   ├── training/
│   │   ├── train.py            # Fine-tuning loop for both architectures
│   │   └── evaluate.py         # Test-set metrics + model comparison
│   └── inference/
│       └── predict.py          # SentimentPredictor: raw text -> label + scores
├── utils/
│   └── logger.py
├── tests/                      # pytest unit tests (preprocessing, models, api)
├── models/                     # Saved checkpoints (gitignored, populated by training)
├── reports/                    # Evaluation JSON reports (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`src/data/preprocessing.py` cleans text with **NLTK**: word tokenization, stopword removal (negation words like "not"/"n't" are kept so sentiment meaning isn't flipped), and verb lemmatization. Required corpora (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`) are downloaded automatically on first use if not already cached.

## Dataset

`data/raw/dataset.csv` ships with **10 sample rows** (`text, sentiment`) purely so the pipeline can be run end-to-end immediately. Replace it with a full dataset (same two columns, labels in `{positive, negative, neutral}`) before training a model you intend to actually use — 10 rows is only enough to exercise the code path, not to learn a meaningful classifier.

To download a larger Kaggle dataset for this project, you can use:

```bash
#!/bin/bash
kaggle datasets download -d abdelmalekeladjelet/sentiment-analysis-dataset
```

Place the downloaded CSV in `data/raw/sentiment_data.csv` so the current config picks it up automatically.

## Training

Fine-tune either or both architectures (uses `bert-base-uncased` and a GPU if available; falls back to CPU automatically). Configuration is read from `config/config.yaml` with optional overrides from `config.yaml` in the repo root — the loader merges defaults then overrides.

Per-model hyperparameters
- You can configure model-specific settings under `training.per_model.<model_name>` in the YAML. Example fields: `batch_size`, `eval_batch_size`, `num_epochs`, `learning_rate`, `gradient_accumulation_steps`, `use_amp`, `freeze_bert`, `use_lora`, `lora_r`, `lora_alpha`, `lora_dropout`.

CLI runtime overrides
- The training script also accepts CLI flags to override common settings per run: `--batch-size`, `--eval-batch-size`, `--num-epochs`, `--learning-rate`, `--freeze-bert`, `--use-lora`, `--gradient-accumulation-steps`, `--use-amp`. CLI flags act as global overrides; per-model config in YAML remains the recommended place for reproducible experiments.

Performance-minded defaults (recommended for T4 / P100)
- Enable mixed precision: `use_amp: true` (on CUDA) to speed up training.
- Use gradient accumulation: e.g., `gradient_accumulation_steps: 2` to increase effective batch size without extra memory.
- Increase `eval_batch_size` to reduce validation wall time (I/O permitting).

Quick commands

- Train only the BERT model (LoRA adapters, AMP, accumulation):
```powershell
python -m src.training.train --model bert --use-lora --use-amp --batch-size 32 --num-epochs 3 --gradient-accumulation-steps 2
```

- Train only the BERT+LSTM model (frozen BERT; trains BiLSTM + head):
```powershell
python -m src.training.train --model bert_lstm --batch-size 32 --num-epochs 5 --use-amp --gradient-accumulation-steps 2
```

- Train both sequentially using per-model config from YAML:
```powershell
python -m src.training.train --model both
```

Checkpoints are saved to `models/<bert|bert_lstm>/` (weights, tokenizer, and architecture config), keeping the best validation macro-F1 epoch.

## Evaluation

```powershell
python -m src.training.evaluate --model both
```

Computes accuracy, macro precision/recall/F1, a confusion matrix, and a full classification report on the held-out test split, saving each to `reports/evaluation_<model>.json` and printing a side-by-side comparison table when both models are evaluated.

## Inference (Python)

```python
from src.inference.predict import SentimentPredictor

predictor = SentimentPredictor("models/bert")
print(predictor.predict("I really love how simple this is!"))
```

## Optional: LoRA adapters and static BERT embeddings

- LoRA (Low-Rank Adapters): you can enable lightweight parameter-efficient fine-tuning for the BERT classifier by setting `model.use_lora: true` in `config/config.yaml` and installing the `peft` package (`pip install peft`). LoRA is optional and will raise an informative error if requested but the package is missing.

- Static BERT embeddings (freeze BERT): set `model.freeze_bert: true` to freeze the BERT encoder weights and train only the downstream layers (useful when you want to train only a small classifier/LSTM on top of fixed BERT features).

Note: The hybrid `BERT+LSTM` pipeline uses frozen BERT embeddings by default (it trains only the BiLSTM + classifier). To change this behavior, set `model.freeze_bert: false` in `config/config.yaml`.

Recommended GPU batch sizes
- For NVIDIA T4 or P100 (approx. 16GB GPU memory) a good starting point is `training.batch_size: 32` and `training.eval_batch_size: 64` with `max_seq_length: 128`. If you have more memory, increase `batch_size` (e.g., 48–64) until you hit an OOM; if you hit OOM, reduce it.

To enable LoRA or freezing, edit `config/config.yaml` or set `SENTIMENT_CONFIG_PATH` to a custom YAML before running training.

## Running the API

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Set `MODEL_TYPE=bert_lstm` (and optionally `MODEL_DIR`) to serve the other checkpoint. Endpoints:

- `GET /health` — model load status
- `POST /predict` — `{"text": "..."}` -> label, confidence, probabilities
- `POST /predict/batch` — `{"texts": ["...", "..."]}`

## Docker

```powershell
docker compose build
docker compose up
```

The container expects trained checkpoints under `./models` on the host, mounted read-only into `/app/models` (see `docker-compose.yml`), since model weights are large binaries excluded from both the git repo and the Docker build context.

## Tests

```powershell
pytest
```

Model architecture tests download a tiny BERT model (pretrained-tiny variant) for fast, lightweight test runs.
