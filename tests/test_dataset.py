from pathlib import Path

import pandas as pd

from src.data.dataset import load_raw_dataframe


def test_load_raw_dataframe_supports_updated_dataset_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "sentiment_data.csv"
    pd.DataFrame(
        {
            "Comment": ["This is great", "This is terrible"],
            "Sentiment": [2, 0],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_raw_dataframe(str(csv_path), "text", "sentiment")

    assert "text" in loaded.columns
    assert "sentiment" in loaded.columns
    assert loaded["text"].tolist() == ["This is great", "This is terrible"]
    assert loaded["sentiment"].tolist() == [2, 0]
