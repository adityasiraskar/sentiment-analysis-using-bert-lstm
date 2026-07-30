from __future__ import annotations

import json

import torch

from src.models import factory


class FakeBertModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded_state_dict = None
        self.moved_to = None
        FakeBertModel.instances.append(self)

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict

    def to(self, device):
        self.moved_to = torch.device(device)
        return self


class FakeTokenizer:
    pass


def test_load_model_moves_model_to_requested_device(tmp_path, monkeypatch):
    FakeBertModel.instances.clear()
    model_dir = tmp_path / "bert"
    model_dir.mkdir()

    model_config = {
        "model_name": "bert",
        "bert_model_name": "fake-bert",
        "num_labels": 3,
        "dropout": 0.1,
        "freeze_bert": False,
        "use_lora": False,
    }
    (model_dir / factory.MODEL_CONFIG_FILENAME).write_text(
        json.dumps(model_config), encoding="utf-8"
    )
    torch.save({"classifier.weight": torch.tensor([1.0])}, model_dir / factory.MODEL_FILENAME)

    monkeypatch.setattr(factory, "BertSentimentClassifier", FakeBertModel)
    monkeypatch.setattr(
        factory.AutoTokenizer,
        "from_pretrained",
        lambda path: FakeTokenizer(),
    )

    model, tokenizer, loaded_config = factory.load_model(model_dir, device="cuda")

    assert isinstance(model, FakeBertModel)
    assert isinstance(tokenizer, FakeTokenizer)
    assert loaded_config == model_config
    assert model.loaded_state_dict["classifier.weight"].device.type == "cpu"
    assert model.moved_to == torch.device("cuda")
