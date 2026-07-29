from src.data.preprocessing import clean_text, id_to_label, label_to_id


def test_clean_text_lowercases_and_strips_urls():
    raw = "Check out this product: http://example.com AMAZING!!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "amazing" in cleaned


def test_clean_text_removes_html_and_mentions():
    raw = "<p>Hello</p> @someone this is great #awesome"
    cleaned = clean_text(raw)
    assert "<p>hello</p>" in cleaned
    assert "@someone" not in cleaned
    assert "#" not in cleaned
    assert "awesome" in cleaned


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_label_roundtrip():
    for label in ["negative", "neutral", "positive"]:
        assert id_to_label(label_to_id(label)) == label


def test_label_to_id_invalid_raises():
    import pytest
    
    with pytest.raises(ValueError):
        label_to_id("unknown_label")