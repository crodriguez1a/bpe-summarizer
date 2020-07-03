import pytest

from src.bpe_summarizer import (bart_tokenizer, bpe_summarize,
                                summarize_sentence)


@pytest.fixture
def mock_tokens():
    # mock since tokenization can be nondeterministic between runtimes
    return [0, 3870, 16, 10, 46622, 14, 839, 227, 80, 1134, 4, 7299, 763, 16, 10, 46622, 61, 839, 624, 50, 1025, 65, 333, 4, 2]


def test_few_sentences():
    blob = "The most meaningful part of this sentence is here. I frost apples. Bannas whwere foo."
    result = bpe_summarize(blob)
    assert result == "The most meaningful part of this sentence is here."


def test_sentence_percentile_above_mean(mock_tokens):
    expected = "prefix Int prefix"
    tokens = mock_tokens
    assert summarize_sentence(tokens, 99, bart_tokenizer) == expected


def test_intra_sentence_custom_percentile(mock_tokens):
    tokens = mock_tokens
    expected = (
        "Inter prefix means between two groups Intra prefix means within inside group"
    )
    assert summarize_sentence(tokens, 50, bart_tokenizer) == expected


def test_intra_sentence(monkeypatch):
    monkeypatch.setattr(
        bart_tokenizer,
        "encode",
        lambda x: [0, 20, 4758, 1437, 3988, 10, 408, 18, 1040, 4, 2],
    )

    blob = "The cat in the hat is a children's book. The cat in the hat is a children's book."
    result = bpe_summarize(
        blob, percentile=0, apply_intra_sentence=True, intra_sentence_percentile=50
    )
    expected = "The cat hat children book The cat hat children book"
    assert result == expected
