from src.bpe_summarizer import bpe_summarize


def test_fewer_sentences():
    blob = "The most meaningful part of this sentence is here. I frost apples. Bannas whwere foo."
    result = bpe_summarize(blob)

    assert result == "The most meaningful part of this sentence is here."


def test_single_sentence():
    blob = "I received a notification today about being subject to extended holds due to repeated overdrafts."
    result = bpe_summarize(blob)

    assert result == "notification repeated overdraft"


def test_sentence_defaults_to_mean():
    blob = "I received a notification today about being subject to extended holds due to repeated overdrafts."
    result = bpe_summarize(blob, percentile=99)

    assert result == "notification repeated overdraft"
