from summarizer import bpe_summarize


def test_small_text():
    blob = "The most meaningful part of this sentence is here. I frost apples. Bannas whwere foo."
    result = bpe_summarize(blob)

    assert result == "The most meaningful part of this sentence is here."
