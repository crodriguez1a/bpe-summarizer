from src.bpe_summarizer import STOPWORDS, remove_stopwords


def test_remove_stopwords():
    result = remove_stopwords("hello my name is Bruce Leroy")
    assert [r for r in result.split() if r in STOPWORDS] == []
