import os
import re

import numpy as np
from nltk.tokenize import PunktSentenceTokenizer
from scipy import stats
from transformers import BartTokenizer, PreTrainedTokenizer

STOPWORDS: set = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
}

bart_tokenizer: BartTokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
sentencizer: PunktSentenceTokenizer = PunktSentenceTokenizer()


def remove_stopwords(blob):
    words = set(blob.split(" "))
    stop_words_found = words.intersection(STOPWORDS)
    pat = re.compile(f"({' | '.join(stop_words_found)})")
    return re.sub(pat, " ", blob)


def summarize_sentence(
    tokens: list, percentile: float, tokenizer: PreTrainedTokenizer, raw: str = ""
) -> str:
    """For a single sentence, filter on the mean
    when the top kth percentile token is above the mean.
    This rule should prevent meaningless summarization
    at the sentence level
    """

    # find percentile of token that represents the mean of tokens
    mn_percentile: float = stats.percentileofscore(tokens, np.mean(np.array(tokens)))
    allowable_percentile: float = mn_percentile if percentile > mn_percentile else percentile

    top_k_tkn: int = int(np.percentile(np.array(tokens), allowable_percentile))
    decoded: str = tokenizer.decode([t for t in tokens if t >= top_k_tkn])

    decoded = re.sub(r"\s{2,}", " ", decoded)
    decoded = decoded.strip()

    if len(decoded) <= 2:
        return raw

    return decoded


def bpe_summarize(
    document: str,
    percentile: float = 99.0,
    tokenizer: PreTrainedTokenizer = bart_tokenizer,
    apply_intra_sentence: bool = False,
    intra_sentence_percentile: float = 50,
) -> str:
    """This summarizer attempts to leverage Byte Pair Encoding (BPE) tokenization
    with a pre-trained vocabulary to filter text by semantic meaningfulness.

    Keyword arguments:
    percentile == Sentences that include tokens in the top kth percentile  will
    remain after summarization (default 99.0)

    tokenizer == A PreTrainedTokenizer instance that relies on
    byte-pair-encoding (default BartTokenizer)

    apply_intra_sentence == If `True`, summarization will be applied at both the
    document level and the sentence level (default False)

    intra_sentence_percentile ==  When `apply_intra_sentence` is `True`, this
    percentile will be applied to individual sentences (default 50.0)
    """
    # parse sentences from document text
    sentences: list = sentencizer.tokenize(document)

    # tokenize all sentences
    tokenized: list = [(i, tokenizer.encode(remove_stopwords(i))) for i in sentences]
    group: list = np.concatenate([i for _, i in tokenized]).ravel().tolist()

    # find the token that represents the top kth percentile for all sentences
    group_top_k_tkn: int = int(np.percentile(np.array(group), percentile))

    # always summarize single sentence unless explicitly said not to
    if len(tokenized) == 1:
        _, tokens = tokenized[0]
        return summarize_sentence(tokens, percentile, tokenizer, raw=document)

    # filter for top k sentences
    result: list = []
    for sentence, tokens in tokenized:
        # only append sentences that have tokens in the top k
        if max(tokens) >= group_top_k_tkn:
            result.append((sentence, tokens))

    # optionally, summarize individual sentences
    summarized: str = ""
    if apply_intra_sentence:
        intra_sentence: list = [
            summarize_sentence(t, intra_sentence_percentile, tokenizer)
            for _, t in result
        ]
        summarized = " ".join(intra_sentence)
    else:
        summarized = " ".join([r for r, _ in result])

    return summarized or document
