import logging
import re

import numpy as np
from nltk.tokenize import PunktSentenceTokenizer
from scipy import stats
from transformers import BartTokenizer, PreTrainedTokenizer

from src.utils import remove_stopwords

logger = logging.getLogger()

bart_tokenizer: BartTokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
sentencizer: PunktSentenceTokenizer = PunktSentenceTokenizer()


def summarize_sentence(
    tokens: list, percentile: float, tokenizer: PreTrainedTokenizer
) -> str:
    """For a single sentence, simply filter on the mean"""
    mn: float = np.mean(np.array(tokens))
    mn_percentile: float = stats.percentileofscore(tokens, mn)
    max_percentile = mn_percentile if percentile > mn_percentile else percentile

    logger.debug(f"max_percentile={max_percentile}")

    largest_threshold: float = np.percentile(np.array(tokens), max_percentile)

    decoded: str = tokenizer.decode([t for t in tokens if t >= largest_threshold])
    decoded = re.sub(r"\s{2,}", " ", decoded)
    return decoded.strip()


def bpe_summarize(
    document: str,
    percentile: float = 99.0,
    tokenizer: PreTrainedTokenizer = bart_tokenizer,
) -> str:
    sentences: list = sentencizer.tokenize(document)

    # find thresholds relative to all sentences
    tokenized: list = [(i, tokenizer.encode(remove_stopwords(i))) for i in sentences]
    group: list = np.concatenate([i for _, i in tokenized]).ravel().tolist()

    # find percentile
    group_threshold: float = np.percentile(np.array(group), percentile)

    # ensure the kth percentile is less than the max
    largest_threshold = np.max(group) - (np.max(group) * (100 - percentile))
    group_threshold = (
        group_threshold if group_threshold < np.max(group) else largest_threshold
    )

    if len(tokenized) == 1:
        _, tokens = tokenized[0]
        return summarize_sentence(tokens, percentile, tokenizer)

    result: list = []
    pruned: list = []
    for sentence, tokens in tokenized:
        top_pk: float = np.percentile(np.array(tokens), percentile)
        if top_pk >= group_threshold:
            result.append((sentence, top_pk))
        else:
            # keep pruned sentences for debugging
            pruned.append((sentence, top_pk))

    summarized: str = " ".join([r for r, _ in result])

    logger.info(f"Summarization: {summarized}")
    logger.debug(f"Pruned sentences: {pruned}")

    return summarized or document
