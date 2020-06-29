import logging

import numpy as np
from nltk.tokenize import PunktSentenceTokenizer
from transformers import BartTokenizer, PreTrainedTokenizer

from src.utils import remove_stopwords

logger = logging.getLogger()

bart_tokenizer: BartTokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
sentencizer: PunktSentenceTokenizer = PunktSentenceTokenizer()


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
    group_pk: float = np.percentile(np.array(group), percentile)

    # ensure the kth percentile is less than the max
    maxed_pk = np.max(group) - (np.max(group) * (100 - percentile))
    group_pk = group_pk if group_pk < np.max(group) else maxed_pk

    result: list = []
    pruned: list = []
    for sentence, tokens in tokenized:
        top_pk: float = np.percentile(np.array(tokens), percentile)
        if top_pk >= group_pk:
            result.append((sentence, top_pk))
        else:
            # keep pruned sentences for debugging
            pruned.append((sentence, top_pk))

    summarized: str = " ".join([r for r, _ in result])

    logger.info(f"Summarization: {summarized}")
    logger.debug(f"Pruned sentences: {pruned}")

    return summarized or document
