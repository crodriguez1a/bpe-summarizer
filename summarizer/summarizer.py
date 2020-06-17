import logging
import re

import numpy as np
from nltk.tokenize import PunktSentenceTokenizer
from transformers import GPT2Tokenizer

logger = logging.getLogger()

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
sentencizer: PunktSentenceTokenizer = PunktSentenceTokenizer()


def bpe_summarize(document: str, percentile: float = 99.0) -> str:
    sentences: list = sentencizer.tokenize(document)

    # find thresholds relative to all sentences
    tokenized: list = [(i, tokenizer.encode(i)) for i in sentences]
    group: list = np.concatenate([i for _, i in tokenized]).ravel().tolist()

    # find percentile
    group_pk: float = np.percentile(np.array(group), percentile)

    # ensure the kth percentile is less than the max
    maxed_pk = np.max(group) - (np.max(group) * (100 - percentile))
    group_pk = group_pk if group_pk < np.max(group) else maxed_pk

    result: list = []
    pruned: list = []
    for sentence, tokens in tokenized:
        if np.percentile(np.array(tokens), percentile) >= group_pk:
            result.append((sentence, np.std(tokens)))
        else:
            # keep pruned sentences for debugging
            pruned.append((sentence, np.std(tokens)))

    summarized: str = " ".join([r for r, _ in result])

    logger.debug(f"Pruned sentences: {pruned}")

    return summarized or document
