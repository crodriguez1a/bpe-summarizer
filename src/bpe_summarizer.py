import re

import numpy as np
from nltk.tokenize import PunktSentenceTokenizer
from scipy import stats
from transformers import BartTokenizer, PreTrainedTokenizer

from src.utils import remove_stopwords

bart_tokenizer: BartTokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
sentencizer: PunktSentenceTokenizer = PunktSentenceTokenizer()


def summarize_sentence(
    tokens: list, percentile: float, tokenizer: PreTrainedTokenizer
) -> str:
    """For a single sentence, filter on the mean
    when the top kth percentile token is above the mean.
    This rule should prevent meaningless summarization"""

    # find percentile of token that represents the mean of tokens
    mn_percentile: float = stats.percentileofscore(tokens, np.mean(np.array(tokens)))
    allowable_percentile: float = mn_percentile if percentile > mn_percentile else percentile

    top_k_tkn: int = int(np.percentile(np.array(tokens), allowable_percentile))
    decoded: str = tokenizer.decode([t for t in tokens if t >= top_k_tkn])

    decoded = re.sub(r"\s{2,}", " ", decoded)
    return decoded.strip()


def bpe_summarize(
    document: str,
    percentile: float = 99.0,
    tokenizer: PreTrainedTokenizer = bart_tokenizer,
    apply_intra_sentence: bool = False,
    intra_sentence_percentile: float = 50,
) -> str:
    sentences: list = sentencizer.tokenize(document)

    # tokenize all sentences
    tokenized: list = [(i, tokenizer.encode(remove_stopwords(i))) for i in sentences]
    group: list = np.concatenate([i for _, i in tokenized]).ravel().tolist()

    # find the token that represents the top kth percentile for all sentences
    group_top_k_tkn: int = int(np.percentile(np.array(group), percentile))

    # ensure the kth percentile is less than the max possible
    mx_percentile: float = stats.percentileofscore(group, np.max(group))
    mx_top_k_tkn: int = int(np.percentile(np.array(group), mx_percentile))
    group_top_k_tkn = (
        group_top_k_tkn if group_top_k_tkn < mx_top_k_tkn else mx_top_k_tkn
    )

    # always summarize single sentence
    if len(tokenized) == 1:
        _, tokens = tokenized[0]
        return summarize_sentence(tokens, percentile, tokenizer)

    # filter for top k sentences
    result: list = []
    for sentence, tokens in tokenized:
        top_k_tkn: int = int(np.percentile(np.array(tokens), percentile))
        # only append sentences that have tokens in the top k
        if top_k_tkn >= group_top_k_tkn:
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
