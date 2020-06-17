import os
import re
from typing import Dict, List, Tuple, Callable

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_text
from rouge import Rouge

SCISUMMNET_R1: str = os.getcwd() + "/summarizer/validation/data/scisummnet_release1.1__20190413/"

USENC_4: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
usenc_module = tfhub.load(USENC_4)


def _open_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def _extract_scicummnet(path: str) -> List[Tuple]:
    text_paths: list = []
    summary_paths: list = []

    if not os.path.isdir(path):
        raise Exception(f"Could not resolve path for {path}")

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            f = os.path.join(root, name)
            if "xml" in f:
                text_paths.append(f)
            elif "summary" in f:
                summary_paths.append(f)

    return list(zip(text_paths, summary_paths))


def scicummnet_validation(path: str = SCISUMMNET_R1) -> List[Tuple]:

    all_paths: list = _extract_scicummnet(path)
    validation_set: list = []
    for xml_path, summary_path in all_paths:
        # extract text from xml
        text: str = _open_file(xml_path)
        # remove markup tags
        text = re.sub(r"<[^>]*>", "", text)

        # extract text from summary
        summary: str = _open_file(summary_path)

        validation_set.append((text, summary))

    return validation_set


def rouge_metric(hypothesis: str, reference: str) -> List[Dict]:
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores


def mean_rouge_fscore(hyp: str, ref: str) -> float:
    rouge_score: List[Dict] = rouge_metric(hyp, ref)
    fscores: List[float] = [list(r.values())[0]["f"] for r in rouge_score]
    return np.mean(fscores)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Note that it is a negative quantity between -1 and 0,
    where 0 indicates orthogonality and values
    closer to -1 indicate greater similarity.
    """
    cos = tf.keras.losses.CosineSimilarity(axis=0)
    return cos(x, y).numpy()


def sentence_encoder(text: str) -> np.ndarray:
    encode: Callable = usenc_module.signatures["serving_default"]
    return encode(tf.convert_to_tensor([text]))["outputs"].numpy()


def similarity_score(a: str, b: str) -> float:
    avec: np.ndarray = sentence_encoder(a)
    bvec: np.ndarray = sentence_encoder(b)
    return cosine_similarity(avec, bvec)
