import os
import re

import nltk
from nltk.corpus import stopwords

nltk.data.path.append("src/nltk_data/")


STOPWORDS: set = set(stopwords.words("english"))


def remove_stopwords(blob):
    words = set(blob.split(" "))
    stop_words_found = words.intersection(STOPWORDS)
    pat = re.compile(f"({' | '.join(stop_words_found)})")
    return re.sub(pat, " ", blob)
