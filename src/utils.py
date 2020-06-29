import os
import re

import nltk
from nltk.corpus import stopwords

dir_name = os.path.dirname(__file__)
file_name = os.path.join(dir_name, "nltk_data/")
nltk.data.path.append(file_name)

STOPWORDS: set = set(stopwords.words("english"))


def remove_stopwords(blob):
    words = set(blob.split(" "))
    stop_words_found = words.intersection(STOPWORDS)
    pat = re.compile(f"({' | '.join(stop_words_found)})")
    return re.sub(pat, " ", blob)
