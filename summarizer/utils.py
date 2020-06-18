import re

from nltk.corpus import stopwords

STOPWORDS: set = set(stopwords.words("english"))


def remove_stopwords(blob):
    words = set(blob.split(" "))
    stop_words_found = words.intersection(STOPWORDS)
    pat = re.compile(f"({' | '.join(stop_words_found)})")
    return re.sub(pat, " ", blob)
