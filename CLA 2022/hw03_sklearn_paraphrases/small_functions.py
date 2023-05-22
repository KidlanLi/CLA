import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def trigram_quadragram_vectorizer(texts):
    """
    Takes a list of text strings and returns a CountVectorizer that considers all trigrams and quadragrams that occur
    in at least 3 texts.

    >>> l = ["My name is Bond", "His name is not Bond", "I think your name is Bond", "I think my name is Bond", "You think my name is Bond"]
    >>> v = trigram_quadragram_vectorizer(l)
    >>> v.get_feature_names()
    ['my name is', 'my name is bond', 'name is bond']
    """
    vectorizer = CountVectorizer(min_df = 3, ngram_range = (3, 4))
    pre_texts = vectorizer.fit_transform(texts)

    return vectorizer
    # TODO: Exercise 1

