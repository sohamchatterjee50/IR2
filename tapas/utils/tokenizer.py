## Helper Functions for Relevant Paragraph Finding in a Website via a Query

import nltk
import functools
import threading


@functools.lru_cache()
def _load_sentence_tokenizer():
    """Returns a sentence tokenization function."""
    # Lock to avoid a race-condition in the creation of the download directory.
    with threading.Lock():
        nltk.download("punkt")
        return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def tokenize(document):
    """Split text into sentences."""
    sentence_tokenizer = _load_sentence_tokenizer()
    result = []
    for sentence in sentence_tokenizer.tokenize(document):

        sentence = sentence.strip()
        if sentence:
            result.append(sentence)

    return result
