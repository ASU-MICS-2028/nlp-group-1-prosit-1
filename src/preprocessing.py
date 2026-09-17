"""
Preprocessing and tokenization utilities for low-resource text and domain datasets.
"""

from collections import Counter
from typing import List, Tuple, Set, Dict, Optional
import re


SPECIAL_BOS = "<s>"
SPECIAL_EOS = "</s>"
SPECIAL_UNK = "<unk>"


def basic_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenizes text into words and punctuation while preserving special characters
    common in African languages (e.g., tone markers, open-o, open-e).

    Args:
        text: Input string to tokenize.
        lowercase: Whether to convert text to lowercase.

    Returns:
        List of string tokens.
    """
    if lowercase:
        text = text.lower()
    # Match words (including unicode characters for African orthographies) and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens


def build_vocabulary(
    tokenized_corpus: List[List[str]],
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Builds a vocabulary from tokenized text and assigns an unknown token for rare words.

    Args:
        tokenized_corpus: List of tokenized sentences.
        min_freq: Minimum frequency for a word to be included.
        max_vocab_size: Optional upper limit on vocabulary size.

    Returns:
        A tuple of (vocabulary set, frequency dictionary).
    """
    counter = Counter()
    for sentence in tokenized_corpus:
        counter.update(sentence)

    vocab = {SPECIAL_BOS, SPECIAL_EOS, SPECIAL_UNK}
    for word, freq in counter.most_common(max_vocab_size):
        if freq >= min_freq:
            vocab.add(word)

    return vocab, dict(counter)


def replace_oov_tokens(
    tokenized_corpus: List[List[str]],
    vocab: Set[str],
    unk_token: str = SPECIAL_UNK,
) -> List[List[str]]:
    """
    Replaces Out-Of-Vocabulary (OOV) tokens with the UNK token.

    Args:
        tokenized_corpus: List of tokenized sentences.
        vocab: Known vocabulary set.
        unk_token: String representation for unknown words.

    Returns:
        Corpus with OOV tokens replaced.
    """
    return [
        [token if token in vocab else unk_token for token in sentence]
        for sentence in tokenized_corpus
    ]


def pad_sentence(
    sentence: List[str],
    n: int,
    bos: str = SPECIAL_BOS,
    eos: str = SPECIAL_EOS,
) -> List[str]:
    """
    Pads a tokenized sentence with (n-1) BOS tokens and 1 EOS token.

    Args:
        sentence: List of tokens.
        n: Order of the n-gram model.
        bos: Beginning of sentence token.
        eos: End of sentence token.

    Returns:
        Padded list of tokens.
    """
    return [bos] * (n - 1) + sentence + [eos]
