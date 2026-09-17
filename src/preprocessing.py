"""
Preprocessing, tokenization, and dataset streaming utilities for low-resource
African language corpora (e.g., Twi) and specialized domain datasets.
"""

from collections import Counter
import random
import re
from typing import List, Tuple, Set, Dict, Optional, Iterator


SPECIAL_BOS = "<s>"
SPECIAL_EOS = "</s>"
SPECIAL_UNK = "<unk>"


def load_twi_streaming_corpus(
    dataset_name: str = "ghana-nlp/abena-twi-corpus",
    split: str = "train",
    scale_factor: float = 0.05,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    Streams the Twi language corpus directly from Hugging Face hub to prevent
    system/Colab memory crashes, applying a scale factor sampling filter.

    Args:
        dataset_name: Hugging Face dataset ID (default: 'ghana-nlp/abena-twi-corpus').
        split: Dataset split to stream.
        scale_factor: Fraction of the stream to sample (e.g., 0.05 for 5%).
        seed: Random seed for reproducible sampling.
        max_samples: Optional hard cap on sampled lines.

    Returns:
        List of text strings.
    """
    from datasets import load_dataset

    random.seed(seed)
    print(f"Connecting to Hugging Face cloud database for '{dataset_name}'...")
    raw_stream = load_dataset(dataset_name, split=split, streaming=True)

    sampled_lines = []
    print(f"Sampling stream with scale factor: {scale_factor * 100:.1f}%...")
    for row in raw_stream:
        # Extract text field from corpus record
        text = row.get("text", "")
        if text and random.random() < scale_factor:
            sampled_lines.append(text.strip())
            if max_samples and len(sampled_lines) >= max_samples:
                break

    print(f"Successfully processed {len(sampled_lines)} lines for model training.")
    return sampled_lines


def basic_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenizes text into words and punctuation while preserving special characters
    common in African languages (e.g., tone markers, open-o 'ɔ', open-e 'ɛ', 'ŋ').

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
    Builds a vocabulary strictly from a tokenized corpus partition (e.g. training set)
    and assigns an unknown token for rare words to prevent test-set data leakage.

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
    Enables computing conditional probabilities for initial tokens: P(w_1 | <s>).

    Args:
        sentence: List of tokens.
        n: Order of the n-gram model.
        bos: Beginning of sentence token.
        eos: End of sentence token.

    Returns:
        Padded list of tokens.
    """
    return [bos] * (n - 1) + sentence + [eos]
