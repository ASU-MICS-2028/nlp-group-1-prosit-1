"""
Preprocessing, tokenization, and dataset loading utilities for low-resource
African language corpora (specifically Ewe / Èʋegbe) and domain datasets.
"""

from collections import Counter
from pathlib import Path
import random
import re
from typing import List, Tuple, Set, Dict, Optional
import unicodedata


SPECIAL_BOS = "<s>"
SPECIAL_EOS = "</s>"
SPECIAL_UNK = "<unk>"

# Ewe specific orthographic glyphs:
# Vowels: a, e, ɛ, i, o, ɔ, u (plus nasal accents)
# Consonants with unique unicode: ɖ, ƒ, ɣ, ŋ, ʋ (and uppercase Ɖ, Ƒ, Ɣ, Ŋ, Ʋ)
EWE_SPECIAL_CHARS = set("ɖƒɣŋɔɛʋƉƑƔŊƆƐƲ")


def normalize_ewe_text(text: str) -> str:
    """
    Applies Unicode NFC normalization to ensure combining tone marks and diacritics
    remain fused to their base vowels in Ewe (e.g., preventing 'ɛ' + accent from decomposing).
    """
    return unicodedata.normalize("NFC", text.strip())


def basic_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenizes text into words and punctuation while strictly preserving
    Ewe orthography (e.g. 'ɖ', 'ŋ', 'ɔ', 'ɛ', 'ʋ', 'ƒ', 'ɣ' and tone diacritics).

    Args:
        text: Input string to tokenize.
        lowercase: Whether to convert text to lowercase.

    Returns:
        List of string tokens.
    """
    text = normalize_ewe_text(text)
    if lowercase:
        text = text.lower()

    # Regex capturing unicode words (including all extended Latin Ewe letters) and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens


def load_corpus_from_file_or_hf(
    source_path_or_id: str,
    split: str = "train",
    scale_factor: float = 1.0,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    Loads text lines either from a local file (e.g. data/raw/low_resource/ewe.txt)
    or streams from a Hugging Face dataset ID.

    Args:
        source_path_or_id: File path or Hugging Face dataset identifier.
        split: Dataset split to use if streaming from Hugging Face.
        scale_factor: Sampling fraction (between 0.0 and 1.0).
        seed: Random seed for reproducible sampling.
        max_samples: Maximum number of lines to retain.

    Returns:
        List of normalized text sentences.
    """
    random.seed(seed)
    path = Path(source_path_or_id)

    if path.exists() and path.is_file():
        print(f"Loading local corpus from: {path}...")
        with open(path, "r", encoding="utf-8") as f:
            lines = [normalize_ewe_text(line) for line in f if line.strip()]
        if scale_factor < 1.0:
            lines = [line for line in lines if random.random() < scale_factor]
        if max_samples:
            lines = lines[:max_samples]
        print(f"Loaded {len(lines)} lines from local file.")
        return lines

    # Otherwise, attempt Hugging Face streaming
    try:
        from datasets import load_dataset

        print(f"Connecting to Hugging Face cloud database for '{source_path_or_id}'...")
        raw_stream = load_dataset(source_path_or_id, split=split, streaming=True)
        sampled_lines = []
        for row in raw_stream:
            text = row.get("text", "") or row.get("translation", {}).get("ee", "") or row.get("sentence", "")
            if text and (scale_factor >= 1.0 or random.random() < scale_factor):
                sampled_lines.append(normalize_ewe_text(text))
                if max_samples and len(sampled_lines) >= max_samples:
                    break
        print(f"Streamed {len(sampled_lines)} lines from Hugging Face.")
        return sampled_lines
    except Exception as e:
        print(f"Notice: Could not stream from '{source_path_or_id}': {e}")
        return []


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
