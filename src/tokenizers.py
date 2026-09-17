"""
Comprehensive tokenization suite for low-resource NLP experimentation in Ewe (Èʋegbe):
1. WhitespaceTokenizer (raw split on whitespace)
2. UnicodeWordTokenizer (orthography-preserving regex with NFC normalization)
3. CharacterTokenizer (atomic character-level representation)
4. EweRuleStemmerTokenizer (morphological affix stripping for Ewe)
5. BPETokenizer (Byte-Pair Encoding subword segmentation)
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Set, Dict, Optional
import unicodedata
import re


def normalize_ewe_text(text: str) -> str:
    """Applies Unicode NFC normalization to preserve combining diacritics."""
    return unicodedata.normalize("NFC", text.strip())


class WhitespaceTokenizer:
    """Splits purely on whitespace without punctuation awareness."""
    name = "Whitespace"

    def tokenize(self, text: str) -> List[str]:
        return normalize_ewe_text(text).split()


class UnicodeWordTokenizer:
    """
    Punctuation-aware regex tokenizer that strictly preserves Ewe's distinctive
    Latin characters (ɖ, ƒ, ɣ, ŋ, ɔ, ɛ, ʋ) and combining tone diacritics.
    """
    name = "Unicode Word"

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        text = normalize_ewe_text(text)
        if self.lowercase:
            text = text.lower()
        # Capture word characters including Unicode combining diacritics (\u0300-\u036f)
        return re.findall(r"[\w\u0300-\u036f]+|[^\w\s]", text, re.UNICODE)


class CharacterTokenizer:
    """
    Splits text into individual unicode characters (letters and punctuation),
    completely eliminating out-of-vocabulary words at the expense of sequence length.
    """
    name = "Character"

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        text = normalize_ewe_text(text)
        if self.lowercase:
            text = text.lower()
        # Filter whitespace or represent space as special marker
        return [c for c in text if c != " "]


class EweRuleStemmerTokenizer:
    """
    Morphology-aware tokenizer implementing rule-based prefix and suffix stripping
    tailored to Ewe grammatical structures:
    - Subject pronouns & verbal prefixes: mí- (we), wó- (they), nà- (you), me- (I)
    - Nominalizing & agentive affixes: nu- (thing/object), a- (nominal prefix)
    - Plural suffix: -wo (e.g. nusrɔ̃lawo -> nusrɔ̃la)
    """
    name = "Ewe Stemmer"

    def __init__(self, lowercase: bool = True):
        self.base_tokenizer = UnicodeWordTokenizer(lowercase=lowercase)
        # Common Ewe affixes
        self.prefixes = ("mí", "wó", "nà", "me", "nu", "agble")
        self.suffixes = ("wo", "la", "ye")

    def stem_word(self, word: str) -> str:
        if len(word) <= 3:
            return word

        # Strip plural and definite suffixes
        for sfx in self.suffixes:
            if word.endswith(sfx) and len(word) > len(sfx) + 2:
                word = word[: -len(sfx)]
                break

        # Strip common pronominal prefixes
        for pfx in self.prefixes:
            if word.startswith(pfx) and len(word) > len(pfx) + 2:
                word = word[len(pfx) :]
                break

        return word

    def tokenize(self, text: str) -> List[str]:
        tokens = self.base_tokenizer.tokenize(text)
        return [self.stem_word(t) if t.isalnum() else t for t in tokens]


class SimpleBPETokenizer:
    """
    Native Byte-Pair Encoding (BPE) subword tokenizer learned iteratively from text.
    Merges the most frequent byte/character pairs to build a compact subword lexicon.
    Zero external dependencies to prevent runtime crashes.
    """
    name = "Byte-Pair Encoding (BPE)"

    def __init__(self, num_merges: int = 100, lowercase: bool = True):
        self.num_merges = num_merges
        self.lowercase = lowercase
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Set[str] = set()

    def train(self, corpus: List[str]):
        """Learns subword merge rules from raw corpus sentences."""
        word_counts = Counter()
        for line in corpus:
            text = normalize_ewe_text(line)
            if self.lowercase:
                text = text.lower()
            words = re.findall(r"[\w\u0300-\u036f]+|[^\w\s]", text, re.UNICODE)
            for w in words:
                # Represent word as space-separated characters with end-of-word marker </w>
                char_tuple = tuple(list(w) + ["</w>"])
                word_counts[char_tuple] += 1

        # Iterative pair merging
        for _ in range(self.num_merges):
            pairs = Counter()
            for word_tuple, freq in word_counts.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pairs[pair] += freq

            if not pairs:
                break

            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)

            # Apply merge to word counts
            new_word_counts = Counter()
            pair_str_0, pair_str_1 = best_pair
            for word_tuple, freq in word_counts.items():
                new_tuple = []
                i = 0
                while i < len(word_tuple):
                    if (
                        i < len(word_tuple) - 1
                        and word_tuple[i] == pair_str_0
                        and word_tuple[i + 1] == pair_str_1
                    ):
                        new_tuple.append(pair_str_0 + pair_str_1)
                        i += 2
                    else:
                        new_tuple.append(word_tuple[i])
                        i += 1
                new_word_counts[tuple(new_tuple)] += freq
            word_counts = new_word_counts

        # Build final vocabulary from pieces
        self.vocab = {"<s>", "</s>", "<unk>"}
        for word_tuple in word_counts.keys():
            self.vocab.update(word_tuple)

    def tokenize_word(self, word: str) -> List[str]:
        word_tuple = tuple(list(word) + ["</w>"])
        for p0, p1 in self.merges:
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if (
                    i < len(word_tuple) - 1
                    and word_tuple[i] == p0
                    and word_tuple[i + 1] == p1
                ):
                    new_tuple.append(p0 + p1)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            word_tuple = tuple(new_tuple)
        return list(word_tuple)

    def tokenize(self, text: str) -> List[str]:
        text = normalize_ewe_text(text)
        if self.lowercase:
            text = text.lower()
        words = re.findall(r"[\w\u0300-\u036f]+|[^\w\s]", text, re.UNICODE)
        tokens = []
        for w in words:
            if not self.merges:
                tokens.extend(list(w))
            else:
                tokens.extend(self.tokenize_word(w))
        return tokens
