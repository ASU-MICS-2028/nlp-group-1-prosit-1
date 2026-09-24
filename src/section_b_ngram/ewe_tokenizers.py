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
    """Splits purely on whitespace without punctuation awareness (lowercases like the other tokenizers)."""
    name = "Whitespace"

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        text = normalize_ewe_text(text)
        return (text.lower() if self.lowercase else text).split()


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
    Spaces become the token "▁", so word boundaries are predicted like any other character.
    """
    name = "Character"

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        text = normalize_ewe_text(text)
        if self.lowercase:
            text = text.lower()
        return ["▁" if c == " " else c for c in " ".join(text.split())]


class EweRuleStemmerTokenizer:
    """
    Morphology-aware tokenizer implementing rule-based prefix and suffix splitting
    tailored to Ewe grammatical structures:
    - Subject pronouns & verbal prefixes: mí- (we), wó- (they), nà- (you), me- (I)
    - Nominalizing prefix nu- (thing/object) and the compound head agble- (farm)
    - Suffixes: plural -wo, agentive/definite -la, -ye
    At most one prefix and one suffix are split off per word, e.g. nusrɔ̃lawo -> nu+ srɔ̃la +wo.
    The affixes stay in the output as their own tokens, so nothing is deleted and perplexity can be
    compared per word with the other tokenizers. (An earlier version deleted the affixes, which made
    the prediction task easier and its perplexity not comparable.) The rules are crude: megbe -> me+ gbe.
    """
    name = "Ewe Stemmer (affixes kept)"

    def __init__(self, lowercase: bool = True):
        self.base_tokenizer = UnicodeWordTokenizer(lowercase=lowercase)
        self.prefixes = ("míe", "wóe", "mí", "wó", "nà", "me", "nu", "agble")
        self.suffixes = ("wo", "la", "ye")

    def split_word(self, word: str) -> List[str]:
        if len(word) <= 3:
            return [word]
        prefix = suffix = ""

        # Split off one suffix (requiring at least 2-char root lemma)
        for sfx in self.suffixes:
            if word.endswith(sfx) and len(word) >= len(sfx) + 2:
                word, suffix = word[: -len(sfx)], sfx
                break

        # Split off one prefix (requiring at least 2-char root lemma)
        for pfx in self.prefixes:
            if word.startswith(pfx) and len(word) >= len(pfx) + 2:
                word, prefix = word[len(pfx) :], pfx
                break

        return ([prefix + "+"] if prefix else []) + [word] + (["+" + suffix] if suffix else [])

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for t in self.base_tokenizer.tokenize(text):
            tokens.extend(self.split_word(t) if re.match(r"^[\w\u0300-\u036f]+$", t) else [t])
        return tokens


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
        self._cache: Dict[str, List[str]] = {}

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
        self._cache.clear()
        self.vocab = {"<s>", "</s>", "<unk>"}
        for word_tuple in word_counts.keys():
            self.vocab.update(word_tuple)

    def tokenize_word(self, word: str) -> List[str]:
        if word not in self._cache:
            self._cache[word] = self._apply_merges(word)
        return self._cache[word]

    def _apply_merges(self, word: str) -> List[str]:
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
