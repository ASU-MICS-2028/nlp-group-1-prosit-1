"""A rule-based Ewe tokenizer that splits common affixes off words and keeps them as their own tokens.

Ewe marks meaning with short prefixes and suffixes: subject/verbal prefixes (mí- we, wó- they, nà- you, me- I),
the nominaliser nu- (thing) and compound head agble- (farm), and suffixes -wo (plural), -la (agentive/definite),
-ye. At most one prefix and one suffix are split per word, and each keeps a "+" marker, so nothing is deleted and
the text can be rebuilt: nusrɔ̃lawo -> nu+ | srɔ̃la | +wo. The rules are crude (megbe -> me+ | gbe).

register() adds it to stage2_ngram.TOKENIZERS as "ewe_stem", so every model in the project can use it.
"""
import re

import stage2_ngram

PREFIXES = ("míe", "wóe", "mí", "wó", "nà", "me", "nu", "agble")  # longest forms first, so "míe" wins over "mí"
SUFFIXES = ("wo", "la", "ye")
WORD = re.compile(r"[\ẁ-ͯ]+|[^\w\s]")  # \w misses combining marks (the tilde in ɔ̃), so allow them explicitly
WORD_ONLY = re.compile(r"^[\ẁ-ͯ]+$")


def split_word(word):
    """One suffix, then one prefix, each only if at least two characters of root remain."""
    if len(word) <= 3:
        return [word]
    prefix = suffix = ""
    for sfx in SUFFIXES:
        if word.endswith(sfx) and len(word) >= len(sfx) + 2:
            word, suffix = word[: -len(sfx)], sfx
            break
    for pfx in PREFIXES:
        if word.startswith(pfx) and len(word) >= len(pfx) + 2:
            word, prefix = word[len(pfx):], pfx
            break
    return ([prefix + "+"] if prefix else []) + [word] + (["+" + suffix] if suffix else [])


def tokenize(text):
    """Words and punctuation as separate tokens (lower-cased), then affixes split off each word."""
    out = []
    for t in WORD.findall(text.lower()):
        out.extend(split_word(t) if WORD_ONLY.match(t) else [t])
    return out


def register():
    stage2_ngram.TOKENIZERS["ewe_stem"] = tokenize
    return "ewe_stem"


if __name__ == "__main__":
    for w in ("nusrɔ̃lawo", "míewɔ", "agbledela", "megbe", "va"):
        print(f"{w:<12} -> {' | '.join(split_word(w))}")
