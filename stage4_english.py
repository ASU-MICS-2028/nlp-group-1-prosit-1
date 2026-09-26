"""Stage 4 explore: run the same pipeline on the English side of the corpus and compare it with Ewe.

Caveat (see the notes, Stage 0 decision 7): the English sentences are the mined "translations", and many are misaligned,
so this is "English text of similar size from the same crawl", not the same content.
Run `uv run python stage4_english.py`; the notebook reads results/stage4_english.json.
"""
import json
from collections import Counter
from pathlib import Path

from stage2_ngram import TOKENIZERS
from stage3_smoothing import Smoothed

SPLIT, RESULTS, TOK, BUDGET = Path("data/splits/90-5-5"), Path("results/stage4_english.json"), "punct", 1_000_000


def profile(lang):
    train = (SPLIT / f"train.{lang}.txt").read_text(encoding="utf-8").splitlines()
    dev = (SPLIT / f"dev.{lang}.txt").read_text(encoding="utf-8").splitlines()
    tokens = [w for s in train for w in TOKENIZERS[TOK](s)]
    counts = Counter(tokens)
    first = Counter(tokens[:BUDGET])  # type/token ratio depends on corpus size, so compare at the same token count
    dev_tokens = [w for s in dev for w in TOKENIZERS[TOK](s)]
    m = Smoothed(train, 5, TOK, method="mkn")
    pp, bpc, _ = m.evaluate(dev)
    known_pp = m.evaluate(dev, known_only=True)[0]
    return {"sentences": len(train), "tokens": len(tokens), "types": len(counts),
            "types_in_first_million_tokens": len(first), "type_token_ratio_first_million": len(first) / BUDGET,
            "seen_once_share": sum(1 for c in counts.values() if c == 1) / len(counts),
            "avg_word_length": sum(len(w) * c for w, c in counts.items()) / len(tokens),
            "dev_oov": sum(w not in counts for w in dev_tokens) / len(dev_tokens),
            "perplexity": pp, "perplexity_known_only": known_pp, "bits_per_char": bpc}


def ladder(lang):
    """The Stage 3 comparison on this language: every smoothing method at order 5, and Kneser-Ney at orders 2-5."""
    train = (SPLIT / f"train.{lang}.txt").read_text(encoding="utf-8").splitlines()
    dev = (SPLIT / f"dev.{lang}.txt").read_text(encoding="utf-8").splitlines()
    m = Smoothed(train, 5, TOK, method="mkn", k=1e-5, lam=0.45, discount=0.8)
    out = {"methods": {}, "kn_by_order": {}}
    for method in ("add_k", "gt", "jm", "katz", "kn", "mkn"):
        m.method = method
        out["methods"][method] = {"pp": m.evaluate(dev)[0], "pp_known": m.evaluate(dev, known_only=True)[0]}
    m.method = "mkn"
    for order in (2, 3, 4, 5):
        m.set_order(order)
        out["kn_by_order"][order] = m.evaluate(dev)[0]
    return out


if __name__ == "__main__":
    out = {lang: profile(lang) for lang in ("ewe", "en")}
    for lang in ("ewe", "en"):
        out[lang]["ladder"] = ladder(lang)
        print(lang, "ladder:", {k: round(v["pp"], 1) for k, v in out[lang]["ladder"]["methods"].items()}, "| KN by order:", {k: round(v, 1) for k, v in out[lang]["ladder"]["kn_by_order"].items()})
    RESULTS.write_text(json.dumps(out, indent=1))
    for key in out["ewe"]:
        if key != "ladder":
            print(f"{key:<34} Ewe {out['ewe'][key]:>14,.4f}   English {out['en'][key]:>14,.4f}")
