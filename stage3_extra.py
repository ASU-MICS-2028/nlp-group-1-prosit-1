"""Two Stage 3 follow-ups.

A. A fair unknown-word charge. A word-level model pays one flat cost for an unknown word; a BPE or character model
   must spell it. To compare fairly, charge the word model the cost of SPELLING each unknown word with a character
   model (order 8, trained on the same data), on top of its "this word is new" cost. Reports bits/char before and after.
B. Push the small tokenisers to higher orders than Stage 3.7 did (bpe1k to 14, bpe2k to 12, char to 16),
   because all three were still improving at their highest tested order.

Run `uv run python stage3_extra.py`; results in results/stage3_extra.json and .md. Memory-hungry: run alone.
"""
import json
import math
import time
from pathlib import Path

import bpe
from stage2_ngram import TOKENIZERS, pad
from stage3_smoothing import Smoothed

RESULTS = Path("results/stage3_extra.json")
SPLIT = Path("data/splits/90-5-5")
WORD_MODELS = (("whitespace", 6), ("lower_punct", 6), ("punct", 6))
BPE_REF = (("bpe2k", 8), ("bpe4k", 8))
HIGHER = {"bpe1k": (12, 14), "bpe2k": (10, 12), "char": (14, 16)}


def fair_bits(word_model, speller, sentences):
    """Bits over the dev set where each unknown word additionally costs its spelling under the character model."""
    tokenize = TOKENIZERS[word_model.tokenizer]
    known = word_model.models[1].counts[()]
    bits_flat = bits_fair = 0.0
    n_unknown = 0
    for s in sentences:
        toks = pad(tokenize(s), word_model.n)
        for i in range(word_model.n - 1, len(toks)):
            p = word_model.prob(toks[i - word_model.n + 1:i], toks[i])
            bits_flat -= math.log2(p)
            bits_fair -= math.log2(p)
            if toks[i] not in known:
                n_unknown += 1
                pp, _, n = speller.evaluate([toks[i]])  # cost of spelling the word character by character (+ its end)
                bits_fair += math.log2(pp) * n
    n_chars = sum(len(s) for s in sentences)
    return bits_flat / n_chars, bits_fair / n_chars, n_unknown


def main():
    train = (SPLIT / "train.ewe.txt").read_text(encoding="utf-8").splitlines()
    dev = (SPLIT / "dev.ewe.txt").read_text(encoding="utf-8").splitlines()
    bpe.register(train)
    out = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"fair_unknown": {}, "higher_orders": {}}

    # ---- A ----
    if not out["fair_unknown"]:
        t0 = time.time()
        speller = Smoothed(train, 8, "char", method="mkn")
        print(f"character speller built ({time.time() - t0:.0f}s)", flush=True)
        for name, order in WORD_MODELS:
            m = Smoothed(train, order, name, method="mkn")
            flat, fair, n_unk = fair_bits(m, speller, dev)
            out["fair_unknown"][name] = {"order": order, "bpc_flat": flat, "bpc_fair": fair, "unknown_tokens": n_unk}
            print(f"A. {name:<12} order {order}: bits/char {flat:.3f} with a flat unknown cost -> {fair:.3f} when unknown words must be spelled ({n_unk:,} unknown tokens)", flush=True)
            del m
        del speller
        for name, order in BPE_REF:
            m = Smoothed(train, order, name, method="mkn")
            bpc = m.evaluate(dev)[1]
            out["fair_unknown"][name] = {"order": order, "bpc_flat": bpc, "bpc_fair": bpc, "unknown_tokens": 0}
            print(f"A. {name:<12} order {order}: bits/char {bpc:.3f} (no unknown tokens, nothing to add)", flush=True)
            del m
        RESULTS.write_text(json.dumps(out, indent=1))

    # ---- B ----
    for name, (lo, hi) in HIGHER.items():
        if name in out["higher_orders"]:
            continue
        t0 = time.time()
        m = Smoothed(train, hi, name, method="mkn")
        scores = {}
        for order in range(lo, hi + 1, 2):
            m.set_order(order)
            scores[order] = m.evaluate(dev)[1]
            print(f"B. {name:<6} order {order:>2}: bits/char {scores[order]:.3f}", flush=True)
        out["higher_orders"][name] = scores
        del m
        RESULTS.write_text(json.dumps(out, indent=1))
        print(f"   ({time.time() - t0:.0f}s)", flush=True)

    md = "# Stage 3 extras\n\n## A. Charging word-level models for spelling unknown words\n\n| tokeniser | order | bits/char, flat unknown cost | bits/char, unknown words spelled | unknown tokens |\n|---|---:|---:|---:|---:|\n"
    md += "".join(f"| {k} | {v['order']} | {v['bpc_flat']:.3f} | {v['bpc_fair']:.3f} | {v['unknown_tokens']:,} |\n" for k, v in out["fair_unknown"].items())
    md += "\n## B. Small tokenisers at higher orders (bits/char)\n\n| tokeniser | " + " | ".join(f"order {o}" for o in range(8, 17, 2)) + " |\n|---|" + "---:|" * 5 + "\n"
    prev = {"bpe1k": {8: 1.521, 10: 1.509}, "bpe2k": {8: 1.496}, "char": {8: 1.708, 10: 1.629, 12: 1.599}}  # from Stage 3.7
    for name, scores in out["higher_orders"].items():
        merged = {**prev.get(name, {}), **{int(k): v for k, v in scores.items()}}
        md += f"| {name} | " + " | ".join(f"{merged[o]:.3f}" if o in merged else "" for o in range(8, 17, 2)) + " |\n"
    Path("results/stage3_extra.md").write_text(md, encoding="utf-8")
    print("wrote results/stage3_extra.md")


if __name__ == "__main__":
    main()
