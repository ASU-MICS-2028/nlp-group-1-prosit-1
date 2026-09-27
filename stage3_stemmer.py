"""Does splitting Ewe affixes off words help? The rule-based stemmer (ewe_stemmer.py) against lower_punct, which is the
same tokenization without the affix splits. Modified Kneser-Ney on the 90/5/5 split, dev set, bits per character,
with unknown words charged the cost of spelling them (the fair comparison from stage3_extra.py).

Run `uv run python stage3_stemmer.py`; writes results/stemmer.json. About 5 minutes.
"""
import json
from pathlib import Path

import ewe_stemmer
from stage3_extra import fair_bits
from stage3_smoothing import Smoothed
from stage2_ngram import TOKENIZERS

SPLIT = Path("data/splits/90-5-5")
train = (SPLIT / "train.ewe.txt").read_text(encoding="utf-8").splitlines()
dev = (SPLIT / "dev.ewe.txt").read_text(encoding="utf-8").splitlines()
ewe_stemmer.register()

speller = Smoothed(train, 8, "char", method="mkn")  # spells out unknown words, character by character
out = {}
for name in ("lower_punct", "ewe_stem"):
    m = Smoothed(train, 6, name, method="mkn")
    rows = {}
    for order in (4, 5, 6):
        m.set_order(order)
        flat, fair, n_unk = fair_bits(m, speller, dev)
        rows[order] = {"bpc_flat": flat, "bpc_fair": fair, "unknown_tokens": n_unk}
        print(f"{name:<12} order {order}: bits/char {flat:.3f} flat unknown cost, {fair:.3f} unknown words spelled ({n_unk:,} unknown)", flush=True)
    vocab = m.vocab_size - 1
    toks = sum(len(TOKENIZERS[name](s)) for s in dev) / len(dev)
    out[name] = {"orders": rows, "vocab": vocab, "tokens_per_sentence": toks}
    del m
Path("results/stemmer.json").write_text(json.dumps(out, indent=1))
