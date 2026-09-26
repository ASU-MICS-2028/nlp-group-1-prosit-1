"""English as a HIGH-resource language: WikiText-103 (103M words of Wikipedia), the same recipe as Ewe, at sizes far
beyond what Ewe has.

Statistical: modified Kneser-Ney order 5 at 10k ... 3M sentences. Our own counter handles up to 300k sentences;
above that KenLM (verified against our code in Stage 3.5b) does the counting, because pure Python cannot hold
100M tokens. Neural: LSTM and transformer at 1k ... 100k, the same sizes as Ewe, via stage4_neural's recipe.

Run `uv run python stage4_wikitext.py prepare | ngram | neural`. Results in results/wikitext.json.
"""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from stage3_smoothing import Smoothed

ROOT = Path("data/wikitext")
RESULTS = Path("results/wikitext.json")
KENLM = Path(".tools/kenlm/build/bin")
NGRAM_SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
OWN_CODE_MAX = 300_000  # our Python counter's comfortable limit at order 5
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def prepare():
    """Paragraphs -> sentences, dedupe, split 90/5/5 by hash, write train/dev/test.txt (whitespace tokens, as Ewe)."""
    sents = []
    for f in sorted(ROOT.glob("train-*.parquet")):
        for para in pq.read_table(f)["text"].to_pylist():
            para = para.strip()
            if not para or para.startswith("="):  # headings
                continue
            for s in SENT.split(para):
                s = " ".join(s.split())
                if 3 <= len(s.split()) <= 80:
                    sents.append(s)
    sents = list(dict.fromkeys(sents))
    parts = {"train": [], "dev": [], "test": []}
    for s in sents:
        b = int.from_bytes(hashlib.md5(s.encode()).digest()[:8]) % 100
        parts["test" if b < 5 else "dev" if b < 10 else "train"].append(s)
    for name, lines in parts.items():
        lines.sort(key=lambda s: hashlib.md5(s.encode()).digest())  # fixed pseudo-random order, so "first N" is fair
        (ROOT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    info = {k: len(v) for k, v in parts.items()}
    info["train_words"] = sum(len(s.split()) for s in parts["train"])
    print(info)
    return info


def load(name, n=None):
    lines = (ROOT / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    return lines[:n] if n else lines


def kenlm_ppl(train_lines, dev_path, order=5):
    tmp = Path(".tools/tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    train_path, arpa = tmp / "wt_train.txt", tmp / "wt.arpa"
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    subprocess.run([str(KENLM / "lmplz"), "-o", str(order), "-S", "6G", "-T", str(tmp), "--discount_fallback", "--text", str(train_path), "--arpa", str(arpa)],
                   check=True, capture_output=True)
    with open(dev_path, "rb") as f:
        out = subprocess.run([str(KENLM / "query"), "-v", "summary", str(arpa)], stdin=f, capture_output=True).stdout.decode()
    ppl_all = float(re.search(r"including OOVs:\s*([\d.]+)", out).group(1))
    ppl_known = float(re.search(r"excluding OOVs:\s*([\d.]+)", out).group(1))
    oov = int(re.search(r"^OOVs:\s*(\d+)", out, re.M).group(1))
    tokens = int(re.search(r"^Tokens:\s*(\d+)", out, re.M).group(1))
    arpa.unlink()
    return {"kenlm_pp": ppl_all, "kenlm_pp_known": ppl_known, "oov_rate": oov / tokens}


def ngram():
    """The statistical scaling curve, far past Ewe's size."""
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    dev = load("dev", 20_000)  # a 20k-sentence dev slice: plenty (Stage 0 showed 10k pins the score within 3 points)
    dev_path = ROOT / "dev20k.txt"
    dev_path.write_text("\n".join(dev) + "\n", encoding="utf-8")
    train_all = load("train")
    results.setdefault("ngram", {})
    for size in NGRAM_SIZES:
        if str(size) in results["ngram"] or size > len(train_all):
            continue
        t0 = time.time()
        subset = train_all[:size]
        row = kenlm_ppl(subset, dev_path)
        row["kenlm_seconds"] = round(time.time() - t0)
        if size <= OWN_CODE_MAX:
            t1 = time.time()
            m = Smoothed(subset, 5, "whitespace", method="mkn", unk="floor")
            row["own_pp"] = m.evaluate(dev)[0]
            row["own_pp_known"] = m.evaluate(dev, known_only=True)[0]
            row["own_bpc"] = m.evaluate(dev)[1]
            row["own_seconds"] = round(time.time() - t1)
            del m
        results["ngram"][str(size)] = row
        RESULTS.write_text(json.dumps(results, indent=1))
        print(f"{size:>9,} sentences | KenLM perplexity {row['kenlm_pp']:>7.1f} (known {row['kenlm_pp_known']:>6.1f}, OOV {row['oov_rate']:.2%}) [{row['kenlm_seconds']}s]"
              + (f" | our code {row['own_pp']:>7.1f} [{row['own_seconds']}s]" if "own_pp" in row else ""), flush=True)


def neural():
    """LSTM and transformer at the Ewe sizes, on English, same recipe (stage4_neural)."""
    import stage4_neural as s4
    from tokenizers import Tokenizer
    from bpe import train_bpe
    import stage2_ngram
    train_s, dev_s = load("train"), load("dev", 20_000)
    path = ROOT / "bpe_4000.json"
    if not path.exists():
        train_bpe(train_s[:300_000], 4000, path)
    tok = Tokenizer.from_file(str(path))
    stage2_ngram.TOKENIZERS["bpe4k_wt"] = lambda x, t=tok: t.encode(x).tokens
    enc = s4.Encoder(str(path))
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results.setdefault("neural", {})
    for size in s4.SIZES:
        row = results["neural"].setdefault(str(size), {})
        subset = train_s[:size]
        if "ngram" not in row:
            row["ngram"], row["tokens"] = s4.best_ngram(subset, dev_s, "bpe4k_wt")
            print(f"WT {size:>7,} | n-gram (order {row['ngram']['order']}): perplexity {row['ngram']['pp']:,.1f}", flush=True)
        for arch in s4.ARCHS:
            if arch not in row:
                row[arch], tokens = s4.fit_and_score(arch, subset, dev_s, enc, log=lambda *_: None)
                assert tokens == row["tokens"]
                print(f"{'':>11}| {arch:<11} perplexity {row[arch]['pp']:,.1f} ({row[arch]['epochs']} epochs, {row[arch]['seconds']}s)", flush=True)
            RESULTS.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "ngram": ngram, "neural": neural}[sys.argv[1]]()
