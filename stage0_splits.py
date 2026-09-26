"""Does the train/dev/test ratio matter? Train the same model on every ratio Stage 0 wrote, score it three ways.

1. On the ratio's own dev and test (its own exam, so scores are NOT comparable across ratios).
2. On one common exam: the 98/1/1 test set. Its sentences sit in hash bucket 0, which every ratio puts in test,
   so no ratio ever trained on them. This is the fair comparison.
3. How much a dev estimate wobbles with dev size: random dev subsets of several sizes, several draws each.

Run `uv run python stage0_splits.py`; results in results/split_ratios.json. ~15 minutes.
"""
import json
import random
import time
from pathlib import Path

from stage3_smoothing import Smoothed

SPLITS = Path("data/splits")
RESULTS = Path("results/split_ratios.json")
ORDER, TOK = 5, "whitespace"


def read(folder, name):
    return (SPLITS / folder / f"{name}.ewe.txt").read_text(encoding="utf-8").splitlines()


def main():
    common = read("98-1-1", "test")
    folders = sorted((p.name for p in SPLITS.iterdir() if p.is_dir()), key=lambda f: int(f.split("-")[0]))
    out = {"order": ORDER, "tokenizer": TOK, "common_test": {"source": "98-1-1 test", "sentences": len(common)}, "ratios": {}, "dev_size": {}}
    for f in folders:
        t0 = time.time()
        train, dev, test = read(f, "train"), read(f, "dev"), read(f, "test")
        m = Smoothed(train, ORDER, TOK, method="mkn")
        known = m.models[1].counts[()]
        row = {"train": len(train), "dev": len(dev), "test": len(test)}
        for name, data in (("dev", dev), ("test", test), ("common", common)):
            pp, bpc, n = m.evaluate(data)
            toks = [w for s in data for w in s.split()]
            row[name + "_pp"], row[name + "_bpc"], row[name + "_oov"] = pp, bpc, sum(w not in known for w in toks) / len(toks)
        if f == "90-5-5":  # 3. dev-size wobble, on the main split's model
            rng = random.Random(0)
            for size in (100, 300, 1000, 3000, 10000):
                draws = [m.evaluate(rng.sample(dev, size))[0] for _ in range(8)]
                out["dev_size"][str(size)] = {"min": min(draws), "max": max(draws), "mean": sum(draws) / len(draws)}
        del m
        out["ratios"][f] = row
        print(f"{f:>8}: train {len(train):>7,} | own dev {row['dev_pp']:>6.1f} | own test {row['test_pp']:>6.1f} | common test {row['common_pp']:>6.1f} | OOV on common {row['common_oov']:.2%} | {time.time() - t0:.0f}s", flush=True)
        RESULTS.write_text(json.dumps(out, indent=1))
    print("\ndev-size wobble (90-5-5 model, 8 random draws per size):")
    for size, d in out["dev_size"].items():
        print(f"  {int(size):>6,} sentences: perplexity {d['min']:.1f} to {d['max']:.1f}")


if __name__ == "__main__":
    main()
