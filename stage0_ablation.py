"""Leave one preparation step out at a time and measure what happens (an "ablation").

Baseline = the Stage 0 pipeline: NFC -> whitespace cleanup -> Ð→Ɖ -> dedupe -> Ewe filter -> split by near-duplicate group.
Each variant removes exactly one step, rebuilds train/dev with the same hash buckets (so dev stays the same exam
wherever the step does not change which sentences exist), and trains modified Kneser-Ney order 5 on whitespace tokens.

Run `uv run python stage0_ablation.py`; results in results/ablation.json and .md. ~5 minutes.
"""
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from stage0_audit import SRC, bucket, clean, fix_eth, is_ewe, near_dup_key, nfc
from stage3_smoothing import Smoothed

RESULTS = Path("results/ablation.json")


def pipeline(rows, do_nfc=True, do_clean=True, do_eth=True, do_dedupe=True, do_filter=True):
    out = []
    for s in rows:
        if do_nfc:
            s = nfc(s)
        if do_clean:
            s = clean(s)
        if do_eth:
            s = fix_eth(s)
        if s and (not do_filter or is_ewe(s)):
            out.append(s)
    return list(dict.fromkeys(out)) if do_dedupe else out


def split_by_group(sents):
    train = [s for s in sents if bucket(s) >= 10]
    dev = [s for s in sents if 5 <= bucket(s) < 10]
    return train, dev


def score(train, dev):
    m = Smoothed(train, 5, "whitespace", method="mkn")
    pp, bpc, _ = m.evaluate(dev)
    known = m.evaluate(dev, known_only=True)[0]
    return {"train": len(train), "dev": len(dev), "vocab": m.vocab_size - 1, "dev_pp": pp, "dev_pp_known": known, "dev_bpc": bpc}


def main():
    raw = pq.read_table(SRC, columns=["Ewe"])["Ewe"].to_pylist()
    base_sents = pipeline(raw)
    base_train, base_dev = split_by_group(base_sents)
    results = {}

    def run(name, train, dev, note):
        r = score(train, dev)
        r["note"] = note
        r["same_dev_as_baseline"] = dev == base_dev
        results[name] = r
        print(f"{name:<28} train {r['train']:>9,} | vocab {r['vocab']:>9,} | dev ppl {r['dev_pp']:>7.1f} | known {r['dev_pp_known']:>7.1f} | bits/char {r['dev_bpc']:.3f}"
              f"{'' if r['same_dev_as_baseline'] else '  (different dev set)'}", flush=True)

    run("baseline (all steps)", base_train, base_dev, "the Stage 0 pipeline")

    s = pipeline(raw, do_nfc=False); t, d = split_by_group(s)
    run("without NFC", t, d, "NFC changed 0 rows, so this is identical")

    s = pipeline(raw, do_clean=False); t, d = split_by_group(s)
    run("without whitespace cleanup", t, d, "extra/leading spaces kept; ' s' and 's' no longer merge in dedupe or the split key")

    s = pipeline(raw, do_eth=False); t, d = split_by_group(s)
    run("without the Ð→Ɖ fix", t, d, "Ð-spelled words stay separate vocabulary entries")

    s = pipeline(raw, do_dedupe=False); t, d = split_by_group(s)
    d = list(dict.fromkeys(d))  # the exam stays unique sentences; the training set keeps every copy
    run("without dedupe", t, d, "training keeps all copies (a hub sentence 1,000+ times); dev is deduped so the exam is the same")

    # random split instead of near-duplicate groups: same sentences, different split
    rng = random.Random(0)
    shuffled = base_sents[:]
    rng.shuffle(shuffled)
    n_dev = len(base_dev)
    t, d = shuffled[n_dev:], shuffled[:n_dev]
    train_keys = {near_dup_key(x) for x in t}
    leaked = sum(near_dup_key(x) in train_keys for x in d)
    run("random split (leakage)", t, d, f"{leaked:,} of {n_dev:,} dev sentences ({leaked / n_dev:.1%}) have a near-copy in train")
    results["random split (leakage)"]["leaked"] = leaked

    RESULTS.write_text(json.dumps(results, indent=1))
    md = "# Ablation: leave one preparation step out\n\nModified Kneser-Ney, order 5, whitespace tokens. Dev perplexity (lower is better).\n\n"
    md += "| variant | train sentences | vocabulary | dev perplexity | known words only | bits/char | note |\n|---|---:|---:|---:|---:|---:|---|\n"
    md += "".join(f"| {k} | {v['train']:,} | {v['vocab']:,} | {v['dev_pp']:.1f} | {v['dev_pp_known']:.1f} | {v['dev_bpc']:.3f} | {v['note']}{'' if v['same_dev_as_baseline'] else ' (different dev set)'} |\n" for k, v in results.items())
    md += "\nAlso measured elsewhere: the Ewe filter (results/filter_test.md: no filter = 25% worse) and smoothing (results/dev_and_smoothing_checks.json: plain counting = infinite perplexity on 81% of dev sentences).\n"
    Path("results/ablation.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
