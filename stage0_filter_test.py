"""Three tests of the data-cleaning choices (Stage 0 follow-ups).

A. Religious-text skew: how much of the training data is Bible / church text? (counted, not guessed)
B. A real language identifier (GlotLID, 1,600+ languages incl. Ewe, Ga, Fon, Twi) against our letter filter:
   what does it say about the sentences we kept, and about the ones we threw away?
C. Filter on vs off: train the same model on the UNFILTERED unique sentences (same hash buckets as the 90/5/5
   train split, so nothing leaks into dev) and score it on the filtered dev set. Does contamination hurt?

Run `uv run python stage0_filter_test.py`; results in results/filter_test.json and .md.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from stage0_audit import SRC, bucket, clean, fix_eth, is_ewe, nfc
from stage3_smoothing import Smoothed

RESULTS = Path("results/filter_test.json")
LID = Path("data/models/glotlid.bin")
ORDER = 3  # the unfiltered corpus is 3.4x bigger; order 3 keeps memory sane. Both models use the same order.
RELIGIOUS = re.compile(r"\b(Yehowa|Mawu|Yesu|Kristo|Biblia|Israel|Mose|Paulo|Ðasefo|Ɖasefo|gbedoxɔ|nyagblɔɖila|Aƒetɔ)\b|^\d+ ")


def main():
    out = {}
    split = Path("data/splits/90-5-5")
    train = (split / "train.ewe.txt").read_text(encoding="utf-8").splitlines()
    dev = (split / "dev.ewe.txt").read_text(encoding="utf-8").splitlines()

    # ---- A. religious skew ----
    hits = sum(bool(RELIGIOUS.search(s)) for s in train)
    verse = sum(bool(re.match(r"^\d+ ", s)) for s in train)
    out["religious"] = {"train_sentences": len(train), "religious_markers": hits, "share": hits / len(train),
                        "verse_number_prefix": verse, "verse_share": verse / len(train),
                        "note": "sentences containing a Bible/church keyword (Yehowa, Mawu, Yesu, Kristo, Biblia, Israel, Mose, Paulo, Ɖasefo, ...) or starting with a verse number; a lower bound"}
    print(f"A. religious markers in {hits / len(train):.1%} of training sentences ({verse / len(train):.1%} start with a verse number)", flush=True)

    # ---- the unique, normalised sentences, as Stage 0 saw them before the filter ----
    ee = pq.read_table(SRC, columns=["Ewe"])["Ewe"].to_pylist()
    uniq = list(dict.fromkeys(fix_eth(clean(nfc(s))) for s in ee))
    uniq = [s for s in uniq if s]
    del ee

    # ---- B. GlotLID vs the letter filter ----
    if LID.exists():
        import fasttext
        m = fasttext.load_model(str(LID))
        lid = lambda s: m.f.predict(s.replace("\n", " "), 1, 0.0, "strict")[0]
        verdict = {}
        for s in uniq:
            p, label = lid(s)
            verdict[s] = (label[9:], p)
        kept = [s for s in uniq if is_ewe(s)]
        rejected = [s for s in uniq if not is_ewe(s)]
        top_kept = Counter(verdict[s][0] for s in kept).most_common(8)
        top_rej = Counter(verdict[s][0] for s in rejected).most_common(8)
        ewe_kept = sum(verdict[s][0] == "ewe_Latn" for s in kept)
        ewe_rej = sum(verdict[s][0] == "ewe_Latn" for s in rejected)
        dev_ewe = sum(verdict.get(s, ("?", 0))[0] == "ewe_Latn" for s in dev)
        rng = random.Random(0)
        out["glotlid"] = {
            "kept": len(kept), "kept_called_ewe": ewe_kept, "kept_called_ewe_share": ewe_kept / len(kept),
            "rejected": len(rejected), "rejected_called_ewe": ewe_rej, "rejected_called_ewe_share": ewe_rej / len(rejected),
            "dev_called_ewe_share": dev_ewe / len(dev),
            "top_languages_in_kept": top_kept, "top_languages_in_rejected": top_rej,
            "examples_kept_not_ewe": [(s[:90], verdict[s][0], round(verdict[s][1], 2)) for s in rng.sample([s for s in kept if verdict[s][0] != "ewe_Latn"], 12)],
            "examples_rejected_but_ewe": [(s[:90], round(verdict[s][1], 2)) for s in rng.sample([s for s in rejected if verdict[s][0] == "ewe_Latn"], 12)],
        }
        print(f"B. GlotLID calls {ewe_kept / len(kept):.1%} of our KEPT sentences Ewe, and {ewe_rej / len(rejected):.1%} of our REJECTED ones Ewe", flush=True)
        print("   top languages among kept:", top_kept[:5])
        print("   top languages among rejected:", top_rej[:5])
        # a third training set: GlotLID-filtered, same buckets as train
        glot_train = [s for s in uniq if verdict[s][0] == "ewe_Latn" and bucket(s) >= 10]
    else:
        print("B. skipped: GlotLID model not present")
        glot_train = None

    # ---- C. filter on vs off (and GlotLID filter), same exam: the filtered dev set ----
    unfiltered_train = [s for s in uniq if bucket(s) >= 10]  # everything that would land in train, Ewe or not
    rows = {}
    for name, corpus in (("letter filter (ours)", train), ("no filter", unfiltered_train)) + ((("GlotLID filter", glot_train),) if glot_train else ()):
        mdl = Smoothed(corpus, ORDER, "whitespace", method="mkn")
        pp, bpc, _ = mdl.evaluate(dev)
        known = mdl.evaluate(dev, known_only=True)[0]
        vocab = mdl.vocab_size - 1
        rows[name] = {"train_sentences": len(corpus), "vocab": vocab, "dev_pp": pp, "dev_pp_known": known, "dev_bpc": bpc, "build_note": f"order {ORDER}"}
        print(f"C. {name:<22} train {len(corpus):>9,} | vocab {vocab:>9,} | dev perplexity {pp:>7.1f} | known-only {known:>7.1f} | bits/char {bpc:.3f}", flush=True)
        del mdl
    out["filter_on_off"] = rows

    RESULTS.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    md = "# Filter tests\n\n"
    md += f"## A. Religious-text skew\n\n{out['religious']['share']:.1%} of training sentences contain a Bible/church keyword or start with a verse number ({out['religious']['verse_share']:.1%} start with a verse number). This is a lower bound.\n\n"
    if "glotlid" in out:
        g = out["glotlid"]
        md += f"## B. GlotLID (a real language identifier) vs our letter filter\n\n- Of the {g['kept']:,} sentences our filter KEPT, GlotLID calls {g['kept_called_ewe_share']:.1%} Ewe.\n- Of the {g['rejected']:,} it REJECTED, GlotLID calls {g['rejected_called_ewe_share']:.1%} Ewe.\n- Of our dev set, GlotLID calls {g['dev_called_ewe_share']:.1%} Ewe.\n\n"
        md += "Top languages among kept: " + ", ".join(f"{l} {n:,}" for l, n in g["top_languages_in_kept"]) + "\n\nTop languages among rejected: " + ", ".join(f"{l} {n:,}" for l, n in g["top_languages_in_rejected"]) + "\n\n"
        md += "Kept by us, not Ewe per GlotLID (sample):\n\n" + "".join(f"- {s} — {l} ({p})\n" for s, l, p in g["examples_kept_not_ewe"]) + "\n"
        md += "Rejected by us, Ewe per GlotLID (sample):\n\n" + "".join(f"- {s} ({p})\n" for s, p in g["examples_rejected_but_ewe"]) + "\n"
    md += f"## C. Filter on vs off (modified Kneser-Ney, order {ORDER}, scored on the filtered dev set)\n\n| training set | sentences | vocabulary | dev perplexity | known words only | bits/char |\n|---|---:|---:|---:|---:|---:|\n"
    md += "".join(f"| {k} | {v['train_sentences']:,} | {v['vocab']:,} | {v['dev_pp']:.1f} | {v['dev_pp_known']:.1f} | {v['dev_bpc']:.3f} |\n" for k, v in rows.items())
    Path("results/filter_test.md").write_text(md, encoding="utf-8")
    print("wrote results/filter_test.md")


if __name__ == "__main__":
    main()
