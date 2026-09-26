"""Stage 0 helpers: normalise, Ewe filter, near-duplicate buckets, split writing. The flow itself lives in main.ipynb."""
import hashlib
import re
import unicodedata
from pathlib import Path

SRC = Path("data/train-00000-of-00001.parquet")
OUT = Path("data/splits")
REPORT = Path("results/stage0_audit.md")
# (train, dev, test) percentages. Each ratio gets its own folder: data/splits/80-10-10/ etc.
SPLITS = [(1, 49, 50), (5, 45, 50), (10, 40, 50), (10, 45, 45), (20, 40, 40), (30, 35, 35), (40, 30, 30),
          (50, 25, 25), (60, 20, 20), (70, 15, 15), (80, 10, 10), (85, 10, 5), (85, 5, 10), (90, 5, 5), (98, 1, 1)]

EWE_LETTERS = set("ɖƒŋɔɛʋɣƉƑŊƆƐƲƔ")
ETH = re.compile("Ð(?=[a-zɔɛ])")  # 'Ð' (U+00D0) typed as a look-alike for Ewe 'Ɖ' (U+0189): "Ðe", "Ðasefowo"


def nfc(s):
    return unicodedata.normalize("NFC", s)


def clean(s):
    return " ".join(s.split())  # strip + collapse all whitespace (also removes stray \t \n)


def fix_eth(s):
    return ETH.sub("Ɖ", s)


def is_ewe(s):
    if "Ð" in s:  # any 'Ð' left after fix_eth is UTF-8 Cyrillic mojibake ("Ð Ð¡Ð¢Ð¬")
        return False
    return any(c in EWE_LETTERS or 0x300 <= ord(c) <= 0x36F for c in s)


def near_dup_key(s):
    # Near-duplicates (case / digits / punctuation differences, e.g. verse numbers) share this key.
    # ponytail: exact-key grouping only; paraphrased near-dups can still leak across splits, MinHash if that matters.
    return " ".join(re.sub(r"[\W\d_]+", " ", s.lower()).split())


def bucket(s):
    # Hash the near-dup key to a bucket 0-99 so the whole group always lands in the same split.
    return int.from_bytes(hashlib.md5(near_dup_key(s).encode()).digest()[:8]) % 100


def vocab(lines):
    return len({tok for s in lines for tok in s.split()})


def write_splits(kept):
    """kept = {ewe_sentence: english_sentence}. Writes every ratio in SPLITS; returns a markdown table of sizes."""
    items = sorted(kept.items(), key=lambda p: hashlib.md5(p[0].encode()).digest())  # fixed pseudo-random order
    buckets = [bucket(s) for s, _ in items]
    md = "| Ratio | Train | Dev | Test |\n|---|---:|---:|---:|\n"
    for tr, dv, ts in SPLITS:
        folder = OUT / f"{tr}-{dv}-{ts}"
        folder.mkdir(parents=True, exist_ok=True)
        md += f"| {tr}/{dv}/{ts} |"
        for name, lo, hi in (("train", ts + dv, 100), ("dev", ts, ts + dv), ("test", 0, ts)):
            pairs = [p for p, b in zip(items, buckets) if lo <= b < hi]
            for lang, i in (("ewe", 0), ("en", 1)):
                (folder / f"{name}.{lang}.txt").write_text("".join(p[i] + "\n" for p in pairs), encoding="utf-8")
            md += f" {len(pairs):,} sent / {sum(len(p[0].split()) for p in pairs):,} tok |"
        md += "\n"
    return md


def selfcheck():
    assert fix_eth(clean("  Ðe  nèle\tgbɔgbɔmenu ")) == "Ɖe nèle gbɔgbɔmenu"
    assert is_ewe("Ɖe nèle gbɔgbɔmenu") and not is_ewe("Late Night Homework")
    assert not is_ewe("Ð Ð¡Ð¢Ð¬ ɖ")  # mojibake rejected even if it contains an Ewe letter
    assert bucket("6 Eya ta!") == bucket("eya ta")  # verse number / case / punctuation don't split a group
    assert all(sum(r) == 100 for r in SPLITS)
