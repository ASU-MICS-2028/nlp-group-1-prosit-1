"""
Rebuilds every Ewe split used in Section B from the raw files in data/raw/low_resource/
(see data/README.md for where each raw file comes from):

    data/processed/dataset_1_csv, dataset_2_json, dataset_3_speech, dataset_4_parquet   one source each
    data/processed/unified                                                              all four, cross-deduplicated

Each folder gets train/val/test.txt (80/10/10 after a seed-42 shuffle) and stats.json.
Cleaning, deduplication and splitting all live in src/section_b_ngram/data_pipeline.py.

Run from the repo root:  python -m src.section_b_ngram.build_datasets
"""

import json
import re
from pathlib import Path

import pandas as pd

from src import ROOT
from src.section_b_ngram.data_pipeline import harmonize_sentences

RANDOM_SEED = 42
RAW = ROOT / "data" / "raw" / "low_resource"
OUT = ROOT / "data" / "processed"
# ponytail: the parquet holds 4.4M rows sorted by alignment score; we keep the first 200k (largely Bible
# and Jehovah's Witnesses text). Switch to a seeded random sample if that skew matters more than continuity.
D4_ROWS = 200_000


def read_sources() -> dict:
    d2 = json.loads((RAW / "dataset_2_json" / "eweenglishsentence(3).json").read_text(encoding="utf-8"))
    d2_rows = next(t for t in d2 if t.get("type") == "table")["data"]  # PHPMyAdmin export
    return {
        "dataset_1_csv": pd.read_csv(RAW / "dataset_1_csv" / "EWE_ENGLISH.csv")["EWE"].dropna().tolist(),
        "dataset_2_json": [row["ee_sentence"] for row in d2_rows],
        "dataset_3_speech": pd.read_excel(RAW / "dataset_3_speech" / "selected transcribed audios.xlsx")[
            "Transcription"
        ]
        .dropna()
        .astype(str)
        .tolist(),
        "dataset_4_parquet": pd.read_parquet(RAW / "dataset_4_parquet" / "ewe_corpus.parquet", columns=["Ewe"])["Ewe"]
        .head(D4_ROWS)
        .tolist(),
    }


def religious_markers(folder: Path) -> dict:
    """Share of kept sentences that mention Yehowa or carry a chapter:verse reference (evidence for the datasheet)."""
    lines = [line.lower() for s in ("train", "val", "test") for line in (folder / f"{s}.txt").read_text(encoding="utf-8").splitlines()]
    return {
        "pct_mentioning_yehowa": round(100 * sum("yehowa" in line for line in lines) / len(lines), 1),
        "pct_with_chapter_verse": round(100 * sum(bool(re.search(r"\b\d+:\d+\b", line)) for line in lines) / len(lines), 1),
    }


def build(name: str, sources: dict) -> None:
    summary = harmonize_sentences(sources, OUT / name, seed=RANDOM_SEED)
    summary.update(religious_markers(OUT / name))
    (OUT / name / "stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sources = read_sources()
    for name, lines in sources.items():
        build(name, {name: lines})
    build("unified", sources)
