"""
Neural baseline for Section B, Question 2: a small LSTM trained on exactly the tokens the best n-gram uses
(BPE with 150 merges learned from the training split, tokens seen once -> <unk>), compared per word with
interpolated Kneser-Ney on the same test sentences. Three seeds by default.

Run from the repo root after the n-gram sweep (it reads results/section_b_ngram/):
    python -m src.section_b_lstm.run_baseline --dataset 2 --config small
Results are merged into results/section_b_lstm/lstm_vs_ngram.json (use --dry-run to only time an epoch).
"""

import argparse
import json
import math

from src import ROOT
from src.section_b_lstm.lstm_lm import build_index, encode, train_and_score
from src.section_b_ngram.ewe_tokenizers import SimpleBPETokenizer
from src.section_b_ngram.experiment_runner import count_words, oov_spelling_nats, tokenize_splits
from src.section_b_ngram.run_sweep import DATASETS, MAX_EVAL, RESULTS as NGRAM_RESULTS, read_lines

BPE = "Byte-Pair Encoding (BPE)"
CONFIGS = {
    "small": {"emb": 128, "hidden": 256, "layers": 1, "dropout": 0.3},
    "large": {"emb": 256, "hidden": 512, "layers": 2, "dropout": 0.3},
}
OUT = ROOT / "results" / "section_b_lstm" / "lstm_vs_ngram.json"


def run(key, config_name, seeds, max_epochs, patience, dry_run):
    folder, label, ngram_file = DATASETS[key]
    data = ROOT / "data" / "processed" / folder
    train, val, test = (read_lines(data / f"{s}.txt") for s in ("train", "val", "test"))
    val, test = val[:MAX_EVAL], test[:MAX_EVAL]

    bpe = SimpleBPETokenizer(num_merges=150)
    bpe.train(train)
    vocab, raw_train, raw_test, (clean_train, clean_val, clean_test) = tokenize_splits(train, val, test, bpe)
    stoi, blocked = build_index(vocab)
    tr, va, te = (encode(x, stoi) for x in (clean_train, clean_val, clean_test))
    spelling = oov_spelling_nats(raw_train, vocab, raw_test, clean_test)
    words = count_words(test)

    # The comparison only means something if both models score the same token stream
    ngram = json.loads((NGRAM_RESULTS / ngram_file).read_text())
    best = ngram["best_order_by_val"][BPE]
    kn = next(r for r in ngram["results"][BPE] if r["order"] == best["order"])
    assert kn["vocab_size"] == len(vocab), "vocabulary differs from the n-gram run"
    assert kn["test_tokens"] + len(te) == sum(len(s) - 1 for s in te), "token stream differs from the n-gram run"
    assert kn["oov_spelling_nats"] == round(spelling, 1), "spelling charge differs from the n-gram run"
    print(f"{label}: {len(tr)} train sentences, vocabulary {len(vocab)}, config {config_name}, seeds {seeds}", flush=True)

    runs = []
    for seed in seeds:
        r = train_and_score(tr, va, te, len(stoi), blocked, CONFIGS[config_name], seed, max_epochs, patience)
        r["per_word_perplexity"] = round(math.exp((r["test_nll"] + spelling) / words), 2)
        r.update(max_epochs=max_epochs, patience=patience)
        runs.append(r)
        print(f"  seed {seed}: best epoch {r['best_epoch']}, test PPL per token {r['test_perplexity']}, per word {r['per_word_perplexity']}", flush=True)
    if dry_run:
        return

    # Seeds can be run in separate invocations; keep earlier seeds of the same configuration
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    if results.get(key, {}).get("config", {}).get("name") == config_name:
        runs = [r for r in results[key]["runs"] if r["seed"] not in seeds] + runs
    runs.sort(key=lambda r: r["seed"])
    per_word = [r["per_word_perplexity"] for r in runs]
    entry = {
        "dataset": label,
        "tokens": "BPE (150 merges learned from the training split); tokens seen once in training -> <unk>",
        "scoring": "each sentence on its own from <s> (LSTM state reset), every token plus </s>; <pad> and <s> never predicted",
        "config": CONFIGS[config_name] | {"name": config_name, "optimizer": "Adam lr 0.002, grad clip 1.0", "batch_tokens": 4096},
        "early_stopping": "keep the epoch with the best validation perplexity; stop after `patience` epochs without improvement (per run: max_epochs, patience)",
        "params_note": "every trainable weight, including the full embedding table",
        "train_sentences": len(tr),
        "test_sentences_scored": len(te),
        "test_words": words,
        "oov_spelling_nats": round(spelling, 1),
        "runs": runs,
        "lstm_per_word_mean": round(sum(per_word) / len(per_word), 2),
        "lstm_per_word_min": min(per_word),
        "lstm_per_word_max": max(per_word),
        "kn_bpe": {"order": best["order"], "per_word_perplexity": kn["per_word_perplexity"], "test_perplexity": kn["perplexity"]},
    }
    results[key] = entry
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  LSTM per word {entry['lstm_per_word_mean']} (range {min(per_word)} to {max(per_word)}) vs Kneser-Ney {kn['per_word_perplexity']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--config", choices=list(CONFIGS), default="small")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="train and print, but do not write the results file")
    a = parser.parse_args()
    run(a.dataset, a.config, a.seeds, a.max_epochs, a.patience, a.dry_run)
