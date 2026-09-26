"""Stage 4: a small neural language model (2-layer LSTM) trained from scratch on the SAME BPE tokens as the n-gram,
and the scaling experiment: n-gram vs neural at growing amounts of training data.

Run `uv run python stage4_neural.py` to (re)compute results/stage4_scaling.json; the notebook only reads that file.
"""
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer

import bpe
from stage3_smoothing import Smoothed

TOKENIZER, BPE_FILE = "bpe4k", "data/bpe/bpe_4000.json"  # same vocabulary at every size, so perplexities are comparable
SIZES = [1000, 3000, 10000, 30000, 100000]
SPLIT = Path("data/splits/90-5-5")
RESULTS = Path("results/stage4_scaling.json")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MAX_TRAIN_LEN = 128  # training sentences are cut here to bound memory; evaluation never cuts
BATCH = 64          # sentences per step; the tokeniser side experiment lowers both for long character-like sequences


class LSTMLM(nn.Module):
    """Embedding -> 2 LSTM layers -> scores for every token. Input and output embeddings are shared (tied).

    Reads a sentence one token at a time, carrying a memory vector forward; the next-token scores come from that memory.
    Tying the input and output embeddings halves the parameter count and is standard for small LMs.
    """

    def __init__(self, vocab, dim=256, layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(dim, dim, layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(dim, vocab)
        self.out.weight = self.emb.weight

    def forward(self, x):
        h, _ = self.lstm(self.drop(self.emb(x)))  # token ids -> embeddings -> LSTM hidden states, one per position
        return self.out(self.drop(h))            # hidden state -> a score for every vocabulary token


class TinyTransformer(nn.Module):
    """A small GPT-style model: token + position embeddings -> 3 self-attention layers that can only look backwards."""

    def __init__(self, vocab, dim=256, layers=3, heads=4, dropout=0.1, max_len=1024):
        super().__init__()
        self.emb, self.pos, self.max_len = nn.Embedding(vocab, dim), nn.Embedding(max_len, dim), max_len
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers)
        self.norm, self.drop = nn.LayerNorm(dim), nn.Dropout(dropout)
        self.out = nn.Linear(dim, vocab)
        self.out.weight = self.emb.weight

    def forward(self, x):
        n = x.shape[1]
        positions = torch.arange(n, device=x.device).clamp(max=self.max_len - 1)
        mask = torch.triu(torch.full((n, n), float("-inf"), device=x.device), diagonal=1)  # -inf above the diagonal: no peeking at later tokens
        h = self.layers(self.drop(self.emb(x) + self.pos(positions)), mask=mask)
        return self.out(self.norm(h))


ARCHS = {  # name: (how to build it, learning rate)
    "lstm": (lambda vocab, dim: LSTMLM(vocab, dim), 2e-3),
    "transformer": (lambda vocab, dim: TinyTransformer(vocab, dim), 5e-4),
}


class Encoder:
    """Sentences -> lists of BPE ids, wrapped as <s> ... </s>, exactly the tokens the n-gram model sees."""

    def __init__(self, bpe_file=BPE_FILE):
        self.tok = Tokenizer.from_file(bpe_file)
        v = self.tok.get_vocab_size()
        self.bos, self.eos, self.pad, self.vocab = v, v + 1, v + 2, v + 3

    def __call__(self, sentences):
        return [[self.bos] + e.ids + [self.eos] for e in self.tok.encode_batch(sentences)]


def batches(seqs, pad, size, shuffle, max_len=None):
    """Yield (input, target) tensors. Target is the input shifted one token left: predict the next token at every position."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # similar lengths together = little padding
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    if shuffle:
        random.shuffle(chunks)
    for chunk in chunks:
        rows = [seqs[i][:max_len] if max_len else seqs[i] for i in chunk]
        x = torch.full((len(rows), max(map(len, rows))), pad)
        for r, row in enumerate(rows):
            x[r, :len(row)] = torch.tensor(row)
        yield x[:, :-1].to(DEVICE), x[:, 1:].to(DEVICE)  # input = everything but the last; target = shifted by one


@torch.no_grad()
def evaluate(model, seqs, pad, n_chars):
    """(perplexity, bits per character, tokens predicted): the same three numbers Smoothed.evaluate returns."""
    model.eval()
    nats, n = 0.0, 0
    for x, y in batches(seqs, pad, BATCH, shuffle=False):
        loss = nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=pad, reduction="sum")
        nats += loss.item()
        n += (y != pad).sum().item()
    bits = nats / math.log(2)
    return 2 ** (bits / n), bits / n_chars, n


def train_neural(arch, train_seqs, dev_seqs, enc, dev_chars, dim=256, max_epochs=40, patience=2, log=print):
    """Train until the dev score stops improving (early stopping), then return the best model seen."""
    torch.manual_seed(0)
    random.seed(0)
    build, lr = ARCHS[arch]
    model = build(enc.vocab, dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = math.inf, None, 0
    for epoch in range(1, max_epochs + 1):  # one epoch = one pass over the training data
        model.train()
        for x, y in batches(train_seqs, enc.pad, BATCH, shuffle=True, max_len=MAX_TRAIN_LEN):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=enc.pad)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # cap the update size; stops rare huge steps from wrecking training
            opt.step()
        pp = evaluate(model, dev_seqs, enc.pad, dev_chars)[0]
        log(f"      epoch {epoch:>2}: dev perplexity {pp:,.1f}")
        if pp < best:  # dev score improved: remember these weights
            best, bad = pp, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:  # no improvement for `patience` epochs: stop (early stopping)
                break
    model.load_state_dict(best_state)
    return model, epoch


def best_ngram(subset, dev_s, tokenizer):
    """Modified Kneser-Ney at its best order (2-6) for this much data."""
    ng = Smoothed(subset, 6, tokenizer, method="mkn")
    scores = {}
    for order in range(2, 7):
        ng.set_order(order)
        scores[order] = ng.evaluate(dev_s)
    order = min(scores, key=lambda o: scores[o][0])
    return {"order": order, "pp": scores[order][0], "bpc": scores[order][1]}, scores[order][2]


def fit_and_score(arch, subset, dev_s, enc, dim=256, log=print):
    dev_seqs, dev_chars = enc(dev_s), sum(map(len, dev_s))
    t0 = time.time()
    model, epochs = train_neural(arch, enc(subset), dev_seqs[:2000], enc, sum(map(len, dev_s[:2000])), dim=dim, log=log)
    pp, bpc, tokens = evaluate(model, dev_seqs, enc.pad, dev_chars)
    params = sum(p.numel() for p in model.parameters())
    return {"pp": pp, "bpc": bpc, "epochs": epochs, "seconds": round(time.time() - t0), "params": params}, tokens


def run_scaling(sizes=SIZES, archs=tuple(ARCHS), log=print):
    """Main experiment: n-gram vs each neural model at every training size, all on bpe4k tokens."""
    train_s = (SPLIT / "train.ewe.txt").read_text(encoding="utf-8").splitlines()
    dev_s = (SPLIT / "dev.ewe.txt").read_text(encoding="utf-8").splitlines()
    bpe.register(train_s)  # loads the saved BPE tokenisers
    enc = Encoder()
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"scaling": {}, "width": {}, "tokenizer": {}}
    for size in sizes:
        row = results["scaling"].setdefault(str(size), {})
        subset = train_s[:size]  # the split files are in a fixed pseudo-random order, so "first N" is a fair sample
        if "ngram" not in row:
            row["ngram"], row["tokens"] = best_ngram(subset, dev_s, TOKENIZER)
            log(f"{size:>7,} sentences | n-gram (best order {row['ngram']['order']}): perplexity {row['ngram']['pp']:,.1f}, bits/char {row['ngram']['bpc']:.3f}")
        for arch in archs:
            if arch not in row:
                row[arch], tokens = fit_and_score(arch, subset, dev_s, enc, log=lambda *_: None)
                assert tokens == row["tokens"], "every model must be graded on exactly the same tokens"
                log(f"{'':>17} | {arch:<11} perplexity {row[arch]['pp']:,.1f}, bits/char {row[arch]['bpc']:.3f}  ({row[arch]['epochs']} epochs, {row[arch]['seconds']}s)")
            save(results)
    return results


def run_fans(size=30000, arch="lstm", log=print):
    """Side experiments at one training size: how wide the model is, and which BPE vocabulary it reads."""
    train_s = (SPLIT / "train.ewe.txt").read_text(encoding="utf-8").splitlines()
    dev_s = (SPLIT / "dev.ewe.txt").read_text(encoding="utf-8").splitlines()
    bpe.register(train_s)
    subset = train_s[:size]
    results = json.loads(RESULTS.read_text())
    width = results["width"] if arch == "lstm" else results.setdefault(f"width_{arch}", {})
    tokz = results["tokenizer"] if arch == "lstm" else results.setdefault(f"tokenizer_{arch}", {})
    for dim in (64, 128, 256, 512):  # model width
        if str(dim) not in width:
            width[str(dim)], _ = fit_and_score(arch, subset, dev_s, Encoder(), dim=dim, log=lambda *_: None)
            log(f"{arch} width {dim:>3}: {width[str(dim)]}")
            save(results)
    global BATCH, MAX_TRAIN_LEN
    BATCH, MAX_TRAIN_LEN = 16, 256  # small vocabularies make sentences 2-3x longer: fewer per step, but keep whole sentences
    for vocab in (1000, 2000, 4000, 16000):  # tokenisation; perplexity is NOT comparable across these, bits/char is
        name = f"bpe{vocab / 1000:g}k"
        if name not in tokz:
            if DEVICE == "mps":
                torch.mps.empty_cache()
            neural, _ = fit_and_score(arch, subset, dev_s, Encoder(f"data/bpe/bpe_{vocab}.json"), log=lambda *_: None)
            ngram, _ = best_ngram(subset, dev_s, name) if arch == "lstm" else (results["tokenizer"][name]["ngram"], None)
            tokz[name] = {"lstm" if arch == "lstm" else arch: neural, "ngram": ngram}
            log(f"{name}: {arch} bits/char {neural['bpc']:.3f} | n-gram bits/char {ngram['bpc']:.3f}")
            save(results)
    return results


def run_english(sizes=SIZES, archs=tuple(ARCHS), log=print):
    """The scaling experiment on the ENGLISH side of the same split: high-resource language, same recipe."""
    from tokenizers import Tokenizer as _T
    from bpe import train_bpe
    train_s = (SPLIT / "train.en.txt").read_text(encoding="utf-8").splitlines()
    dev_s = (SPLIT / "dev.en.txt").read_text(encoding="utf-8").splitlines()
    path = Path("data/bpe/bpe_en_4000.json")
    if not path.exists():
        train_bpe(train_s, 4000, path)
    import stage2_ngram
    tok = _T.from_file(str(path))
    stage2_ngram.TOKENIZERS["bpe4k_en"] = lambda x, t=tok: t.encode(x).tokens
    enc = Encoder(str(path))
    results_path = Path("results/stage4_scaling_english.json")
    results = json.loads(results_path.read_text()) if results_path.exists() else {"scaling": {}}
    for size in sizes:
        row = results["scaling"].setdefault(str(size), {})
        subset = train_s[:size]
        if "ngram" not in row:
            row["ngram"], row["tokens"] = best_ngram(subset, dev_s, "bpe4k_en")
            log(f"EN {size:>7,} sentences | n-gram (best order {row['ngram']['order']}): perplexity {row['ngram']['pp']:,.1f}, bits/char {row['ngram']['bpc']:.3f}")
        for arch in archs:
            if arch not in row:
                row[arch], tokens = fit_and_score(arch, subset, dev_s, enc, log=lambda *_: None)
                assert tokens == row["tokens"]
                log(f"{'':>20} | {arch:<11} perplexity {row[arch]['pp']:,.1f}, bits/char {row[arch]['bpc']:.3f}  ({row[arch]['epochs']} epochs, {row[arch]['seconds']}s)")
            results_path.write_text(json.dumps(results, indent=1))
    return results


def run_scaled(sizes=(30000, 100000), dim=512, log=print):
    """Does a wider LSTM keep up at the large sizes? (The 256-wide one stalled at 100k.)"""
    train_s = (SPLIT / "train.ewe.txt").read_text(encoding="utf-8").splitlines()
    dev_s = (SPLIT / "dev.ewe.txt").read_text(encoding="utf-8").splitlines()
    bpe.register(train_s)
    results = json.loads(RESULTS.read_text())
    for size in sizes:
        key = f"lstm_{dim}"
        row = results["scaling"][str(size)]
        if key not in row:
            row[key], _ = fit_and_score("lstm", train_s[:size], dev_s, Encoder(), dim=dim, log=lambda *_: None)
            log(f"{size:>7,} sentences | LSTM width {dim}: perplexity {row[key]['pp']:,.1f} ({row[key]['epochs']} epochs, {row[key]['seconds']}s)")
            save(results)


def save(results):
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    print(f"device: {DEVICE}", flush=True)
    run_scaling()
    run_fans()
