"""
Small LSTM language model: the neural baseline for Section B, Question 2.

It is scored exactly like the n-gram models so the numbers compare: the same tokens and <unk> rule
(src/experiment_runner.tokenize_splits), each sentence scored on its own starting from <s>, every token
plus </s> predicted, and <pad> and <s> never predictable (the n-gram's distribution excludes <s> too).
"""

import copy
import functools
import math
import random
import time
from typing import Dict, List, Tuple

import torch
from torch import nn

from src.preprocessing import SPECIAL_BOS, SPECIAL_EOS

PAD = "<pad>"


class LSTMLM(nn.Module):
    """Embedding -> LSTM -> linear layer over the vocabulary."""

    def __init__(self, vocab_size: int, emb: int = 128, hidden: int = 256, layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.drop(self.embed(x)))
        return self.out(self.drop(h))


def build_index(vocab) -> Tuple[Dict[str, int], torch.Tensor]:
    """Token -> id (0 is padding), and a mask of ids the model may never predict (<pad>, <s>)."""
    itos = [PAD] + sorted(vocab)
    stoi = {t: i for i, t in enumerate(itos)}
    blocked = torch.zeros(len(itos), dtype=torch.bool)
    blocked[[stoi[PAD], stoi[SPECIAL_BOS]]] = True
    return stoi, blocked


def encode(sentences: List[List[str]], stoi: Dict[str, int]) -> List[List[int]]:
    """<s> tokens </s> as ids; the model reads ids[:-1] and predicts ids[1:]."""
    return [[stoi[SPECIAL_BOS]] + [stoi[t] for t in s] + [stoi[SPECIAL_EOS]] for s in sentences]


def group_indices(seqs: List[List[int]], batch_tokens: int, rng=None) -> List[List[int]]:
    """Length-sorted groups of sentence indices holding about batch_tokens tokens each; shuffled if rng."""
    groups, group, longest = [], [], 0
    for i in sorted(range(len(seqs)), key=lambda i: len(seqs[i])):
        if group and max(longest, len(seqs[i])) * (len(group) + 1) > batch_tokens:
            groups.append(group)
            group, longest = [], 0
        group.append(i)
        longest = max(longest, len(seqs[i]))
    groups.append(group)
    if rng:
        rng.shuffle(groups)
    return groups


def make_batch(seqs: List[List[int]], group: List[int]):
    """Pads one group into (inputs, targets); padding is 0 in the input and -100 in the target."""
    length = max(len(seqs[i]) for i in group) - 1
    x = torch.zeros(len(group), length, dtype=torch.long)
    y = torch.full((len(group), length), -100, dtype=torch.long)
    for row, i in enumerate(group):
        x[row, : len(seqs[i]) - 1] = torch.tensor(seqs[i][:-1])
        y[row, : len(seqs[i]) - 1] = torch.tensor(seqs[i][1:])
    return x, y


def batches(seqs: List[List[int]], batch_tokens: int, rng=None):
    """Padded batches of about batch_tokens tokens each."""
    for group in group_indices(seqs, batch_tokens, rng):
        yield make_batch(seqs, group)


def loss_sum(model: LSTMLM, x, y, blocked) -> torch.Tensor:
    logits = model(x).masked_fill(blocked, float("-inf"))
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum")


def total_nll(model: LSTMLM, seqs, blocked) -> Tuple[float, int]:
    """Sum of -ln P over every predicted token, and how many tokens that is."""
    model.eval()
    nll, n = 0.0, 0
    with torch.no_grad():
        for x, y in batches(seqs, 8192):
            nll += loss_sum(model, x, y, blocked).item()
            n += int((y != -100).sum())
    return nll, n


def train_and_score(train, val, test, vocab_size, blocked, config, seed, max_epochs, patience, lr=2e-3,
                    log=functools.partial(print, flush=True)) -> dict:  # flush: these runs are hours long and redirected to a file
    """Adam with gradient clipping; keeps the epoch with the best validation perplexity, then scores test once."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = LSTMLM(vocab_size, **config)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, stale, history, t0 = float("inf"), None, 0, [], time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        groups = group_indices(train, 4096, rng)
        done, t_epoch = 0, time.time()
        for b, group in enumerate(groups, 1):
            x, y = make_batch(train, group)
            n = int((y != -100).sum())
            loss = loss_sum(model, x, y, blocked) / n
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            done += n
            if b % 50 == 0:
                log(f"      batch {b}/{len(groups)}, {done / (time.time() - t_epoch):.0f} tokens/s")
        val_nll, val_n = total_nll(model, val, blocked)
        history.append(round(math.exp(val_nll / val_n), 3))
        log(f"    seed {seed} epoch {epoch}: val PPL {history[-1]}  [{time.time() - t0:.0f}s]")
        if history[-1] < best:
            best, best_state, stale = history[-1], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    test_nll, test_n = total_nll(model, test, blocked)
    return {
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": history.index(best) + 1,
        "val_perplexity_per_epoch": history,
        "test_nll": test_nll,
        "test_tokens": test_n,
        "test_perplexity": round(math.exp(test_nll / test_n), 2),
        "params": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(time.time() - t0, 1),
    }
