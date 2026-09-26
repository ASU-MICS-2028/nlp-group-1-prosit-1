"""BPE tokenisers: learn subword pieces from the TRAINING split only, then register them in stage2_ngram.TOKENIZERS."""
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import stage2_ngram

DIR = Path("data/bpe")
SIZES = (1000, 1500, 2000, 2500, 3000, 3500, 4000, 16000)  # the 1k-4k range is swept finely to find the best size


def train_bpe(sentences, vocab_size, path):
    """Glue the most frequent neighbouring pair together, over and over, until the vocabulary is this big."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Metaspace()  # marks word starts with ▁ so text can be rebuilt exactly
    tok.decoder = decoders.Metaspace()
    tok.train_from_iterator(sentences, trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def register(sentences, sizes=SIZES):
    """Train (or reload) one BPE tokeniser per size; returns the names added to TOKENIZERS ('bpe1k', ...)."""
    names = []
    for size in sizes:
        path = DIR / f"bpe_{size}.json"
        tok = Tokenizer.from_file(str(path)) if path.exists() else train_bpe(sentences, size, path)
        name = f"bpe{size / 1000:g}k"  # 1000 -> bpe1k, 1500 -> bpe1.5k
        stage2_ngram.TOKENIZERS[name] = lambda s, t=tok: t.encode(s).tokens
        names.append(name)
    return names
