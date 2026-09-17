"""
Experimentation harness for exploring tokenization strategies and N-gram orders (1 to 6).
Computes vocabulary size, total tokens, zero-count sparsity rate, perplexity, and generation.
"""

from typing import List, Dict, Any, Tuple
import math
from src.ngram import NGramLM
from src.preprocessing import build_vocabulary, replace_oov_tokens, pad_sentence


def compute_ngram_sparsity(train_tokens: List[List[str]], test_tokens: List[List[str]], n: int) -> float:
    """
    Computes the percentage of n-grams in the test set that never appeared in the training set.
    Sparsity = (Unseen Test N-grams / Total Test N-grams) * 100
    """
    train_ngrams = set()
    for sent in train_tokens:
        padded = pad_sentence(sent, n)
        for i in range(n - 1, len(padded)):
            train_ngrams.add(tuple(padded[i - n + 1 : i + 1]))

    total_test = 0
    unseen_test = 0
    for sent in test_tokens:
        padded = pad_sentence(sent, n)
        for i in range(n - 1, len(padded)):
            ng = tuple(padded[i - n + 1 : i + 1])
            total_test += 1
            if ng not in train_ngrams:
                unseen_test += 1

    if total_test == 0:
        return 0.0
    return (unseen_test / total_test) * 100.0


def run_ngram_experiment(
    train_corpus: List[str],
    test_corpus: List[str],
    tokenizer,
    max_order: int = 6,
    smoothing: str = "laplace",
) -> List[Dict[str, Any]]:
    """
    Runs an experimental sweep across N-gram orders N=1 to max_order for a given tokenizer.

    Returns a list of metrics dictionaries per order N:
    - order N
    - tokenizer_name
    - vocab_size
    - total_train_tokens
    - total_test_tokens
    - sparsity_pct (% unseen n-grams in test)
    - perplexity
    - sample_generation
    """
    tokenized_train = [tokenizer.tokenize(line) for line in train_corpus if line.strip()]
    tokenized_test = [tokenizer.tokenize(line) for line in test_corpus if line.strip()]

    # Induce closed vocabulary strictly from training partition
    vocab, freqs = build_vocabulary(tokenized_train, min_freq=1)
    clean_train = replace_oov_tokens(tokenized_train, vocab)
    clean_test = replace_oov_tokens(tokenized_test, vocab)

    results = []

    for n in range(1, max_order + 1):
        # Measure sparsity
        sparsity = compute_ngram_sparsity(clean_train, clean_test, n)

        # Train model
        model = NGramLM(n=n, smoothing=smoothing, k=0.1 if n > 2 else 1.0)
        model.fit(clean_train, vocab=vocab)

        # Compute perplexity
        ppl = model.perplexity(clean_test)

        # Sample generation
        sample = model.generate(max_length=12, temperature=0.7)

        results.append({
            "order": n,
            "order_name": f"{n}-gram" if n > 3 else ("Unigram" if n == 1 else ("Bigram" if n == 2 else "Trigram")),
            "tokenizer": tokenizer.name,
            "vocab_size": len(vocab),
            "train_tokens": sum(len(s) for s in clean_train),
            "test_tokens": sum(len(s) for s in clean_test),
            "sparsity_pct": round(sparsity, 2),
            "perplexity": round(ppl, 2) if not math.isinf(ppl) else 999999.0,
            "sample_generation": sample,
        })

    return results
