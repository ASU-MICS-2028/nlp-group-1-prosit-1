"""
Experimentation harness for exploring tokenization strategies and N-gram orders (1 to 6).
Computes vocabulary size, total tokens, OOV rate, zero-count sparsity rate, validation and test
perplexity, per-word perplexity, and a seeded generation sample.
"""

from typing import List, Dict, Any
import math
import random
from src.ngram import NGramLM
from src.preprocessing import build_vocabulary, replace_oov_tokens, pad_sentence, SPECIAL_UNK

RANDOM_SEED = 42


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
    val_corpus: List[str],
    test_corpus: List[str],
    tokenizer,
    max_order: int = 6,
    smoothing: str = "kneser_ney",
) -> List[Dict[str, Any]]:
    """
    Runs an experimental sweep across N-gram orders N=1 to max_order for a given tokenizer.
    The order to report is chosen on validation perplexity, never on test.

    Returns a list of metrics dictionaries per order N. Besides the test perplexity per token, each
    row has per_word_perplexity: the total test log-probability divided by the number of whitespace
    words (+1 end-of-sentence per sentence). This is the number to compare across tokenizers;
    per-token perplexity is not (a model that predicts single characters has far fewer choices per
    step than one that predicts whole words). For it to be fair, every model must pay for the whole
    text: predicting <unk> does not say which word it was, so each <unk> is also charged the cost of
    spelling the word with a small character model of the rare training words (oov_spelling_nats).
    """
    tokenized_train = [tokenizer.tokenize(line) for line in train_corpus if line.strip()]
    tokenized_val = [tokenizer.tokenize(line) for line in val_corpus if line.strip()]
    tokenized_test = [tokenizer.tokenize(line) for line in test_corpus if line.strip()]

    # Closed vocabulary from the training partition only. Tokens seen once in training become <unk>,
    # so <unk> gets a real probability estimate for the unknown words it stands for at test time.
    vocab, _ = build_vocabulary(tokenized_train, min_freq=2)
    clean_train = replace_oov_tokens(tokenized_train, vocab)
    clean_val = replace_oov_tokens(tokenized_val, vocab)
    clean_test = replace_oov_tokens(tokenized_test, vocab)

    test_tokens = sum(len(s) + 1 for s in clean_test)  # the denominator perplexity() uses (+1 for </s>)
    test_words = sum(len(line.split()) + 1 for line in test_corpus if line.strip())
    oov_tokens = sum(t == SPECIAL_UNK for s in clean_test for t in s)

    # Character trigram "speller" trained on the tokens that became <unk> in training
    speller = NGramLM(n=3, smoothing="kneser_ney").fit(
        [list(t) for s in tokenized_train for t in s if t not in vocab] or [[SPECIAL_UNK]]
    )

    def spell_nats(token: str) -> float:
        chars = replace_oov_tokens([list(token)], speller.vocab)[0]
        return math.log(speller.perplexity([chars])) * (len(chars) + 1)

    spelling = sum(
        spell_nats(orig) for raw, clean in zip(tokenized_test, clean_test) for orig, c in zip(raw, clean) if c == SPECIAL_UNK
    )

    results = []

    for n in range(1, max_order + 1):
        # Measure sparsity
        sparsity = compute_ngram_sparsity(clean_train, clean_test, n)

        # Train model
        model = NGramLM(n=n, smoothing=smoothing).fit(clean_train, vocab=vocab)

        val_ppl = model.perplexity(clean_val)
        ppl = model.perplexity(clean_test)

        # Same counts, equal-weight linear interpolation, for comparison
        model.smoothing = "interpolation"
        interp_ppl = model.perplexity(clean_test)
        model.smoothing = smoothing

        # Sample generation
        random.seed(RANDOM_SEED)
        sample = model.generate(max_length=12, temperature=0.7)

        results.append({
            "order": n,
            "order_name": f"{n}-gram" if n > 3 else ("Unigram" if n == 1 else ("Bigram" if n == 2 else "Trigram")),
            "tokenizer": tokenizer.name,
            "smoothing": smoothing,
            "vocab_size": len(vocab),
            "train_tokens": sum(len(s) for s in clean_train),
            "test_tokens": sum(len(s) for s in clean_test),
            "test_words": test_words,
            "oov_rate_pct": round(100 * oov_tokens / test_tokens, 2),
            "sparsity_pct": round(sparsity, 2),
            "val_perplexity": round(val_ppl, 2),
            "perplexity": round(ppl, 2),
            "per_word_perplexity": round(math.exp((math.log(ppl) * test_tokens + spelling) / test_words), 2),
            "oov_spelling_nats": round(spelling, 1),
            "interpolation_perplexity": round(interp_ppl, 2),
            "kn_discounts": [round(model.discounts[k], 3) for k in range(1, n + 1)] if smoothing == "kneser_ney" else None,
            "sample_generation": sample,
        })

    return results
