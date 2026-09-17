"""
N-Gram Language Model implementations with multiple smoothing techniques:
- Maximum Likelihood Estimation (MLE)
- Add-k (Laplace / Lidstone) Smoothing
- Linear Interpolation
- Interpolated Kneser-Ney Smoothing
Includes perplexity calculation and text generation.
"""

from collections import defaultdict, Counter
import math
import random
from typing import List, Tuple, Dict, Optional, Set
from src.preprocessing import pad_sentence, SPECIAL_BOS, SPECIAL_EOS, SPECIAL_UNK


class NGramLM:
    """
    N-gram Language Model with configurable smoothing.
    """

    def __init__(self, n: int = 3, smoothing: str = "laplace", k: float = 1.0):
        """
        Args:
            n: Order of n-gram (e.g. 1 for unigram, 2 for bigram, 3 for trigram).
            smoothing: Smoothing technique: 'mle', 'laplace', 'interpolation', 'kneser_ney'.
            k: Parameter for add-k smoothing (default 1.0 for Laplace).
        """
        self.n = n
        self.smoothing = smoothing.lower()
        self.k = k

        # Counts: context (n-1 gram tuple) -> Counter of next words
        self.ngram_counts: Dict[int, Dict[Tuple[str, ...], Counter]] = {
            i: defaultdict(Counter) for i in range(1, n + 1)
        }
        self.context_totals: Dict[int, Dict[Tuple[str, ...], int]] = {
            i: defaultdict(int) for i in range(1, n + 1)
        }
        self.vocab: Set[str] = set()
        self.total_words: int = 0

        # Weights for linear interpolation (default uniform)
        self.lambdas: List[float] = [1.0 / n] * n

        # Discount for Kneser-Ney
        self.discount: float = 0.75

    def fit(self, tokenized_corpus: List[List[str]], vocab: Optional[Set[str]] = None) -> "NGramLM":
        """
        Trains the language model on a tokenized corpus.

        Args:
            tokenized_corpus: List of tokenized sentences.
            vocab: Optional pre-defined vocabulary set.
        """
        if vocab is not None:
            self.vocab = set(vocab)
        else:
            self.vocab = {SPECIAL_BOS, SPECIAL_EOS, SPECIAL_UNK}
            for sent in tokenized_corpus:
                self.vocab.update(sent)

        for sent in tokenized_corpus:
            padded = pad_sentence(sent, self.n)
            self.total_words += len(sent) + 1  # tokens plus EOS

            for order in range(1, self.n + 1):
                for i in range(order - 1, len(padded)):
                    if order == 1:
                        word = padded[i]
                        self.ngram_counts[1][()][word] += 1
                        self.context_totals[1][()] += 1
                    else:
                        context = tuple(padded[i - order + 1 : i])
                        word = padded[i]
                        self.ngram_counts[order][context][word] += 1
                        self.context_totals[order][context] += 1

        return self

    def set_interpolation_weights(self, lambdas: List[float]):
        """Sets weights for linear interpolation across n-gram orders."""
        assert len(lambdas) == self.n, f"Expected {self.n} weights, got {len(lambdas)}"
        total = sum(lambdas)
        self.lambdas = [w / total for w in lambdas]

    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Computes P(word | context).
        Context length is expected to be (self.n - 1).
        """
        if self.smoothing == "mle":
            return self._prob_mle(word, context)
        elif self.smoothing == "laplace":
            return self._prob_laplace(word, context)
        elif self.smoothing == "interpolation":
            return self._prob_interpolation(word, context)
        elif self.smoothing == "kneser_ney":
            return self._prob_kneser_ney(word, context)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing}")

    def _prob_mle(self, word: str, context: Tuple[str, ...]) -> float:
        context_count = self.context_totals[self.n].get(context, 0)
        if context_count == 0:
            return 0.0
        return self.ngram_counts[self.n][context][word] / context_count

    def _prob_laplace(self, word: str, context: Tuple[str, ...]) -> float:
        V = len(self.vocab)
        count = self.ngram_counts[self.n][context][word]
        total = self.context_totals[self.n].get(context, 0)
        return (count + self.k) / (total + self.k * V)

    def _prob_interpolation(self, word: str, context: Tuple[str, ...]) -> float:
        """Linear interpolation of unigram, bigram, ..., n-gram probabilities."""
        prob = 0.0
        V = len(self.vocab)

        for i, weight in enumerate(self.lambdas):
            order = i + 1
            if order == 1:
                c = self.ngram_counts[1][()][word]
                tot = self.context_totals[1][()]
                order_prob = (c + 1e-5) / (tot + 1e-5 * V) if tot > 0 else 1.0 / V
            else:
                ctx = context[-(order - 1) :]
                c = self.ngram_counts[order][ctx][word]
                tot = self.context_totals[order].get(ctx, 0)
                order_prob = c / tot if tot > 0 else 0.0

            prob += weight * order_prob

        return max(prob, 1e-12)

    def _prob_kneser_ney(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Interpolated Kneser-Ney smoothing for bigrams and trigrams.
        """
        if self.n == 1:
            return self._prob_laplace(word, ())

        d = self.discount
        V = len(self.vocab)

        # Continuation count for lower orders
        def continuation_prob(w: str) -> float:
            num_contexts = sum(
                1 for ctx, words in self.ngram_counts[2].items() if w in words
            )
            total_bigram_types = max(1, sum(len(words) for words in self.ngram_counts[2].values()))
            return max(num_contexts, 1e-5) / total_bigram_types

        if self.n == 2:
            ctx_count = self.context_totals[2].get(context, 0)
            if ctx_count == 0:
                return continuation_prob(word)
            word_count = self.ngram_counts[2][context][word]
            highest_order = max(word_count - d, 0) / ctx_count
            lambda_val = (d * len(self.ngram_counts[2][context])) / ctx_count
            return highest_order + lambda_val * continuation_prob(word)

        # General / trigram fallback
        ctx_count = self.context_totals[self.n].get(context, 0)
        if ctx_count == 0:
            lower_ctx = context[1:]
            return self._prob_laplace(word, lower_ctx)

        word_count = self.ngram_counts[self.n][context][word]
        highest = max(word_count - d, 0) / ctx_count
        lambda_val = (d * len(self.ngram_counts[self.n][context])) / ctx_count
        lower_prob = self._prob_laplace(word, context[1:])
        return highest + lambda_val * lower_prob

    def perplexity(self, tokenized_test: List[List[str]]) -> float:
        """
        Computes the perplexity of the model on unseen test sentences.
        Perplexity = exp(- (1/N) * sum(log P(w_i | context)))
        """
        log_prob_sum = 0.0
        total_tokens = 0

        for sent in tokenized_test:
            padded = pad_sentence(sent, self.n)
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1 : i])
                word = padded[i]
                prob = self.probability(word, context)
                if prob <= 0:
                    prob = 1e-12
                log_prob_sum += math.log(prob)
                total_tokens += 1

        if total_tokens == 0:
            return float("inf")

        cross_entropy = -log_prob_sum / total_tokens
        return math.exp(cross_entropy)

    def generate(self, max_length: int = 20, seed_tokens: Optional[List[str]] = None, temperature: float = 1.0) -> str:
        """
        Generates text by sampling from the language model given an optional prompt.
        """
        if seed_tokens is None:
            tokens = [SPECIAL_BOS] * (self.n - 1)
        else:
            tokens = [SPECIAL_BOS] * max(0, self.n - 1 - len(seed_tokens)) + seed_tokens

        result = list(tokens)

        for _ in range(max_length):
            context = tuple(result[-(self.n - 1) :])
            candidates = list(self.vocab - {SPECIAL_BOS})
            probs = [self.probability(w, context) for w in candidates]

            if temperature != 1.0 and temperature > 0:
                # Apply temperature
                log_p = [math.log(max(p, 1e-12)) / temperature for p in probs]
                max_lp = max(log_p)
                exp_p = [math.exp(lp - max_lp) for lp in log_p]
                sum_exp = sum(exp_p)
                probs = [ep / sum_exp for ep in exp_p]

            total_p = sum(probs)
            if total_p == 0:
                break
            norm_probs = [p / total_p for p in probs]

            next_word = random.choices(candidates, weights=norm_probs, k=1)[0]
            if next_word == SPECIAL_EOS:
                break
            result.append(next_word)

        # Strip special BOS tokens for readable output
        generated = [t for t in result if t not in (SPECIAL_BOS, SPECIAL_EOS)]
        return " ".join(generated)
