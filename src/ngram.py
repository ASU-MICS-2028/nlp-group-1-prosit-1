"""
N-Gram Language Model implementations with multiple smoothing techniques:
- Maximum Likelihood Estimation (MLE)
- Add-k (Laplace / Lidstone) Smoothing
- Linear Interpolation
- Interpolated Kneser-Ney Smoothing
Includes perplexity calculation and text generation.

Every smoothed model here is a proper distribution: for any context, P(w | context) summed over the
vocabulary is 1 (tests/test_pipeline.py checks this for seen and unseen contexts).
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

    def __init__(self, n: int = 3, smoothing: str = "laplace", k: float = 1.0, discount: Optional[float] = None):
        """
        Args:
            n: Order of n-gram (e.g. 1 for unigram, 2 for bigram, 3 for trigram).
            smoothing: Smoothing technique: 'mle', 'laplace', 'interpolation', 'kneser_ney'.
            k: Parameter for add-k smoothing (default 1.0 for Laplace).
            discount: Kneser-Ney discount D. None (default) estimates one per order from the training
                counts with Ney's formula D = n1 / (n1 + 2 * n2).
        """
        self.n = n
        self.smoothing = smoothing.lower()
        self.k = k
        self.discount = discount

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
        # <s> is only ever context, never predicted, so it is not part of the distribution
        self.num_predictable = len(self.vocab - {SPECIAL_BOS})

        for sent in tokenized_corpus:
            padded = pad_sentence(sent, self.n)
            self.total_words += len(sent) + 1  # tokens plus EOS

            # Count every order only at positions we actually predict (real tokens and </s>).
            # Counting from index 0 would make the <s> padding a "word" in the lower orders.
            for i in range(self.n - 1, len(padded)):
                for order in range(1, self.n + 1):
                    context = tuple(padded[i - order + 1 : i])
                    self.ngram_counts[order][context][padded[i]] += 1
                    self.context_totals[order][context] += 1

        if self.smoothing == "kneser_ney":
            self._fit_kneser_ney()
        return self

    def _fit_kneser_ney(self):
        """
        Builds the tables Interpolated Kneser-Ney needs (Chen & Goodman, 1998).
        Highest order: raw counts. Every lower order: continuation counts, i.e. for (context, w) the
        number of distinct words seen immediately before it. n-grams that start with <s> cannot be
        extended to the left, so they keep their raw counts.
        """
        self.kn_counts = {self.n: self.ngram_counts[self.n]}
        for order in range(self.n - 1, 0, -1):
            cont = defaultdict(Counter)
            for context, words in self.ngram_counts[order + 1].items():
                for w in words:
                    cont[context[1:]][w] += 1
            for context, words in self.ngram_counts[order].items():
                if context and context[0] == SPECIAL_BOS:
                    cont[context] = Counter(words)
            self.kn_counts[order] = cont

        self.kn_totals, self.kn_types, self.discounts = {}, {}, {}
        for order, table in self.kn_counts.items():
            self.kn_totals[order] = {ctx: sum(c.values()) for ctx, c in table.items()}
            self.kn_types[order] = {ctx: len(c) for ctx, c in table.items()}
            n1 = sum(1 for c in table.values() for v in c.values() if v == 1)
            n2 = sum(1 for c in table.values() for v in c.values() if v == 2)
            ney = n1 / (n1 + 2 * n2) if n1 else 0.75
            self.discounts[order] = self.discount if self.discount is not None else ney

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

    def _count(self, order: int, context: Tuple[str, ...], word: str) -> int:
        # .get so that looking up an unseen context does not insert an empty entry
        return self.ngram_counts[order].get(context, {}).get(word, 0)

    def _prob_mle(self, word: str, context: Tuple[str, ...]) -> float:
        context_count = self.context_totals[self.n].get(context, 0)
        if context_count == 0:
            return 0.0
        return self._count(self.n, context, word) / context_count

    def _prob_laplace(self, word: str, context: Tuple[str, ...]) -> float:
        total = self.context_totals[self.n].get(context, 0)
        return (self._count(self.n, context, word) + self.k) / (total + self.k * self.num_predictable)

    def _prob_interpolation(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Linear interpolation of unigram, bigram, ..., n-gram estimates.
        An order whose context never occurred in training has no estimate at all, so its weight is
        dropped and the remaining weights are renormalised. (Treating that estimate as 0 while keeping
        its weight throws probability away, more and more as n grows.)
        """
        prob = weight_used = 0.0
        for i, weight in enumerate(self.lambdas):
            order = i + 1
            ctx = context[len(context) - order + 1 :]
            total = self.context_totals[order].get(ctx, 0)
            if total == 0:
                continue
            count = self._count(order, ctx, word)
            if order == 1:  # tiny add-k so a vocabulary word unseen in training keeps non-zero mass
                order_prob = (count + 1e-5) / (total + 1e-5 * self.num_predictable)
            else:
                order_prob = count / total
            prob += weight * order_prob
            weight_used += weight
        return prob / weight_used

    def _prob_kneser_ney(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Interpolated Kneser-Ney, built up from the unigram:
            P_k(w | h) = max(c_k(h, w) - D_k, 0) / c_k(h) + D_k * T_k(h) / c_k(h) * P_{k-1}(w | h')
        c_k are raw counts at the highest order and continuation counts below it, T_k(h) is the number
        of distinct words seen after h, and h' drops the oldest word of h. An unseen context passes all
        of its mass to the lower order. The recursion starts from a uniform distribution, so every
        vocabulary word keeps some probability.
        """
        p = 1.0 / self.num_predictable
        for order in range(1, self.n + 1):
            ctx = context[len(context) - order + 1 :]
            total = self.kn_totals[order].get(ctx, 0)
            if total == 0:
                continue
            d = self.discounts[order]
            count = self.kn_counts[order][ctx].get(word, 0)
            p = max(count - d, 0) / total + d * self.kn_types[order][ctx] / total * p
        return p

    def perplexity(self, tokenized_test: List[List[str]]) -> float:
        """
        Computes the perplexity of the model on unseen test sentences.
        Perplexity = exp(- (1/N) * sum(log P(w_i | context)))
        Returns infinity if any token gets probability 0 (unsmoothed MLE on an unseen n-gram).
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
                    return float("inf")
                log_prob_sum += math.log(prob)
                total_tokens += 1

        if total_tokens == 0:
            return float("inf")

        cross_entropy = -log_prob_sum / total_tokens
        return math.exp(cross_entropy)

    def generate(self, max_length: int = 20, seed_tokens: Optional[List[str]] = None, temperature: float = 1.0) -> str:
        """
        Generates text by sampling from the language model given an optional prompt.
        Call random.seed(...) first for a reproducible sample.
        """
        if seed_tokens is None:
            tokens = [SPECIAL_BOS] * (self.n - 1)
        else:
            tokens = [SPECIAL_BOS] * max(0, self.n - 1 - len(seed_tokens)) + seed_tokens

        result = list(tokens)

        for _ in range(max_length):
            if self.n > 1:
                context = tuple(result[-(self.n - 1) :])
                candidates = list(self.ngram_counts[self.n].get(context, {}).keys())
                if not candidates:
                    # Backoff to unigram high-frequency continuations
                    candidates = [w for w, _ in self.ngram_counts[1][()].most_common(100) if w != SPECIAL_BOS]
            else:
                context = ()
                candidates = [w for w, _ in self.ngram_counts[1][()].most_common(200) if w != SPECIAL_BOS]

            if not candidates:
                break
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
