"""Stage 3: smoothing. Turns MLE's zeros into small probabilities, so perplexity is finite and models are comparable.

Unknown words: the vocabulary gets one extra slot standing for "any token not seen in training", so every method
leaves some probability for it. No <UNK> substitution in training, which keeps Stage 2's vocabulary numbers intact.
"""
import math
from collections import Counter, defaultdict

from stage2_ngram import BOS, TOKENIZERS, pad, train

METHODS = ("add_k", "jm", "gt", "katz", "kn", "mkn")
GT_K = 5  # Good-Turing only adjusts counts up to this; bigger counts are trusted as they are (Katz's rule)


def gt_discounts(model):
    """Good-Turing discount ratios d_r for r = 1..GT_K: 'a count of r is really worth d_r * r'.

    Built from counts-of-counts: if many n-grams were seen once and few twice, a count of 1 is mostly luck.
    """
    n = counts_of_counts(model, upto=GT_K + 1)
    fallback = {r: (r - 0.75) / r for r in range(1, GT_K + 1)}  # tiny corpora: counts-of-counts too patchy to use
    if not all(n[r] for r in range(1, GT_K + 2)):
        return fallback
    cut = (GT_K + 1) * n[GT_K + 1] / n[1]
    if cut >= 1:  # small corpora with few singletons: the formula breaks down, so use the fixed discounts
        return fallback
    d = {r: ((r + 1) * n[r + 1] / (r * n[r]) - cut) / (1 - cut) for r in range(1, GT_K + 1)}
    return d if all(0 < v <= 1 for v in d.values()) else fallback


def counts_of_counts(model, upto=4):
    """How many n-grams were seen exactly once, twice, ... Needed for the modified Kneser-Ney discounts."""
    n = Counter()
    for followers in model.counts.values():
        for c in followers.values():
            if c <= upto:
                n[c] += 1
    return n


def mkn_discounts(model):
    """Chen & Goodman's three discounts D1, D2, D3+, estimated from how many n-grams were seen 1/2/3/4 times."""
    n = counts_of_counts(model)
    if not n[1] or not n[2]:
        return 0.5, 1.0, 1.5  # too few counts to estimate; fall back to sensible fixed values
    y = n[1] / (n[1] + 2 * n[2])
    d1 = 1 - 2 * y * n[2] / n[1]
    d2 = 2 - 3 * y * n[3] / n[2] if n[2] and n[3] else 1.0
    d3 = 3 - 4 * y * n[4] / n[3] if n[3] and n[4] else 1.5
    return max(d1, 0.0), max(d2, 0.0), max(d3, 0.0)


class Smoothed:
    """One smoothed n-gram model. `method` picks the rung of the ladder; k / lam / D are its settings."""

    def __init__(self, sentences, n, tokenizer="whitespace", method="kn", k=0.01, lam=0.7, discount=0.75, unk="gt"):
        self.n, self.tokenizer, self.method, self.k, self.lam, self.D = n, tokenizer, method, k, lam, discount
        self.models = {order: train(sentences, order, tokenizer) for order in range(1, n + 1)}
        self.vocab_size = len(self.models[1].counts[()]) + 1  # +1 slot for "a token never seen in training"

        # How likely is the next token to be one we have never seen? Good-Turing's answer: the share of training
        # tokens that were seen exactly once (they were "new" the moment before we met them).
        # unk="gt": interpolation / Kneser-Ney give the unknown slot this much. unk="floor": the old behaviour,
        # where it only got a crumb from the uniform floor (kept for the before/after comparison and the NLTK check).
        self.unk = unk
        self.p_unk = max(counts_of_counts(self.models[1])[1], 1) / self.models[1].totals[()]

        # Continuation counts: cont[k][context][token] = how many DIFFERENT tokens preceded "context token".
        # Kneser-Ney's key idea: in a novel context, a token that turns up after many different words is a better bet.
        self.cont, self.cont_totals = {}, {}
        for order in range(1, n):
            d = defaultdict(Counter)
            for ctx, followers in self.models[order + 1].counts.items():
                for token in followers:
                    d[ctx[1:]][token] += 1  # each longer context contributes exactly one distinct predecessor
            self.cont[order] = d
            self.cont_totals[order] = {ctx: sum(c.values()) for ctx, c in d.items()}

        # Prepared for every method, so the notebook can switch `method` on one model without recounting.
        self.mkn = {order: mkn_discounts(self.models[order]) for order in range(1, n + 1)}
        self.gt = {order: gt_discounts(self.models[order]) for order in range(1, n + 1)}
        self._gamma, self._left, self._alpha = {}, {}, {}  # per-context caches: these values never change

    def set_order(self, n):
        """Score as an order-n model using the counts already built (n <= the order it was built with).
        Gives exactly what building a fresh order-n model would, without recounting."""
        assert 1 <= n <= max(self.models)
        self.n = n
        self._gamma.clear()  # these weights depend on which level is the top one

    def _freed(self, order, context):
        """(scale, leftover) for a seen context: what its seen followers keep after discounting, and what is freed."""
        hit = self._left.get((order, context))
        if hit is None:
            c, total, d = self.models[order].counts[context], self.models[order].totals[context], self.gt[order]
            leftover = 1 - sum(d.get(v, 1.0) * v for v in c.values()) / total
            # every follower seen more than GT_K times frees nothing; keep one observation's worth for the unseen
            hit = (1.0, leftover) if leftover > 1e-12 else (total / (total + 1), 1 / (total + 1))
            self._left[(order, context)] = hit
        return hit

    def _katz(self, order, context, token, backoff):
        """Good-Turing discounted estimate. backoff=False shares the freed mass equally (plain Good-Turing);
        backoff=True hands it to the shorter context instead (Katz)."""
        model = self.models[order]
        c = model.counts.get(context)
        if not c:  # context never seen
            return self._katz(order - 1, context[1:], token, True) if backoff and order > 1 else 1 / self.vocab_size
        scale, leftover = self._freed(order, context)
        count = c.get(token, 0)
        if count:
            return scale * self.gt[order].get(count, 1.0) * count / model.totals[context]
        if not backoff or order == 1:
            return leftover / (self.vocab_size - len(c))  # equal shares among tokens never seen after this context
        alpha = self._alpha.get((order, context))
        if alpha is None:  # scale the shorter context's probabilities so everything still adds up to 1
            seen_lower = sum(self._katz(order - 1, context[1:], w, True) for w in c)
            alpha = leftover / max(1 - seen_lower, 1e-12)
            self._alpha[(order, context)] = alpha
        return alpha * self._katz(order - 1, context[1:], token, True)

    def _level(self, order, context):
        """(counts, total) at this order: real counts at the top order, continuation counts below it."""
        if order == self.n:
            return self.models[order].counts.get(context), self.models[order].totals.get(context, 0)
        return self.cont[order].get(context), self.cont_totals[order].get(context, 0)

    def prob(self, context, token):
        context = tuple(context)[max(0, len(context) - self.n + 1):] if self.n > 1 else ()
        if self.method == "add_k":
            c, total = self.models[self.n].counts.get(context), self.models[self.n].totals.get(context, 0)
            return ((c[token] if c else 0) + self.k) / (total + self.k * self.vocab_size)
        if self.method in ("gt", "katz"):
            return self._katz(self.n, context, token, backoff=self.method == "katz")

        if self.unk == "gt":  # first decide "is this a brand-new token?", then smooth over the known ones only
            if token not in self.models[1].counts[()]:
                return self.p_unk
            p, keep = 1 / (self.vocab_size - 1), 1 - self.p_unk
        else:
            p, keep = 1 / self.vocab_size, 1.0  # "all tokens equally likely" floor, unknown slot included
        p = self._interpolate(context, token, p)  # build the estimate from the unigram level up to the top order
        return keep * p                            # scale so known-token mass + the unknown share sums to 1

    def _interpolate(self, context, token, p):
        """Build the estimate from the unigram level up to the top order, starting from the floor `p`."""
        for order in range(1, self.n + 1):
            sub = context[len(context) - order + 1:] if order > 1 else ()
            if self.method == "jm":
                c, total = self.models[order].counts.get(sub), self.models[order].totals.get(sub, 0)
                if not total:
                    continue  # context never seen at this level: keep the lower-order estimate, or λ's mass vanishes
                p = self.lam * (c[token] / total) + (1 - self.lam) * p
                continue

            c, total = self._level(order, sub)
            if not total:
                continue  # nothing seen at this level: keep the lower-order estimate
            count = c[token] if c else 0
            if self.method == "kn":
                left = max(count - self.D, 0) / total
                weight = self.D * len(c) / total
            else:  # mkn: a different discount depending on whether the count is 1, 2, or 3+
                d1, d2, d3 = self.mkn[order]
                d = 0 if count == 0 else d1 if count == 1 else d2 if count == 2 else d3
                left = max(count - d, 0) / total
                weight = self._gamma.get((order, sub))
                if weight is None:  # depends only on the context, so work it out once and keep it
                    n1 = sum(1 for v in c.values() if v == 1)
                    n2 = sum(1 for v in c.values() if v == 2)
                    weight = (d1 * n1 + d2 * n2 + d3 * (len(c) - n1 - n2)) / total
                    self._gamma[(order, sub)] = weight
            p = left + weight * p
        return p

    def evaluate(self, sentences, known_only=False):
        """Returns (perplexity, bits per character, tokens). Both come from the same pass.

        known_only=True scores only tokens that occur in training (unknown ones still serve as context). That
        separates "how good is the smoothing" from "how does the method treat unknown words". Ignore bits/char then.
        """
        bits, n_tokens = 0.0, 0
        tokenize = TOKENIZERS[self.tokenizer]
        known = self.models[1].counts[()]
        for s in sentences:
            toks = pad(tokenize(s), self.n)
            for i in range(self.n - 1, len(toks)):
                if known_only and toks[i] not in known:
                    continue
                bits -= math.log2(self.prob(toks[i - self.n + 1:i], toks[i]))
                n_tokens += 1
        n_chars = sum(len(s) for s in sentences)
        return 2 ** (bits / n_tokens), bits / n_chars, n_tokens


def selfcheck():
    """Probabilities must add up to 1 over the whole vocabulary, and nothing may be impossible."""
    toy = ["Aƒetɔ, dzɔ nye.", "Aƒetɔ, va kaba!", "Si va ɖe mí,", "Xɔ mí, ɖe mí", "Eya ɖe mí vavã."]
    for method, unk in [(m, u) for m in METHODS for u in ("gt", "floor")]:
        for n in (1, 2, 3):
            m = Smoothed(toy, n, method=method, k=0.1, lam=0.6, unk=unk)
            vocab = list(m.models[1].counts[()]) + ["A-WORD-NEVER-SEEN"]
            for context in [("ɖe",) * (n - 1), ("ZZZ",) * (n - 1)]:  # a seen context and an unseen one
                total = sum(m.prob(context, w) for w in vocab)
                assert abs(total - 1) < 1e-9, f"{method} n={n} context={context} sums to {total}"
                assert m.prob(context, "A-WORD-NEVER-SEEN") > 0
            pp, bpc, _ = m.evaluate(["Aƒetɔ, ɖe mí"])  # the sentence MLE called impossible
            assert math.isfinite(pp) and bpc > 0
    # one big model scored at a lower order must equal a model built at that order
    big = Smoothed(toy, 3, method="mkn")
    big.set_order(2)
    assert big.evaluate(toy)[0] == Smoothed(toy, 2, method="mkn").evaluate(toy)[0]
