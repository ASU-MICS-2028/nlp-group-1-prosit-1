"""Stage 2 helpers: tokenisers, n-gram counting, MLE probability, perplexity and generation. Written from scratch, no NLP libraries."""
import math
import re
from collections import Counter, defaultdict, namedtuple

BOS, EOS = "<s>", "</s>"  # sentence start / end markers

# Each tokeniser turns one sentence into a list of tokens. Add a line here to test another one.
TOKENIZERS = {
    "whitespace": str.split,                                        # Aƒetɔ, | va   — punctuation stays stuck to words
    "punct": lambda s: re.findall(r"\w+|[^\w\s]", s),               # Aƒetɔ | , | va
    "lower_punct": lambda s: re.findall(r"\w+|[^\w\s]", s.lower()),  # aƒetɔ | , | va
    "char": list,                                                    # A | ƒ | e | t | ɔ | , | ' ' | v | a
}

Model = namedtuple("Model", "n counts totals tokenizer")  # counts[context][next token] = count


def detokenize(tokens, tokenizer):
    if tokenizer.startswith("bpe"):
        return "".join(tokens).replace("▁", " ").strip()  # ▁ marks where a word started
    return ("" if tokenizer == "char" else " ").join(tokens)


def pad(tokens, n):
    return [BOS] * (n - 1) + tokens + [EOS]  # n-1 starts so the first real token has a full context


def train(sentences, n, tokenizer="whitespace"):
    """Count every n-gram. This is the whole of 'training' for an n-gram model."""
    tokenize = TOKENIZERS[tokenizer]
    # ponytail: plain dict-of-Counters held in RAM (~GBs at n=5 on 4M tokens); move to a trie/db only if it stops fitting.
    counts = defaultdict(Counter)
    for s in sentences:
        toks = pad(tokenize(s), n)
        for i in range(n - 1, len(toks)):
            counts[tuple(toks[i - n + 1:i])][toks[i]] += 1
    totals = {ctx: sum(c.values()) for ctx, c in counts.items()}  # C(context), the denominator
    return Model(n, counts, totals, tokenizer)


def prob(model, context, token):
    """MLE probability: C(context + token) / C(context). Zero if either was never seen."""
    c = model.counts.get(context)
    return c[token] / model.totals[context] if c else 0.0


def perplexity(model, sentences):
    """Perplexity of the MLE model. Infinite if any token gets probability 0 (an unseen n-gram)."""
    logp, n_tokens, zeros, zero_sentences = 0.0, 0, 0, 0
    for s in sentences:
        toks = pad(TOKENIZERS[model.tokenizer](s), model.n)
        bad = 0
        for i in range(model.n - 1, len(toks)):
            p = prob(model, tuple(toks[i - model.n + 1:i]), toks[i])
            n_tokens += 1
            if p:
                logp += math.log2(p)  # add logs instead of multiplying: millions of small fractions would underflow to 0
            else:
                bad += 1
        zeros += bad
        zero_sentences += bad > 0
    pp = math.inf if zeros else 2 ** (-logp / n_tokens)
    return pp, n_tokens, zeros, zero_sentences


def first_unseen(model, sentence):
    """The first n-gram of this sentence that the model never saw, or None."""
    toks = pad(TOKENIZERS[model.tokenizer](sentence), model.n)
    for i in range(model.n - 1, len(toks)):
        ctx = tuple(toks[i - model.n + 1:i])
        if not prob(model, ctx, toks[i]):
            return ctx + (toks[i],)
    return None


def generate(model, rng, max_tokens=40):
    """Sample a sentence: pick each next token at random, weighted by how often it followed this context."""
    context, out = [BOS] * (model.n - 1), []
    while len(out) < max_tokens:
        followers = model.counts[tuple(context[len(context) - model.n + 1:])]
        token = rng.choices(list(followers), weights=list(followers.values()))[0]
        if token == EOS:
            break
        out.append(token)
        context.append(token)
    return detokenize(out, model.tokenizer)


def selfcheck():
    """Check the code against the Stage 1 hand calculation."""
    toy = ["Aƒetɔ, dzɔ nye.", "Aƒetɔ, va kaba!", "Si va ɖe mí,", "Xɔ mí, ɖe mí", "Eya ɖe mí vavã."]
    m = train(toy, 2)
    assert prob(m, ("ɖe",), "mí") == 2 / 3 and prob(m, ("mí",), EOS) == 1 / 2
    pp, n, zeros, _ = perplexity(m, ["Aƒetɔ, va ɖe mí"])
    assert (n, zeros) == (5, 0) and math.isclose(pp, 30 ** (1 / 5))  # P = 1/30, N = 5  → paper answer
    assert perplexity(m, ["Aƒetɔ, ɖe mí"])[0] == math.inf  # unseen pair "Aƒetɔ, ɖe"
    assert first_unseen(m, "Aƒetɔ, ɖe mí") == ("Aƒetɔ,", "ɖe")
    pp_punct = perplexity(train(toy, 2, "punct"), ["Aƒetɔ, va ɖe mí"])[0]
    assert math.isclose(pp_punct, 80 ** (1 / 6))  # same text, different tokens → different number, not a better model
