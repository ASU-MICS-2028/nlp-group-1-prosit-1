"""
Unit test suite for low-resource Ewe language modeling pipeline.
Validates Unicode normalization, 5 tokenizers, probability conservation,
and N-gram smoothing algorithms.
"""

import math
import pytest
from src.section_b_ngram.preprocessing import normalize_ewe_text, basic_tokenize, build_vocabulary, replace_oov_tokens
from src.section_b_ngram.ewe_tokenizers import (
    WhitespaceTokenizer,
    UnicodeWordTokenizer,
    CharacterTokenizer,
    EweRuleStemmerTokenizer,
    SimpleBPETokenizer,
)
from src.section_b_ngram.ngram import NGramLM
from src.section_b_ngram.data_pipeline import clean_and_normalize_ewe_sentence
from src.section_b_ngram.experiment_runner import run_ngram_experiment


class TestUnicodePreprocessing:
    def test_ewe_characters_preserved(self):
        text = "Míedi ŋutifafa le dukɔa me kple nuɖuɖu vivi."
        norm = normalize_ewe_text(text)
        assert "ŋ" in norm
        assert "ɔ" in norm
        assert "ɖ" in norm

    def test_combining_diacritics_preserved(self):
        # Ewe nasalized vowel with combining tilde
        text = "nusrɔ̃lawo"
        tokens = basic_tokenize(text)
        assert len(tokens) == 1
        assert "ɔ̃" in tokens[0]

    def test_lookalike_letters_normalized(self):
        # Capital eth looks like Ɖ but lowercases to ð, splitting one Ewe word into two vocabulary entries
        assert clean_and_normalize_ewe_sentence("Ðasefowo ðe nya") == "Ɖasefowo ɖe nya"

    def test_html_and_url_stripping(self):
        raw = "<p>Visit https://ankora.ai for Ewe text.</p>"
        cleaned = clean_and_normalize_ewe_sentence(raw)
        assert cleaned is not None
        assert "<p>" not in cleaned
        assert "https://" not in cleaned
        assert "Ewe text." in cleaned


class TestTokenizers:
    @pytest.fixture
    def sample_sentence(self):
        return "Woezɔ loo! Nusrɔ̃lawo le suku me."

    def test_whitespace_tokenizer(self, sample_sentence):
        tok = WhitespaceTokenizer()
        tokens = tok.tokenize(sample_sentence)
        assert "loo!" in tokens  # Punctuation attached
        assert len(tokens) == 6

    def test_unicode_word_tokenizer(self, sample_sentence):
        tok = UnicodeWordTokenizer()
        tokens = tok.tokenize(sample_sentence)
        assert "!" in tokens  # Punctuation separated
        assert "woezɔ" in tokens
        assert "nusrɔ̃lawo" in tokens

    def test_ewe_stemmer(self):
        tok = EweRuleStemmerTokenizer()
        # Plural suffix -wo split off
        res_plural = tok.tokenize("atíwo")
        assert "atí" in res_plural

        # Subject pronoun prefix mí- split off
        res_prefix = tok.tokenize("míewɔ")
        assert "wɔ" in res_prefix

    def test_ewe_stemmer_is_lossless(self):
        # Affixes are kept as tokens, so the word can be rebuilt and perplexity compared per word
        pieces = EweRuleStemmerTokenizer().tokenize("nusrɔ̃lawo")
        assert pieces == ["nu+", "srɔ̃la", "+wo"]
        assert "".join(p.strip("+") for p in pieces) == "nusrɔ̃lawo"

    def test_character_tokenizer(self):
        tok = CharacterTokenizer()
        tokens = tok.tokenize("Ewe")
        assert tokens == ["e", "w", "e"]
        # Word boundaries are kept, so the character model predicts the same text as the others
        assert tok.tokenize("Ewe gbe") == ["e", "w", "e", "\u2581", "g", "b", "e"]

    def test_bpe_training_and_tokenize(self):
        corpus = [
            "woezɔ loo",
            "nusrɔ̃lawo le suku me",
            "kofi yi suku",
            "woezɔ nusrɔ̃lawo"
        ]
        bpe = SimpleBPETokenizer(num_merges=10)
        bpe.train(corpus)
        tokenized = bpe.tokenize("woezɔ")
        assert len(tokenized) > 0


class TestNGramLanguageModel:
    @pytest.fixture
    def small_corpus(self):
        sentences = [
            ["woezɔ", "loo", "nusrɔ̃lawo"],
            ["kofi", "yi", "suku", "le", "keta"],
            ["ama", "fle", "nuɖuɖu", "vivi"],
            ["woezɔ", "loo", "kofi"],
        ]
        vocab, _ = build_vocabulary(sentences, min_freq=1)
        return sentences, vocab

    def test_laplace_probability_conservation(self, small_corpus):
        sentences, vocab = small_corpus
        model = NGramLM(n=2, smoothing="laplace", k=1.0).fit(sentences, vocab=vocab)

        context = ("woezɔ",)
        total_prob = sum(model.probability(w, context) for w in model.vocab - {"<s>"})
        assert math.isclose(total_prob, 1.0, rel_tol=1e-5)

    def test_lidstone_probability_conservation(self, small_corpus):
        sentences, vocab = small_corpus
        model = NGramLM(n=2, smoothing="laplace", k=0.1).fit(sentences, vocab=vocab)

        context = ("kofi",)
        total_prob = sum(model.probability(w, context) for w in model.vocab - {"<s>"})
        assert math.isclose(total_prob, 1.0, rel_tol=1e-5)

    def test_interpolation_probability_conservation(self, small_corpus):
        sentences, vocab = small_corpus
        model = NGramLM(n=3, smoothing="interpolation").fit(sentences, vocab=vocab)
        model.set_interpolation_weights([0.2, 0.3, 0.5])

        context = ("woezɔ", "loo")
        total_prob = sum(model.probability(w, context) for w in model.vocab - {"<s>"})
        assert math.isclose(total_prob, 1.0, rel_tol=1e-4)

    def test_interpolation_sums_to_one_for_unseen_context(self, small_corpus):
        # Default equal weights; the bigram and trigram contexts below never occur in training
        sentences, vocab = small_corpus
        model = NGramLM(n=3, smoothing="interpolation").fit(sentences, vocab=vocab)
        total_prob = sum(model.probability(w, ("keta", "ama")) for w in model.vocab - {"<s>"})
        assert math.isclose(total_prob, 1.0, rel_tol=1e-9)

    @pytest.mark.parametrize("context", [("woezɔ", "loo"), ("<s>", "<s>"), ("keta", "ama")])
    def test_kneser_ney_sums_to_one(self, small_corpus, context):
        sentences, vocab = small_corpus
        model = NGramLM(n=3, smoothing="kneser_ney").fit(sentences, vocab=vocab)
        total_prob = sum(model.probability(w, context) for w in model.vocab - {"<s>"})
        assert math.isclose(total_prob, 1.0, rel_tol=1e-9)

    def test_padding_is_never_a_target(self, small_corpus):
        # <s> padding is context only; counting it as a word would steal unigram probability mass
        sentences, vocab = small_corpus
        model = NGramLM(n=6, smoothing="kneser_ney").fit(sentences, vocab=vocab)
        assert model.ngram_counts[1][()]["<s>"] == 0

    def test_kneser_ney_positive_probabilities(self, small_corpus):
        sentences, vocab = small_corpus
        model = NGramLM(n=2, smoothing="kneser_ney").fit(sentences, vocab=vocab)

        context = ("woezɔ",)
        prob_seen = model.probability("loo", context)
        prob_unseen = model.probability("keta", context)

        assert prob_seen > prob_unseen
        assert prob_unseen > 0.0

    def test_perplexity_computation(self, small_corpus):
        sentences, vocab = small_corpus
        model = NGramLM(n=2, smoothing="laplace", k=1.0).fit(sentences, vocab=vocab)

        test_sentences = [["woezɔ", "loo"], ["ama", "fle"]]
        ppl = model.perplexity(test_sentences)
        assert ppl > 1.0
        assert not math.isinf(ppl)


def test_unknown_words_pay_spelling_cost_per_word():
    # "keta" occurs once in training, so it becomes <unk>; the per-word score must charge for spelling it
    train = ["woezɔ loo kofi", "woezɔ loo ama", "kofi yi suku", "ama yi suku", "kofi yi keta"]
    rows = run_ngram_experiment(train, ["kofi yi suku"], ["woezɔ loo keta"], UnicodeWordTokenizer(), max_order=2)
    assert rows[0]["oov_rate_pct"] > 0
    assert rows[0]["oov_spelling_nats"] > 0


def test_lstm_scoring_is_a_proper_distribution():
    # With a zeroed output layer every allowed id is equally likely, so perplexity must equal the number of
    # predictable ids: everything except <pad> and <s>
    import torch
    from src.section_b_lstm.lstm_lm import LSTMLM, build_index, encode, total_nll

    stoi, blocked = build_index({"<s>", "</s>", "<unk>", "woezɔ", "loo"})
    model = LSTMLM(len(stoi))
    torch.nn.init.zeros_(model.out.weight)
    torch.nn.init.zeros_(model.out.bias)
    nll, n = total_nll(model, encode([["woezɔ", "loo"], ["loo"]], stoi), blocked)
    assert n == 5  # 2 + 1 tokens, plus one </s> per sentence
    assert math.isclose(math.exp(nll / n), len(stoi) - 2, rel_tol=1e-6)  # torch computes in float32
