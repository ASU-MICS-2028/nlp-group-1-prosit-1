"""
Unit test suite for low-resource Ewe language modeling pipeline.
Validates Unicode normalization, 5 tokenizers, probability conservation,
and N-gram smoothing algorithms.
"""

import math
import pytest
from src.preprocessing import normalize_ewe_text, basic_tokenize, build_vocabulary, replace_oov_tokens
from src.ewe_tokenizers import (
    WhitespaceTokenizer,
    UnicodeWordTokenizer,
    CharacterTokenizer,
    EweRuleStemmerTokenizer,
    SimpleBPETokenizer,
)
from src.ngram import NGramLM
from src.data_pipeline import clean_and_normalize_ewe_sentence


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
        # Plural suffix -wo stripped
        res_plural = tok.tokenize("atíwo")
        assert "atí" in res_plural

        # Subject pronoun prefix mí- stripped
        res_prefix = tok.tokenize("míewɔ")
        assert "wɔ" in res_prefix

    def test_character_tokenizer(self):
        tok = CharacterTokenizer()
        tokens = tok.tokenize("Ewe")
        assert tokens == ["e", "w", "e"]

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
        total_prob = sum(model.probability(w, context) for w in model.vocab)
        assert math.isclose(total_prob, 1.0, rel_tol=1e-5)

    def test_lidstone_probability_conservation(self, small_corpus):
        sentences, vocab = small_corpus
        model = NGramLM(n=2, smoothing="laplace", k=0.1).fit(sentences, vocab=vocab)

        context = ("kofi",)
        total_prob = sum(model.probability(w, context) for w in model.vocab)
        assert math.isclose(total_prob, 1.0, rel_tol=1e-5)

    def test_interpolation_probability_conservation(self, small_corpus):
        sentences, vocab = small_corpus
        model = NGramLM(n=3, smoothing="interpolation").fit(sentences, vocab=vocab)
        model.set_interpolation_weights([0.2, 0.3, 0.5])

        context = ("woezɔ", "loo")
        total_prob = sum(model.probability(w, context) for w in model.vocab)
        assert math.isclose(total_prob, 1.0, rel_tol=1e-4)

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
