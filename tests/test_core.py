"""Fast checks of the logic every result depends on. Run: uv run --with pytest python -m pytest"""
import math

import ewe_stemmer
import stage0_audit
import stage2_ngram
import stage3_smoothing
from stage0_sources import clean_extra


def test_cleaning_and_splits():
    stage0_audit.selfcheck()  # Ð→Ɖ, whitespace, Ewe filter, mojibake rejection, near-duplicates share a bucket


def test_counting_matches_the_hand_calculation():
    stage2_ngram.selfcheck()  # bigram probabilities 2/3 and 1/2, perplexity 30^(1/5), unseen pair = infinity


def test_every_smoothing_method_is_a_distribution():
    stage3_smoothing.selfcheck()  # sums to 1 over the vocabulary, seen and unseen contexts, both unknown-word modes


def test_extra_source_cleaning():
    assert clean_extra("Ðasefowo ðe nya") == "Ɖasefowo ɖe nya"            # look-alike letters
    assert clean_extra("<b>Mawu</b> le https://x.org afi") == "Mawu le afi"  # markup and URLs removed
    assert clean_extra("nya​ la") == "nya la"                            # zero-width characters removed
    assert clean_extra("\\E7\\A3 bad row") is None                           # escaped binary rows dropped
    assert clean_extra("va") is None                                          # fewer than two words


def test_stemmer_splits_affixes_and_loses_nothing():
    assert ewe_stemmer.split_word("nusrɔ̃lawo") == ["nu+", "srɔ̃la", "+wo"]
    assert ewe_stemmer.split_word("va") == ["va"]  # short words are left alone
    pieces = ewe_stemmer.tokenize("Nusrɔ̃lawo va.")
    rebuilt = "".join(p.strip("+") for p in pieces if p.isalpha() or "+" in p or any("̀" <= ch <= "ͯ" for ch in p))
    assert rebuilt == "nusrɔ̃lawova"  # every character of the words survives, affixes included
    assert "ɔ̃" in "".join(pieces)     # the combining tilde is kept with its letter


def test_stemmer_plugs_into_the_models():
    name = ewe_stemmer.register()
    m = stage3_smoothing.Smoothed(["nusrɔ̃lawo va afi sia", "míewɔ dɔ la"], 2, name, method="mkn")
    assert math.isfinite(m.evaluate(["nusrɔ̃la va"])[0])
