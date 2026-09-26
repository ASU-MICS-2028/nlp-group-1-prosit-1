# Prosit 1: Building and Adapting Language Models

ICS554 Natural Language Processing, Ashesi University. Two models: an **n-gram language model for Ewe** (a low-resource language), compared against neural models trained from scratch, and an **English model adapted to a domain** (health and agriculture) by fine-tuning a pretrained LLM.

## Start here

`main.ipynb` is the whole project, top to bottom, with every output saved. Stages 0–3 (data audit, the worked example, the n-gram counter, smoothing and tokenisation) run inside the notebook in about an hour. Stages 4–5 (neural models, domain adaptation) and the follow-up experiments took many GPU-hours; their scripts save to `results/`, and the notebook reads those files, so opening the notebook needs no training.

## Files

| | |
|---|---|
| `stage0_audit.py` | Cleaning, the Ewe filter, and the train/dev/test splits (used by the notebook) |
| `stage2_ngram.py` | The n-gram counter, MLE probabilities, perplexity and generation, written from scratch |
| `stage3_smoothing.py` | Six smoothing methods (add-k, Good-Turing, interpolation, Katz, Kneser-Ney, modified Kneser-Ney), perplexity and bits per character |
| `bpe.py` | BPE tokenisers, learned from the training split |
| `stage4_neural.py` | LSTM and transformer from scratch; the n-gram-vs-neural scaling curve; width and tokeniser sweeps; the English side |
| `stage4_english.py`, `stage4_wikitext.py` | English as a high-resource language: the same pipeline on the English side of the corpus and on WikiText-103 |
| `stage5_domain.py`, `stage5_generate.py` | Domain adaptation of Qwen2.5-0.5B and SmolLM2-135M with LoRA, DoRA and full fine-tuning; seed repeats; generated text |
| `stage0_splits.py`, `stage0_ablation.py`, `stage0_filter_test.py`, `stage2_samples.py`, `stage3_extra.py` | Follow-up experiments: every split ratio, every cleaning step left out, the filter graded by a language identifier, and more |
| `notebook_stage45.py`, `notebook_extras.py` | Rebuild the notebook's Stage 4/5 and follow-up sections from `results/` |
| `build_slides.py` | Builds `presentation.pptx` from `results/` |
| `results/` | Every table, chart and generated sample the report cites |

## Running it

```
uv sync                                   # Python 3.14, dependencies from pyproject.toml
# put the dataset at data/train-00000-of-00001.parquet (HuggingFace: ghananlpcommunity/english-ewe-sentence-pairs-4m)
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace main.ipynb   # Stages 0-3, ~1 hour
uv run python stage4_neural.py            # hours on a GPU; resumes from results/
uv run python stage5_domain.py            # Apple Silicon only (mlx-lm); base models go in data/models/
```

`data/` is not in the repository (about 20 GB: splits, tokenisers, base models, adapters, WikiText). The scripts recreate everything except the trained LSTM and transformer weights, which were scored and discarded.

Stage 3.5b compares our Kneser-Ney with KenLM, built from source into `.tools/kenlm`; the cell skips itself if KenLM is absent.

## Team contributions

`contrib/eric/` holds Eric Elikplim Sunu's parallel implementation of the same prosit (his commits are in this repository's history). It is self-contained and runnable (`cd contrib/eric && python -m pytest`), and it covers ground the main project does not:

- **A four-source Ewe corpus** (123,511 sentences), including University of Ghana Waxal speech transcriptions, the only conversational Ewe in either project.
- **A rule-based Ewe stemmer tokenizer** that keeps affixes as tokens (2.6% better per word than plain words).
- **Section C on distilgpt2**: standard vs **prompt-masked** LoRA loss, **answer-only perplexity**, and a **decoding benchmark** (repetition penalty, n-gram blocking, Distinct-3).
- **A datasheet** (Gebru et al.) recording personal data and unverified licences in the Ewe sources, a claims table tying every number to its script, and 22 unit tests.

The two projects reached the same conclusions independently: the Ð/Ɖ look-alike letter, the interpolation bug that loses probability on unseen contexts, per-word/per-character comparison with unknown words charged for spelling, BPE as the best tokenizer, and a crossover (not a verdict) between n-gram and neural models as data grows. His finding that the agriculture corpus repeats questions led us to re-score our agriculture models on a leak-free test set (`rescore_agri.py`, `results/agriculture_leakfree.json`); the result was unchanged.
