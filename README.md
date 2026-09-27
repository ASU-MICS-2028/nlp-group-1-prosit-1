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
| `stage0_sources.py` | Three more Ewe sources (Bible/JW sentence pairs, a dictionary export, Waxal **spoken** Ewe transcriptions): cleaned, added to training, and scored separately |
| `ewe_stemmer.py`, `stage3_stemmer.py` | A rule-based Ewe tokenizer that splits affixes off words and keeps them as tokens, and its fair comparison |
| `stage5_distilgpt2.py` | A CPU-sized second Section C experiment: distilgpt2 + LoRA with the loss on all tokens vs on answers only, answer-only perplexity, and a decoding benchmark; its adapters are in `models/distilgpt2_agriculture/` |
| `stage0_splits.py`, `stage0_ablation.py`, `stage0_filter_test.py`, `stage2_samples.py`, `stage3_extra.py`, `rescore_agri.py` | Follow-up experiments: every split ratio, every cleaning step left out, the filter graded by a language identifier, a leak-free re-score of the agriculture models, and more |
| `notebook_stage45.py`, `notebook_extras.py` | Rebuild the notebook's Stage 4/5 and follow-up sections from `results/` |
| `build_slides.py` | Builds `presentation.pptx` from `results/` |
| `results/` | Every table, chart and generated sample the report cites |
| `DATASHEET.md` | Where every corpus comes from, what is in it, personal data, licences and limits |
| `tests/` | Fast checks of the core logic (`uv run --with pytest python -m pytest`) |

## Running it

```
uv sync                                   # Python 3.14, dependencies from pyproject.toml
# put the dataset at data/train-00000-of-00001.parquet (HuggingFace: ghananlpcommunity/english-ewe-sentence-pairs-4m)
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace main.ipynb   # Stages 0-3, ~1 hour
uv run python stage4_neural.py            # hours on a GPU; resumes from results/
uv run python stage5_domain.py            # Apple Silicon only (mlx-lm); base models go in data/models/
uv run python stage0_sources.py           # the three extra Ewe sources go in data/raw/ (see DATASHEET.md)
```

`stage5_distilgpt2.py` was run in its own environment (Python 3.12, `requirements-distilgpt2.txt`); its data split is reproducible with `uv run python stage5_distilgpt2.py prepare`.

**Data and models.** Small files are in the repository under `data/`: the BPE tokenisers (`data/bpe/`, `data/wikitext/bpe_4000.json`), the English Section C splits (`data/domain/*/`) and the SmolLM2-135M adapters (`data/domain/adapters/SmolLM2-135M/`). The big files are on Google Drive: https://drive.google.com/drive/folders/1MsJ24hAK_nxFcqwV6GUNiWeM7IZ3ic6B

| Drive folder | What | Put it at |
|---|---|---|
| `ewe_ngram/` | The final Ewe model: modified Kneser-Ney, order 5, ARPA format (383 MB) | `data/models/` |
| `ewe_splits/` | The 15 frozen train/dev/test splits of the cleaned Ewe corpus (not committed: the Ewe sources' licences are unclear) | `data/splits/` |
| `domain_adapters/Qwen2.5-0.5B/` | The Qwen2.5-0.5B adapters: LoRA rank 2/8/32, DoRA, full fine-tuning, seed repeats (644 MB) | `data/domain/adapters/Qwen2.5-0.5B/` |

The scripts can also recreate all of these. Base models, the raw Ewe corpus and WikiText-103 are downloaded from their sources (see `DATASHEET.md`). The trained LSTM and transformer weights were scored and discarded.

Stage 3.5b compares our Kneser-Ney with KenLM, built from source into `.tools/kenlm`; the cell skips itself if KenLM is absent.
