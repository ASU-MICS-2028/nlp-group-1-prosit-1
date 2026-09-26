# Prosit 1: Building and Adapting Language Models

ICS554 Natural Language Processing · MICS 2028 · Group 1

**The Scenario.** We are working as an intern engineering team at **Ankora**, an AI research lab based in Ghana specializing in speech recognition systems. An essential component of specialized ASR pipelines is effective language modeling. 

Ankora has tasked us with two core deliverables:
1. Build a specialized language model for a **low-resource African language** using statistical n-gram techniques rather than neural models due to severe data scarcity.
2. Build an **English-based language model adapted to a specialized domain** (e.g., healthcare, agriculture, or finance) and demonstrate empirical proof of learning and domain effectiveness.

---

## Three Models, Kept Apart

The repository holds three separate models. Each one has its own code folder and its own results folder.

| | 1. N-gram | 2. LSTM baseline | 3. Fine-tuned LLM |
|---|---|---|---|
| **Report section** | B (the main model) | B, Question 2 only | C |
| **Language and data** | Ewe: 123,511 sentences from four sources | the same Ewe sentences, cut into the same BPE tokens as the n-gram | English: 2,212 agricultural questions with answers |
| **What it is** | counts plus smoothing (interpolated Kneser-Ney); no neural network | a small neural network trained from scratch | pretrained distilgpt2 (82M parameters), 0.18% of it trained with LoRA |
| **Why it exists** | the Section B deliverable | to answer Section B, Question 2: are n-grams better than neural models for a low-resource language? | the Section C deliverable |
| **Code** | `src/section_b_ngram/` | `src/section_b_lstm/` | `src/section_c_llm/` |
| **Results** | `results/section_b_ngram/` | `results/section_b_lstm/` | `results/section_c_llm/`, adapters in `models/section_c_llm/` |
| **Notebooks** | `b1_ngram_smoothing`, `b2_ngram_tokenizers` | shown in `summary_all_models` | `c1_llm_finetuning` |

**The LSTM and the LLM are not related.** The LSTM exists only to answer Section B's n-gram-versus-neural question on Ewe; the LLM is Section C's English model. They share no code, data or weights. The LSTM does reuse the n-gram's data, tokenizer and scoring, on purpose, so the two Ewe models can be compared number for number. `notebooks/summary_all_models.ipynb` shows all three side by side, read straight from the result files.

---

## Deliverables & Links

- **GitHub Repository**: [https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git](https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git)
- **Technical Report**: Divided into Section A (Theory), Section B (Low-Resource Model), and Section C (Domain Model). Submitted to Canvas.
- **10-Minute Group Presentation**: Summarizing Sections B and C. Slides submitted to Canvas.
- **klenam.ai Submission**: Repository link submitted on klenam.ai.

---

## Setup & Environment

```bash
git clone https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git
cd nlp-group-1-prosit-1

# Create and activate a Python 3.12 virtual environment (the pins in requirements.txt need 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Verify setup:
```bash
python -c "import torch, transformers, datasets, peft; print('Environment ready!')"
python -m pytest
```

Open the notebooks with `jupyter lab notebooks/`.

## Reproducing Every Number

All report and slide numbers come from files these commands write (see `reports/claims_table.md`). Run them from the repository root:

```bash
# 1. N-gram (Section B)
python -m src.section_b_ngram.build_datasets           # Ewe splits from the raw files (see data/README.md)
python -m src.section_b_ngram.run_sweep --dataset all  # 5 tokenizers x N=1..6 -> results/section_b_ngram/ (hours on a laptop CPU)

# 2. LSTM baseline (Section B, Question 2); reads the n-gram results above
python -m src.section_b_lstm.run_baseline --dataset 2 --config small --max-epochs 200 --patience 3  # 420 sentences (minutes)
python -m src.section_b_lstm.run_baseline --dataset unified --config large --max-epochs 10          # full corpus (~3 h per seed)

# 3. Fine-tuned LLM (Section C)
python -m src.section_c_llm.prepare_data        # KisanVaani splits, one row per question
python -m src.section_c_llm.train_lora          # both adapters -> models/section_c_llm/, scores -> results/section_c_llm/ (~15 min)
python -m src.section_c_llm.benchmark_decoding  # decoding benchmark -> results/section_c_llm/decoding_benchmark.json

# The 10-minute presentation, rebuilt from the result files above
python presentation/build_deck.py               # -> presentation/Prosit1_Language_Models.pptx
```

---

## Repository Layout

```
.
├── src/
│   ├── section_b_ngram/          # 1. Ewe n-gram models (Section B)
│   │   ├── data_pipeline.py      #    clean, deduplicate and split the Ewe sources
│   │   ├── build_datasets.py     #    raw files -> data/processed/ (run first)
│   │   ├── preprocessing.py      #    special tokens, vocabulary from training only, <unk>
│   │   ├── ewe_tokenizers.py     #    Whitespace, Unicode Word, Ewe Stemmer, BPE, Character
│   │   ├── ngram.py              #    MLE, Laplace, interpolation, Kneser-Ney
│   │   ├── experiment_runner.py  #    one tokenizer, N=1..6, perplexity per token and per word
│   │   ├── run_sweep.py          #    5 tokenizers x N=1..6 on each dataset
│   │   └── viz.py                #    plots for notebook b1
│   ├── section_b_lstm/           # 2. LSTM baseline (Section B, Question 2)
│   │   ├── lstm_lm.py            #    the network, its training loop and scoring
│   │   └── run_baseline.py       #    LSTM vs Kneser-Ney on the same tokens, 3 seeds
│   └── section_c_llm/            # 3. distilgpt2 + LoRA (Section C)
│       ├── prepare_data.py       #    KisanVaani, one row per question, 80/10/10
│       ├── train_lora.py         #    trains both adapters, scores base, standard and masked
│       └── benchmark_decoding.py #    decoding settings vs repetition
├── results/                      # every number in the reports comes from a file here
│   ├── section_b_ngram/          #    dataset_1..4.json, unified.json, two figures
│   ├── section_b_lstm/           #    lstm_vs_ngram.json
│   └── section_c_llm/            #    lora_results.json, decoding_benchmark.json, lora_perplexity.png
├── models/section_c_llm/         # the two LoRA adapters, standard/ and masked/ (see models/README.md)
├── notebooks/
│   ├── b1_ngram_smoothing.ipynb  # smoothing methods compared on a 10,000-sentence sample
│   ├── b2_ngram_tokenizers.ipynb # tokenizers and N=1..6, from the sweep results
│   ├── c1_llm_finetuning.ipynb   # loads the adapters, re-scores them, samples answers
│   └── summary_all_models.ipynb  # all three models, from the result files
├── presentation/                 # the 10-minute group presentation
│   ├── Prosit1_Language_Models.pptx   # the deck (speaker notes: what to say, and each number's source)
│   ├── build_deck.py             # builds the deck; reads every number from results/ and data/processed/
│   ├── ashesi_presentation_red.pptx   # Ashesi Presentation Red template, as used for the ICS553 Prosit 1 deck
│   └── presentation_outline.md   # the slide plan and timings
├── reports/                      # the written deliverables
│   ├── section_a_theory.md       # theory (14 questions with space constraints)
│   ├── section_b_low_resource_lm.md   # Section B report: n-gram, plus the LSTM in Question 2
│   ├── section_c_domain_adaptation.md # Section C report: distilgpt2 + LoRA
│   ├── claims_table.md           # every quoted number -> the file, key and script behind it
│   ├── datasheet.md              # Gebru et al. datasheet for the Ewe and agriculture corpora
│   ├── quiz_revision_guide.md    # viva quiz revision guide
│   └── LEARNING_JOURNAL.md       # team journal: what we built, what we found, what we fixed
├── data/                         # raw and processed data, gitignored except stats.json (see data/README.md)
├── tests/test_pipeline.py        # unit tests (python -m pytest)
├── METHODOLOGY_GUIDE.md          # best practices playbook & instructions
├── WORKLOG.md                    # shared AI-use and contribution log
├── RULES.md                      # team working agreement & code standards
├── CLAUDE.md                     # AI assistant constraints & context
├── pytest.ini
└── requirements.txt
```

---

## Team Roles & Ownership

| Seat | Technical Ownership | PBL Role |
| --- | --- | --- |
| **Statistical Modeler** | N-gram implementations, smoothing algorithms, vocabulary OOV policy | Chairperson |
| **Neural Adaptation Lead** | LoRA/PEFT pipeline, base model selection, training loop | Secretary |
| **Evaluation & Benchmarks** | Perplexity calculations, loss tracking, comparison tables | Scribe |
| **Data & Ethics Lead** | Corpus acquisition, orthography tokenization, Section A & C writeups | Steward |

---

## Conventions & Working Rules

See [`RULES.md`](RULES.md) for the complete rules. In summary:
- **No data leakage**: Vocabularies and parameter thresholds are induced strictly on the training partition.
- **Reproducibility**: Explicit random seeds (`RANDOM_SEED = 42`) and relative paths (`ROOT` in `src/__init__.py`, derived from `Path(__file__)`).
- **Clean notebooks**: Strip notebook output cells before committing to avoid git conflicts (`nbstripout`).
- **AI-use declarations**: Log every assistant session at the top of [`WORKLOG.md`](WORKLOG.md) before pushing.
- **Branch per task**: Feature branches with peer review before merging into `main`.
