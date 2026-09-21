# Prosit 1 — Building and Adapting Language Models

ICS554 Natural Language Processing · MICS 2028 · Group 1

**The Scenario.** We are working as an intern engineering team at **Ankora**, an AI research lab based in Ghana specializing in speech recognition systems. An essential component of specialized ASR pipelines is effective language modeling. 

Ankora has tasked us with two core deliverables:
1. Build a specialized language model for a **low-resource African language** using statistical n-gram techniques rather than neural models due to severe data scarcity.
2. Build an **English-based language model adapted to a specialized domain** (e.g., healthcare, agriculture, or finance) and demonstrate empirical proof of learning and domain effectiveness.

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

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Verify setup:
```bash
python3 -c "import torch, transformers, datasets, nltk, peft; print('Environment ready!')"
```

---

## Repository Layout

```
.
├── data/
│   ├── raw/                  # Raw text corpora (gitignored)
│   ├── processed/            # Cleaned, tokenized text splits
│   └── README.md             # Dataset documentation & provenance
├── notebooks/
│   ├── 01_low_resource_ngram_lm.ipynb         # Section B pipeline & experiments
│   ├── 02_domain_specific_llm_adaptation.ipynb # Section C LoRA fine-tuning
│   ├── 03_evaluation_and_comparisons.ipynb     # Benchmarks & comparison tables
│   └── 04_tokenization_and_ngram_ablation.ipynb # Comprehensive 5-tokenizer sweep
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      # Unified multi-source corpus loader & normalizer
│   ├── ngram.py              # Statistical LM: MLE, Laplace, Interpolation, Kneser-Ney
│   ├── preprocessing.py      # Unicode tokenization, OOV handling, vocabulary split
│   ├── tokenizers.py         # 5 tokenizers: Whitespace, Word, Stemmer, BPE, Char
│   ├── domain_adaptation.py  # HuggingFace & PEFT/LoRA fine-tuning utilities
│   ├── evaluation.py         # Perplexity, cross-entropy, markdown table formatting
│   ├── experiment_runner.py  # Automated ablation experiment harness
│   └── viz.py                # Perplexity & frequency distribution plotting
├── figures/                  # Exported plots for reports and presentation slides
├── reports/
│   ├── section_a_theory.md   # Theoretical foundations (14 questions with space constraints)
│   ├── section_b_low_resource_lm.md  # Team report on African LM (Ewe / Èʋegbe)
│   ├── section_c_domain_adaptation.md # Team report on domain-adapted English LM
│   ├── claims_table.md       # Claims traceability table (14 claims verified)
│   ├── datasheet.md          # Gebru et al. datasheet for Ewe Mega-Corpus
│   ├── presentation_outline.md # 10-minute presentation slide outline (6 slides)
│   ├── quiz_revision_guide.md # Viva quiz revision guide for klenam.ai
│   └── LEARNING_JOURNAL.md   # Detailed experimental journal & reflections
├── scripts/
│   ├── launch_notebook.sh    # 1-click browser JupyterLab launcher
│   └── run_multi_tokenizer_ablation.py # Headless multi-tokenizer benchmark runner
├── tests/
│   └── test_pipeline.py      # Automated unit tests (13 tests passing)
├── METHODOLOGY_GUIDE.md      # Best practices playbook & instructions
├── WORKLOG.md                # Shared AI-use and contribution log
├── RULES.md                  # Team working agreement & code standards
├── CLAUDE.md                 # AI assistant constraints & context
├── pytest.ini                # Pytest configuration
├── requirements.txt
└── README.md
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
- **Reproducibility**: Explicit random seeds (`RANDOM_SEED = 42`) and relative paths (`Path(__file__).parent`).
- **Clean notebooks**: Strip notebook output cells before committing to avoid git conflicts (`nbstripout`).
- **AI-use declarations**: Log every assistant session at the top of [`WORKLOG.md`](WORKLOG.md) before pushing.
- **Branch per task**: Feature branches with peer review before merging into `main`.
