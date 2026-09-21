# Worklog

Shared record of who did what, when, and with which tools.

**Rules**

- Newest entry at the **top**, directly under this block.
- Append only. Never edit or delete someone else's entry. If an earlier entry
  turned out to be wrong, write a new one correcting it.
- One entry per working session where you changed anything in the repo.
- Write it **before you push**, not at the end of the week.
- If an AI assistant wrote or modified code that ended up in the repo, it goes
  in the `Assistant` field. This is what your individual AI declaration is
  built from.

**Template** — copy this block, fill it in, paste it at the top.

```markdown
## YYYY-MM-DD · HH:MM–HH:MM · Your name

**Branch:** feature/...
**Assistant:** none | Claude / ChatGPT / Copilot / Gemini / other — and what you used it for
**Did:**
- ...
**Decided:**
- ...
**Blocked / open questions:**
- ...
**Next:**
- ...
```

Field notes:

- **Assistant** — be specific and honest. "Gemini, to debug Kneser-Ney smoothing backoff and draft the mathematical derivations in Section A" is useful. "Used AI" is not.
- **Decided** — only real decisions, the kind someone might otherwise reverse without knowing. Leave it out if nothing was decided.
- **Blocked** — this is the field that saves the project. Write it even when it feels like admitting you're stuck. Especially then.

## 2026-09-21 · 12:40–12:50 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to preprocess Dataset 3 (`selected transcribed audios.xlsx` — 19,152 spoken audio transcriptions from Univ of Ghana Waxal project), run the 5-tokenizer ablation sweep ($N=1\dots 6$), document the oral domain perplexity dynamics in `reports/LEARNING_JOURNAL.md`, and output `reports/results_dataset_3_all_tokenizers.json`.
**Did:**
- Extracted, normalized (NFC), and deduplicated 19,151 spoken Ewe transcriptions into `data/processed/dataset_3_speech/` (15,320 train / 1,916 test / 508k train words).
- Updated `scripts/run_multi_tokenizer_ablation.py` with `--dataset 3` configuration.
- Executed full 5-tokenizer ablation ($N=1\dots 6$) on Dataset 3: Whitespace (Trigram PPL 390.0), Unicode Word (Trigram PPL 157.5), Ewe Stemmer (Trigram PPL 137.5), BPE (5-gram PPL 17.8), Character (6-gram PPL 5.5).
- Logged cross-domain analysis in `reports/LEARNING_JOURNAL.md` comparing written literary language (Dataset 1) vs oral descriptive speech (Dataset 3).
**Decided:**
- Identified that oral speech has a more concentrated high-frequency core vocabulary, yielding lower Unigram perplexity (492.7 vs 663.5), while phonetic transcriptions benefit heavily from morphological stemming ($157.5 \to 137.5$ PPL reduction).
**Next:**
- Run Phase 4: Dataset 4 (`ewe_corpus.parquet` — 4.4M web/aligned sentences).

---

## 2026-09-21 · 12:05–12:15 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to engineer `scripts/run_multi_tokenizer_ablation.py` and run the full 5-tokenizer matrix (Whitespace, Unicode Word, Ewe Stemmer, BPE, Character) across $N=1\dots 6$ on both Dataset 1 (`EWE_ENGLISH.csv`) and Dataset 2 (`eweenglishsentence.json`), documenting breaking point shifts and saving structured evaluation JSONs.
**Did:**
- Fixed combining diacritic compatibility in `src/tokenizers.py` for `EweRuleStemmerTokenizer` where Python `isalnum()` falsely classified words with combining marks (`\u0300-\u036f`) as punctuation.
- Created `scripts/run_multi_tokenizer_ablation.py` to automate simultaneous 5-tokenizer evaluations across $N=1\dots 6$.
- Executed full 5-tokenizer sweep on Dataset 2 (Micro-Data: 420 train sents): Whitespace (Bigram PPL 19,468), Unicode Word (Bigram PPL 3,098), Ewe Stemmer (Bigram PPL 2,539), BPE (Trigram PPL 40.8), Character (5-gram PPL 9.8).
- Executed full 5-tokenizer sweep on Dataset 1 (Cultural Stories: 21,275 train sents): Whitespace (Trigram PPL 476.9), Unicode Word (Trigram PPL 139.4), Ewe Stemmer (Trigram PPL 127.4), BPE (5-gram PPL 14.0), Character (6-gram PPL 7.0).
- Updated `reports/LEARNING_JOURNAL.md` with comprehensive 5-tokenizer summary tables and cross-dataset breaking point analysis.
**Decided:**
- Verified that BPE subwords grant +2 orders of contextual headroom before breaking ($N=5$ on Dataset 1, $N=3$ on Dataset 2) compared to word-level models.
- Established that morphological stemming consistently beats raw word tokenization for Ewe by merging inflected forms into unified root counts.
**Next:**
- Run the full 5-tokenizer sweep on Dataset 3 (`selected transcribed audios.xlsx` — 19,152 spoken audio transcriptions).

---

## 2026-09-17 · 21:30–21:42 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to configure the dedicated Python 3.14 virtual environment, install dev and analysis dependencies (`ruff`, `black`, `ipykernel`), register the Jupyter kernel `nlp-prosit-1`, and create workspace settings for Zed IDE (`.zed/settings.json`, `pyrightconfig.json`).
**Did:**
- Created and validated `.venv` virtual environment in `nlp-group-1-prosit-1/` using Python 3.14.
- Installed code formatting and linting utilities (`ruff`, `black`) and interactive execution support (`ipykernel`).
- Registered system-wide Jupyter kernelspec: `Python (NLP Prosit 1 - Ewe)` (`nlp-prosit-1`).
- Configured Zed IDE workspace settings in `.zed/settings.json` and parent directory, binding Python LSP (`pyright`) directly to `.venv/bin/python` with `extraPaths: ["src"]` and format-on-save via `ruff`.
- Created `pyrightconfig.json` defining `venvPath: "."` and `venv: ".venv"`.
**Decided:**
- Use Zed's native Pyright LSP integration pointing to the project's local `.venv` to ensure zero import errors and seamless intellisense across `src/` modules.
**Next:**
- Begin interactive step-by-step masterclass: Step 2 Zero-Probability Dilemma & Smoothing with live Ewe examples.
- Drop in the 4 Ewe datasets and run the harmonization pipeline.

---

## 2026-09-17 · 20:35–20:55 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to implement an experimental exploration suite for deep learning: 5 tokenization strategies (Whitespace, Unicode NFC Word, Character, Ewe Rule Stemmer, Byte-Pair Encoding BPE), N-gram scaling ablation from $N=1$ to $N=6$, multi-source dataset harmonization pipeline (`src/data_pipeline.py`) supporting 4+ datasets, interactive laboratory notebook (`04_tokenization_and_ngram_ablation.ipynb`), and the master reflective journal (`reports/LEARNING_JOURNAL.md`).
**Did:**
- Built `src/tokenizers.py` containing 5 tokenization implementations: `WhitespaceTokenizer`, `UnicodeWordTokenizer` (with combining diacritic regex `[\w\u0300-\u036f]+`), `CharacterTokenizer`, `EweRuleStemmerTokenizer` (stripping Ewe affixes), and `SimpleBPETokenizer` (native Byte-Pair Encoding subword learner).
- Discovered and resolved the Combining Diacritic Tokenization Trap where standard `\w+` split `"Nusrɔ̃lawo"` into 3 pieces (`['Nusrɔ', '̃', 'lawo']`), solving it with Unicode NFC normalization and explicit `\u0300-\u036f` range matching.
- Built `src/experiment_runner.py` with `run_ngram_experiment()` measuring vocabulary size $|V|$, total tokens, zero-count test sparsity rate (%), test perplexity, and text generation across $N \in [1, 2, 3, 4, 5, 6]$.
- Built `src/data_pipeline.py` with `merge_and_harmonize_datasets()` to ingest up to 4 disparate Ewe datasets, normalize to Unicode NFC, remove URLs/markup, deduplicate via normalized sentence hashing, and generate leak-free stratified splits (`train.txt`, `val.txt`, `test.txt`).
- Created `reports/LEARNING_JOURNAL.md` documenting the full PBL reflective learning cycle, breaking point analysis ($|V|^N$ combinatorial explosion at $N \ge 4$), problems encountered, and circumventions.
- Created `notebooks/04_tokenization_and_ngram_ablation.ipynb` as an interactive visual laboratory.
**Decided:**
- Identified $N=3$ as the empirical sweet spot for low-resource Ewe text: $N \ge 4$ causes test sparsity to exceed 92–99%, where Laplace smoothing degrades rapidly due to pseudo-count over-allocation.
- Multi-source Ewe datasets will be staged in `data/raw/low_resource/` and unified using `merge_and_harmonize_datasets()`.
**Next:**
- Drop the user's 4 Ewe datasets into `data/raw/low_resource/` and run the harmonization pipeline.
- Run the full tokenization and N-gram sweep on the unified corpus and populate final figures in `figures/`.

---

## 2026-09-17 · 19:55–20:05 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to rename the active working branch from `feature/repo-setup` to `eric` as the primary development playground branch.
**Did:**
- Renamed branch to `eric` (`git branch -m eric`).
**Decided:**
- Working playground and experimental development will live on personal branch `eric`.
**Next:**
- Place Ewe text dataset in `data/raw/low_resource/` and begin exploratory modeling.

---

## 2026-09-17 · 16:40–16:55 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to adapt low-resource language modeling pipeline for Ewe (Èʋegbe). Updated unicode tokenization and NFC normalization in `src/preprocessing.py` to preserve Ewe glyphs (`ɖ`, `ƒ`, `ɣ`, `ŋ`, `ɔ`, `ɛ`, `ʋ`) and tone markers. Updated `notebooks/01_low_resource_ngram_lm.ipynb`, `reports/section_b_low_resource_lm.md`, `reports/claims_table.md`, and `reports/datasheet.md` to center on Ewe.
**Did:**
- Enhanced `src/preprocessing.py` with `normalize_ewe_text()` and updated `basic_tokenize()` using Unicode NFC normalization, ensuring tone diacritics and distinct Ewe characters (`ɖ`, `ƒ`, `ɣ`, `ŋ`, `ɔ`, `ɛ`, `ʋ`) remain fused to base vowels without splitting into broken accent tokens.
- Added `load_corpus_from_file_or_hf()` supporting both local files (`data/raw/low_resource/ewe.txt`) and cloud streaming.
- Verified smoke test for Ewe tokenizer on complex phrases with bilabial and retroflex phonemes (`ŋutifafa`, `nuɖuɖu`, `woezɔ`, `dukɔa`).
- Updated `notebooks/01_low_resource_ngram_lm.ipynb` to model Ewe text with start token `<s>` and test smoothing techniques.
- Updated `reports/section_b_low_resource_lm.md` to document the linguistic characteristics of Ewe, corpus curation without liturgical skew, and n-gram vs. neural model trade-offs.
- Created `data/raw/low_resource/README.md` with guidelines on Ewe dataset formats and open sources (Menyo-20k, Masakhane, mC4).
**Decided:**
- Group consensus confirmed target African language as **Ewe (Èʋegbe)**.
- Tokenizer enforces Unicode NFC normalization to prevent character decomposition.
**Blocked / open questions:**
- Awaiting placement of raw Ewe dataset in `data/raw/low_resource/ewe.txt` to run full-scale count matrices and export final figures.
**Next:**
- Drop Ewe corpus into `data/raw/low_resource/ewe.txt`.
- Execute `notebooks/01_low_resource_ngram_lm.ipynb` and generate final perplexity plots in `figures/`.

---

## 2026-09-17 · 16:30–16:45 GMT · Eric Elikplim Sunu

**Branch:** feature/repo-setup
**Assistant:** Gemini (Gemini 3.8 Flash), to integrate the course lecture master blueprint and Gemini notebook context into the repository. Updated theoretical report Section A with lecture concepts (Jagged Intelligence, Sycophancy, Stochastic Parrots), aligned Section B to low-resource language modeling with cloud streaming and sample scaling (`SCALE_FACTOR = 0.05`), aligned Section C to the Agro-Extension agricultural corpus with LoRA attention adapters (`["q_proj", "v_proj"]`), created `reports/quiz_revision_guide.md` for the automated AI Viva Quiz on `clenam.ai`, and updated the course-level `SOLUTION_PLAN.md`.
**Did:**
- Enhanced `src/preprocessing.py` with flexible streaming capabilities to avoid Google Colab/local system crashes.
- Updated `reports/section_a_theory.md` with complete technical formulations, exact mathematical formulas, and insights on Jagged Intelligence (character masking via subword tokenization) and Sycophancy.
- Updated `reports/section_b_low_resource_lm.md` to document the African corpus selection, explicitly avoiding religious texts (Bible) to prevent liturgical skew, and defending the n-gram vs. neural model trade-off.
- Updated `reports/section_c_domain_adaptation.md` to detail the Agro-Extension text array corpus, evaluating the three approaches (from scratch, RAG, PEFT/LoRA) and justifying LoRA on attention layers.
- Created `reports/quiz_revision_guide.md` providing complete derivations and study answers for all 6 self-check questions from Section 6 of the course blueprint.
- Updated `notebooks/01_low_resource_ngram_lm.ipynb` and `notebooks/02_domain_specific_llm_adaptation.ipynb` to match the python blueprint specifications.
- Updated `reports/claims_table.md`, `reports/datasheet.md`, and course-level `SOLUTION_PLAN.md`.
**Decided:**
- Target domain confirmed as Agriculture (Agro-Extension text array) using LoRA ($r=8, \alpha=16$) targeting `["q_proj", "v_proj"]`.
- The Viva Quiz platform is verified as `clenam.ai` (Ashesi automated oral defense).
**Blocked / open questions:**
- None. Scaffolding, pipelines, and revision guides are fully aligned with course blueprints.
**Next:**
- Push `main` and `feature/repo-setup` to GitHub.
- Open Pull Request on GitHub from `feature/repo-setup` into `main`.

---

## 2026-09-17 · 13:55–14:20 GMT · Eric Elikplim Sunu

**Branch:** feature/repo-setup
**Assistant:** Gemini (Gemini 3.8 Flash), to inspect Machine Learning Prosit 1 (`mle-group-3-prosit-1`), replicate Ashesi MICS engineering standards, scaffold the repository structure (`src/`, `notebooks/`, `reports/`, `figures/`, `data/`), implement statistical N-gram algorithms and smoothing (MLE, Laplace, Lidstone, Linear Interpolation, Kneser-Ney), build unicode tokenization for African languages, set up PEFT/LoRA domain adaptation, compile initial report drafts for Section A, B, C, create `reports/claims_table.md`, and generate course-level `SOLUTION_PLAN.md` and interactive `prosit1-roadmap.html`. No licensed data rows surfaced.
**Did:**
- Cloned remote repository `https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git` into `Natural Language Processing/nlp-group-1-prosit-1`.
- Built course-level master strategy document `SOLUTION_PLAN.md` and interactive dashboard `prosit1-roadmap.html` matching the Machine Learning course design.
- Scaffolded modular source packages in `src/`:
  - `src/ngram.py`: Extensible N-gram LM supporting arbitrary orders $n$, Laplace ($+1$), Lidstone ($+k$), Linear Interpolation, and Interpolated Kneser-Ney smoothing with log-likelihood, perplexity, and temperature sampling.
  - `src/preprocessing.py`: Unicode-aware regex tokenizer preserving African language orthography and diacritics, vocabulary builder, and `<unk>` OOV replacer strictly on training partitions.
  - `src/domain_adaptation.py`: Hugging Face causal LM and PEFT/LoRA adapter setup with perplexity evaluation.
  - `src/evaluation.py` and `src/viz.py`: Benchmark table formatters and perplexity plotting routines.
- Scaffolded experiment notebooks in `notebooks/`:
  - `01_low_resource_ngram_lm.ipynb`: Section B statistical LM pipeline.
  - `02_domain_specific_llm_adaptation.ipynb`: Section C PEFT/LoRA adaptation pipeline.
  - `03_evaluation_and_comparisons.ipynb`: Benchmark tables, ablation studies, and slide figures.
- Scaffolded deliverables and templates in `reports/`:
  - `section_a_theory.md`: Complete theoretical foundations answering all 14 questions with exact paragraph constraints ($s$).
  - `section_b_low_resource_lm.md`: Section B team report template for African language LM.
  - `section_c_domain_adaptation.md`: Section C team report template for domain-adapted English model.
  - `claims_table.md`: Traceability table linking every figure on slides to specific notebook cells.
  - `datasheet.md`: Dataset provenance, composition, and collection datasheet based on Gebru et al.
  - `presentation_outline.md`: 10-minute slide deck outline with GitHub link on Slide 1.
- Initialized pinned `requirements.txt`, `.gitignore`, `RULES.md`, `CLAUDE.md`, and `README.md`.
**Decided:**
- Preprocessing enforces strict split-before-vocabulary hygiene: out-of-vocabulary words are replaced with `<unk>` using frequencies derived strictly from the training partition to prevent test set data leakage.
- LoRA ($r=8, \alpha=32$) is chosen over full fine-tuning for Section C to preserve general language competence and allow fast adaptation on consumer GPUs.
- All numbers presented in the technical report and slides must be cross-referenced in `reports/claims_table.md` to ensure defense readiness during the Viva Quiz (35%).
**Blocked / open questions:**
- Team consensus on the target low-resource African language (e.g., Akan/Twi vs. Yoruba vs. Ewe) and domain-specific corpus (e.g., Agriculture vs. Clinical Healthcare).
**Next:**
- Finalize selection of raw text corpora and download them into `data/raw/`.
- Run `01_low_resource_ngram_lm.ipynb` and export Section B perplexity curves into `figures/`.
- Run `02_domain_specific_llm_adaptation.ipynb` with selected domain corpus.
