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

---

## 2026-09-17 · 16:40–16:55 GMT · Eric Elikplim Sunu

**Branch:** feature/repo-setup
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
