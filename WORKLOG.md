# Worklog — Prosit 1

Every session where someone works on this repo gets an entry here. **Add new entries at the top.**

---

## [2026-09-17] Repository Initialization & Scaffolding
- **Author**: Antigravity & Team
- **Assistant**: Antigravity
- **Branch**: `main`
- **What changed**:
  - Cloned remote repository from `https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git`.
  - Structured directories: `data/`, `notebooks/`, `src/`, `figures/`, `reports/`.
  - Implemented core statistical N-gram engine (`src/ngram.py`), preprocessing and unicode tokenization (`src/preprocessing.py`), domain adaptation wrapper (`src/domain_adaptation.py`), and evaluation benchmarks (`src/evaluation.py`, `src/viz.py`).
  - Created initial exploratory and experiment notebooks (`01_low_resource_ngram_lm.ipynb`, `02_domain_specific_llm_adaptation.ipynb`, `03_evaluation_and_comparisons.ipynb`).
  - Prepared full report templates in `reports/` for Section A (Theory, 14 questions with exact space constraints), Section B (Low-Resource Model), Section C (Domain Adaptation), and the 10-minute presentation slides outline.
  - Added `.gitignore`, `requirements.txt`, `RULES.md`, `CLAUDE.md`, and `README.md`.
- **Decisions made**:
  - Adopted modular design: algorithmic logic lives in `src/` to ensure notebooks remain clear, reproducible narratives.
  - Implemented multiple smoothing options (MLE, Laplace, Lidstone, Linear Interpolation, Kneser-Ney) to enable rigorous empirical comparison as required by rubric.
- **Blockers / Next steps**:
  - Group alignment on chosen African language corpus (e.g. Akan/Twi, Ewe, Yoruba) and English domain corpus (e.g. agriculture, clinical medicine).
