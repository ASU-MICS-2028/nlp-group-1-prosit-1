# Team Rules — Prosit 1 (Language Models & Adaptation)

Read this before your first commit. These rules exist so our team can work together effectively on one repository without breaking reproducibility, introducing data leakage, or encountering painful git conflicts.

---

## 1. Data Integrity & Leakage Prevention

The grading rubric directly tests our pipeline discipline and statistical validity:

- **Split first, build vocabulary second**: Never build a vocabulary or compute word counts on data that includes the validation or test sets. Out-of-vocabulary (OOV) tokens must be mapped to `<unk>` based strictly on training set frequencies.
- **Never fit on the test set**: Hyperparameters (such as smoothing parameter $k$ in Lidstone or interpolation weights $\lambda$) must be tuned on validation splits, never on test splits.
- **Data files stay out of git**: Large raw datasets (`data/raw/`) must remain gitignored. Commit only small, reproducible synthetic test fixtures or scripts to fetch data.

---

## 2. Git & Commit Hygiene

- **Branch per task**: `feature/ngram-kneser-ney`, `feature/lora-finetune`, `fix/oov-tokenization`. Never commit directly to `main`.
- **Pull before you start**: Run `git pull origin main` at the start of every session.
- **Clear notebook outputs before commit**: Output cells create severe merge conflicts and bloat git history. Run `nbstripout` or `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`.
- **Peer review**: At least one other team member must review and approve code before merging into `main`.

---

## 3. Code Standards

- **Python 3.11**: Keep dependencies pinned in `requirements.txt`.
- **Formatting**: `black` for formatting and `ruff` for linting.
- **Relative paths only**: Always use `Path(__file__).parent` or `Path.cwd()`. Never hardcode absolute paths like `/Users/...` or `C:\...`.
- **No magic numbers**: Always define `RANDOM_SEED = 42` at the top of every script and notebook, passing it to splits, sampling, and model initializers.
- **Logic lives in `src/`**: Notebooks should read as high-level scientific narratives: import, call, visualize, explain. Cells longer than 25 lines belong in a module under `src/`.

---

## 4. Working with AI Assistants & Academic Integrity

- **Own every line**: You will be examined individually during the Viva Quiz (35% of total grade). If you cannot explain the mathematical derivation or code implementation to the examiners, do not commit it.
- **Log every session**: Every session utilizing an AI assistant must be recorded at the top of `WORKLOG.md` before pushing.
- **Focus on understanding**: Use AI to clarify theoretical concepts (e.g. why Kneser-Ney uses continuation probabilities, or how LoRA gradient rank works) rather than blindly generating code.

---

## 5. The Worklog

`WORKLOG.md` is our unified audit trail. New entries are placed at the **top** using the specified format (date, author, AI tool used, branch, summary of changes, decisions made, and blockers).
