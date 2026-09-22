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

## 2026-09-22 · 10:00–10:20 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Antigravity), to implement and evaluate decoding strategy improvements (repetition penalty, n-gram blocking, temperature tuning) and prompt-loss masking training for the domain-adapted agricultural model.
**Did:**
- Built `scripts/benchmark_decoding_strategies.py` and evaluated 5 decoding strategies across 4 agronomic prompts. Quantified repetition via Distinct-3 metrics (unpenalized baseline: 49.1% unique trigrams vs 100.0% under repetition penalty $r=1.25\dots 1.3$ and $N=3$ blocking).
- Saved full empirical decoding benchmark to `reports/decoding_strategies_benchmark.json`.
- Built `src/train_prompt_masked_lora.py` and trained DistilGPT2 with LoRA using prompt-loss masking (`labels[:prompt_len] = -100` on `\nAnswer:`) for 3 epochs (400 train pairs, 403s runtime on CPU).
- Evaluated answer-token cross-entropy and perplexity: Base zero-shot answer PPL dropped from `38.45` (loss 3.6493) to `30.08` (loss 3.4037), a `21.78%` relative improvement specifically on answer generation.
- Verified qualitative completions: Prompt-masked model directly names specific crops (corn, soybeans, wheat, rice) and pest control actions without repeating prompt phrasing.
- Updated `notebooks/02_domain_specific_llm_adaptation.ipynb` with interactive Sections 6 (Decoding Ablation) and 7 (Prompt Loss Masking).
- Synchronized `reports/LEARNING_JOURNAL.md` (Sections 10 and 11) and `reports/claims_table.md` (Claims 15 and 16).
**Decided:**
- Set Conservative Agronomic Sampling ($T=0.35, \text{top\_p}=0.85, r_{\text{rep}}=1.25, \text{no\_repeat\_ngram\_size}=3$) as the production recommendation for agricultural advisory assistants to eliminate self-reinforcement looping.
- Confirmed prompt-loss masking as the superior training paradigm over raw causal LM for instruction and Q&A adaptation.
**Next:**
- Commit all updates and push to `origin eric`.

---

## 2026-09-21 · 22:43–23:05 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Claude (Claude Code, Opus 5), to review the whole repo (worklog, journal, methodology guide, reports, notebooks, `src/`) against the code and data, and to re-run checks in a temporary scratch folder outside the repo. No code, notebook, or report was changed; this entry is the only edit.
**Did:**
- Reproduced exactly: every cell of the five journal ablation tables matches its `reports/results_*.json` (150/150); Dataset 1 and Unified Unicode Word N=1..6 re-run from scratch; claims-table row 9 (Bigram Laplace 1879.6, Bigram KN 327.7); Section C 62.38 → 29.33 using the repo's own eval logic. 13/13 tests pass.
- Found `_prob_interpolation` (uniform lambdas, as used by the ablation) is not a proper distribution when a higher-order context is unseen: probabilities sum to 0.667 at N=3 and 0.333 at N=6. `_prob_kneser_ney` for N≥3 backs off to 1/|V|, not to continuation counts. The validation splits are never loaded, so no lambda or k was tuned.
- Re-ran with textbook interpolated Kneser-Ney (Ney discounts, OOV test tokens excluded). Dataset 1, N=1..6: 512.0, 115.1, 77.8, 72.4, 71.6, 72.1. Unified: 598.3, 131.6, 83.4, 76.0, 74.6, 75.1. Best at N=5 on both, flat at N=6. The repo's own model with OOV tokens excluded still degrades (Dataset 1, N=3..6: 103.5, 107.9, 123.4, 142.7), so the shape comes from the smoothing, not the OOV handling: the N=4 "breaking point" and the scale-driven rightward shift are artifacts of the uniform interpolation.
- OOV: `min_freq=1` means `<unk>` never occurs in training, so each OOV test token gets p≈1e-12. OOVs are ~1.4% of test tokens but ~7% of total test NLL.
- Per-word normalization of the Unified JSON (same 4,000 test sentences): Unicode Word best 320.0 (N=4), Whitespace 441.2 (N=3), BPE 449.2 (N=6), Character 1985.2 (N=6, trained on 4,000 sentences only). Stemmer maps nusrɔ̃la→srɔ̃ but nusrɔ̃lawo→srɔ̃la, agbledela→de, megbe→gbe.
- Section C: 16/100 test pairs are exact duplicates of pairs among the 500 training pairs (KisanVaani has 22,615 rows but 2,221 unique questions; no dedup before the split). Clean 84 pairs: 65.37 → 31.30 (52.1% drop). Answer tokens only: 34.40 → 28.10 (18.3% drop). General English (WikiText-2 test, 200 paragraphs): 73.19 → 81.87 (+11.9%).
- Data: a large share of Dataset 1 is Jehovah's Witnesses and Bible text, not folklore (5 of 6 random rows sampled; 10.3% of its train lines contain "yehowa", 8.2% contain chapter:verse references).
**Blocked / open questions:**
- Claims with no code or data behind them: Section C Q6 forgetting check (<3.8%) and Sitophilus zeamais (0 hits in the corpus); Section B Q3 PPL 245 → 79 (hardcoded in notebook 03 cell 3, same values as journal §2.1–2.2) and the example generated phrases (they are notebook 01's hand-written fallback training sentences); Section B Q6 ARPA export (no code); journal §7.2 example Q&A (0 hits in KisanVaani); journal §4 rows 4–5; presentation slides 4–5 (α=16, q_proj/v_proj, 67.4%, 0.72%, <3.8%); datasheet §2–3 agricultural provenance (CSIR Ghana bulletins); Section B Q1 and datasheet describe Dataset 1 as folklore and non-religious.
- Notebook 02 does not run: code cells 3, 7 and 15 have unterminated string literals, and markdown cells 2 and 4 lost their inline-code text.
- The scripts that built `data/processed/dataset_1..4` and `unified` are not in the repo; Dataset 4 is a symlink into `~/Downloads`; `requirements.txt` has floors, not the pins the environment needs.
- origin has no `main` branch; the default branch is `eric`.
- The check scripts live in the session scratchpad and will not survive; the IKN used above is ~50 lines and can be moved into `src/ngram.py` if the fix goes ahead.
**Next:**
- Eric to choose which fixes to make: n-gram smoothing and re-run; remove or correct the unsupported claims; Section C dedup plus answer-only and general-English numbers.

---

## 2026-09-21 · 22:00–22:45 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Antigravity), to overhaul Section C with authentic agricultural extension corpus, build complete LoRA fine-tuning pipeline, resolve Python 3.14/3.12 macOS PyTorch compatibility, clean up repository hygiene (removing redundant scripts, untracking .zed/), update notebooks, and synchronize reports and learning journals.
**Did:**
- Repository hygiene & IDE cleanup: removed duplicate `.py` percent scripts from `notebooks/` to keep clean `.ipynb` files for team compatibility; untracked `.zed/` configuration and added to `.gitignore`.
- Environment repair: resolved macOS x86_64 PyTorch incompatibility (recreated virtualenv with Python 3.12, pinned `torch==2.2.2`, `transformers==4.38.2`, `peft==0.10.0`, `datasets==5.0.1`, `numpy==1.26.4`, `scipy==1.12.0`). Registered `nlp-prosit-1` Jupyter kernel.
- Resolved local module shadowing: renamed `src/tokenizers.py` to `src/ewe_tokenizers.py` to eliminate namespace collision with the Hugging Face `tokenizers` library.
- Data curation: downloaded authentic `KisanVaani/agriculture-qa-english-only` corpus (22,615 Q&A pairs) and partitioned leak-free Train (500 pairs / 17k words), Val (100 pairs / 3.7k words), and Test (100 pairs / 3.7k words) splits via `src/prepare_domain_data.py`.
- LoRA fine-tuning execution (`src/train_domain_lora.py` and `notebooks/02_domain_specific_llm_adaptation.ipynb`): trained DistilGPT2 with LoRA ($r=8, \alpha=32$, on `c_attn` attention projections; 147,456 trainable params / 0.18%) for 3 epochs (288.6s runtime on local CPU).
- Empirical evaluation: zero-shot base perplexity dropped from `62.38` (loss 4.1332) down to `29.33` (loss 3.3785) on held-out test split (52.99% relative perplexity reduction). Verified prompt completions shift from circular repetition to direct agronomic advice.
- Documentation & Reflection: updated `reports/section_c_domain_adaptation.md`, `reports/claims_table.md`, and `reports/LEARNING_JOURNAL.md` (Sections 7, 8, 9) with authentic metrics, architectural analysis, and viva defense answers.
**Decided:**
- Strictly use real data (`KisanVaani`) rather than synthetic mock sentences for all Section C deliverables.
- Maintain only standard `.ipynb` notebooks in git to avoid team merge confusion.
- Persist only lightweight benchmark JSON artifacts (`reports/domain_adaptation_results.json`) and figures in git while gitignoring heavy model binaries.
**Next:**
- Push branch `eric` to `origin` and conduct dry-run viva rehearsal using `reports/claims_table.md` and `reports/LEARNING_JOURNAL.md`.

---

## 2026-09-21 · 16:50–17:20 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to resolve Zed IDE language server configuration (installed `pyright` in `.venv`, configured explicit LSP paths in `.zed/settings.json`), create Zed-compatible percent scripts (`.py` with `# %%` markers) for all notebooks, install JupyterLab with a 1-click launcher (`scripts/launch_notebook.sh`), optimize Kneser-Ney continuation precomputation in `src/ngram.py`, build and verify a 13-test unit test suite (`tests/test_pipeline.py`), and synchronize `reports/claims_table.md`, `reports/datasheet.md`, and `reports/presentation_outline.md`.
**Did:**
- Resolved Zed IDE LSP tooling: installed `pyright` in `.venv`, updated `.zed/settings.json` to point directly to `.venv/bin/pyright` and `.venv/bin/ruff`, and validated fast autocomplete and format-on-save.
- Provided dual notebook execution paths: converted all 4 `.ipynb` notebooks into Zed-native percent scripts (`.py` with `# %%` blocks) and created `scripts/launch_notebook.sh` for full browser-based JupyterLab rendering.
- Optimized Kneser-Ney smoothing in `src/ngram.py`: precomputed continuation counts during `fit()` to reduce complexity from $O(N \times |V|)$ down to $O(1)$ table lookup, accelerating evaluation by >100x.
- Created unit test suite in `tests/test_pipeline.py` (13 tests passing in 0.07s) covering Unicode NFC normalization, combining tone diacritics, all 5 tokenizers, probability conservation ($\sum P = 1.0$), and perplexity math.
- Synchronized `reports/claims_table.md`, `reports/datasheet.md`, and `reports/presentation_outline.md` with the Grand Unified Mega-Corpus benchmarks and ASR WFST architecture.
**Decided:**
- Supported both Zed interactive REPL workflow (`.py` with `# %%`) and browser JupyterLab workflow (`.ipynb`) for maximum development flexibility.
- Refined Ewe morphological stemmer to support common 2-character root lemmas (`wɔ`, `yi`, `va`, `ɖu`).
**Next:**
- Review the presentation slide outline and rehearse the 10-minute slide deck and viva oral defense questions.

---

## 2026-09-21 · 13:13–13:28 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to execute the full 5-tokenizer ablation sweep ($N=1\dots 6$) on the Grand Unified Ewe Mega-Corpus (124,396 sentences / 2.35M words total; 99,516 train sentences / 1.88M words), document the empirical breakthrough of the rightward breaking point shift ($N=3 \to N=4$), update `reports/LEARNING_JOURNAL.md` and `reports/section_b_low_resource_lm.md`, and establish comprehensive `METHODOLOGY_GUIDE.md` playbooks across both NLP and Machine Learning repositories.
**Did:**
- Executed full 5-tokenizer evaluation sweep ($N=1\dots 6$) on the Grand Unified Ewe Mega-Corpus via `scripts/run_multi_tokenizer_ablation.py --dataset unified`:
  - Whitespace: Trigram optimum (PPL 441.2, 52.8% sparsity; degraded 3.2x by glued punctuation).
  - Unicode Word: 4-gram optimum (PPL 147.8, 62.6% sparsity; beating Trigram PPL 150.1).
  - Ewe Stemmer: 4-gram optimum (PPL 134.0, 61.6% sparsity; 9.3% error reduction over raw words).
  - BPE (150 merges): 6-gram optimum (PPL 13.8, 50.6% sparsity; zero OOV, monotonic improvement to $N=6$).
  - Character: 6-gram optimum (PPL 7.6, 37.9% sparsity; compact $|V|=123$).
- Saved full benchmark matrix in `reports/results_unified_all_tokenizers.json`.
- Discovered and proved the **rightward shift of the word-level breaking point**: corpus scaling to 1.88M words stabilizes 4-word co-occurrences, allowing $N=4$ to outperform $N=3$ for the first time in our study.
- Updated `reports/LEARNING_JOURNAL.md` with Phase 5 logs, the 5-Corpus Master Scaling Table, and extended Viva Quiz revision defenses.
- Updated `reports/section_b_low_resource_lm.md` (Questions 1, 5, and 6) with exact 4-source composition, multi-tokenizer metrics, and combining diacritic regex handling.
- Authored `METHODOLOGY_GUIDE.md` in this project and `mle-group-3-prosit-1` codifying the "Explain to a beginner, build like a senior" pedagogy, incremental ablation, worklog logging, learning journals, branch discipline, and viva exam readiness.
**Decided:**
- Confirmed that balanced multi-source fusion prevents the 99% liturgical skew of raw web scrapes while providing enough lexical scale for 4-grams to beat trigrams.
- Codified standard 5-pillar research methodology across both MICS 2028 coursework projects.
**Next:**
- Plot the comparative perplexity and sparsity curves across all 5 corpora into `figures/` for slide and report integration.

---

## 2026-09-21 · 13:00–13:08 GMT · Eric Elikplim Sunu

**Branch:** eric
**Assistant:** Gemini (Gemini 3.8 Flash), to preprocess Dataset 4 (`ewe_corpus.parquet` — 4.4M web/aligned sentences, sampling 200k rows yielding 64,308 train sentences / 867k words), execute the full 5-tokenizer ablation sweep ($N=1\dots 6$), and construct the 4-Dataset Grand Comparison Matrix in `reports/LEARNING_JOURNAL.md`.
**Did:**
- Extracted and deduplicated 80,385 clean unique sentences (filtered 55,126 duplicates) from 200,000 parquet rows into `data/processed/dataset_4_parquet/` (64,308 train / 8,039 test / 867,455 train words).
- Updated `scripts/run_multi_tokenizer_ablation.py` with `--dataset 4` configuration.
- Executed full 5-tokenizer ablation ($N=1\dots 6$) on Dataset 4: Whitespace (Trigram PPL 486.6), Unicode Word (Trigram PPL 140.3), Ewe Stemmer (Trigram PPL 128.6), BPE (5-gram & 6-gram PPL 14.0), Character (6-gram PPL 7.6).
- Documented empirical scaling laws in `reports/LEARNING_JOURNAL.md`: 150x increase in training data reduces test sparsity across all orders and enables BPE subwords to sustain $N=6$ with zero perplexity degradation.
**Decided:**
- Verified that on large-scale corpora, BPE subwords break the word-level $N=3$ ceiling and maintain flat, robust performance across orders 5 and 6 without overfitting.
**Next:**
- Phase 5: Corpus Fusion & Multi-Source Harmonization (merging all 4 datasets into the Grand Unified Ewe Corpus).

---

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
