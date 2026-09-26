# Claims & Traceability Table: Prosit 1

Every number in the technical report, the slides and the learning journal must be copied from a file
that a committed script writes. This table names that file, the exact key, and the script. Data under
`data/` is gitignored except each split's small `stats.json`; rebuild everything with the commands in
`README.md`. Rows marked *audit* are
measurements from the 2026-09-21/22 verification, recorded in `WORKLOG.md`, of code or data that no
longer exists in that form.

---

## Section B: Ewe n-gram models (Èʋegbe), `src/section_b_ngram/`

| # | Claim | Value | Source (file → key) | Written by |
|---|---|---|---|---|
| 1 | Unified corpus size and split | 123,511 sentences; 98,808 / 12,351 / 12,352 | `data/processed/unified/stats.json` → `total_unique_sentences`, `train_sentences`, `val_sentences`, `test_sentences` | `src/section_b_ngram/build_datasets.py` |
| 2 | Training words (unified) | 1,874,130 | same file → `train_words` | `src/section_b_ngram/build_datasets.py` |
| 3 | Religious share | unified 5.1% mention Yehowa, 6.2% chapter:verse; Dataset 1: 10.6% and 6.8% | `data/processed/{unified,dataset_1_csv}/stats.json` → `pct_mentioning_yehowa`, `pct_with_chapter_verse` | `src/section_b_ngram/build_datasets.py` |
| 4 | Unicode Word test perplexity, $N=1..6$ | 534.7, 121.2, 77.7, 70.5, 69.3, 69.7 | `results/section_b_ngram/unified.json` → `results["Unicode Word"][N-1].perplexity` | `src/section_b_ngram/run_sweep.py` |
| 5 | Flat from $N=4$ | validation perplexities for $N=4..6$ are 3% or less apart on every dataset | all five `results/section_b_ngram/*.json` → `val_perplexity` | `src/section_b_ngram/run_sweep.py` |
| 6 | Best order and per-word perplexity per tokenizer (unified) | BPE 189.1 ($N=6$), Stemmer 196.9, Unicode Word 202.2, Whitespace 261.6 ($N=5$), Character 447.1 ($N=6$) | `results/section_b_ngram/unified.json` → `best_order_by_val`, `results[...].per_word_perplexity` | `src/section_b_ngram/run_sweep.py` |
| 7 | Per-word perplexity with `<unk>` free (journal §2.2 only) | Whitespace 120.1, Unicode Word 132.1, Stemmer 136.7, BPE 188.7, Character 446.2 | derived from the same rows: $\exp(\ln \text{PPL} \times (\text{test\_tokens} + 4000) / \text{test\_words})$ | computed from the JSON |
| 8 | Kneser-Ney beats equal-weight interpolation | 77.7 vs 93.6 at $N=3$ (Unicode Word); true at every $N \ge 2$ on all datasets | `results[...].perplexity` vs `results[...].interpolation_perplexity` | `src/section_b_ngram/run_sweep.py` |
| 9 | Ney discount best on validation | unified: Ney 66.75 vs 0.5: 83.3, 0.75: 68.24, 0.9: 67.41 | `discount_check_val_unicode_word` in each results JSON | `src/section_b_ngram/run_sweep.py` |
| 10 | Sparsity of test 6-grams (Unicode Word) | 80.5% | `results["Unicode Word"][5].sparsity_pct` | `src/section_b_ngram/run_sweep.py` |
| 11 | Vocabulary, `<unk>` rate, tokens per word (unified) | e.g. Unicode Word 26,489 types, 1.93%, 1.16 tokens per word | `results[...][0]`: `vocab_size`, `oov_rate_pct`, `test_tokens / (test_words - 4000)` | `src/section_b_ngram/run_sweep.py` |
| 12 | Cross-dataset table (journal §2.3) | per-dataset best $N$, `<unk>` rate, sparsity, per-word perplexity | `results/section_b_ngram/dataset_{1,2,3,4}.json` | `src/section_b_ngram/run_sweep.py` |
| 13 | Seeded sample reproducing Jonah 1:1 | "2 eye yehowa ƒe gbe va na yona , amitai vi ," | `results["Unicode Word"][3].sample_generation` | `src/section_b_ngram/run_sweep.py` |
| 14 | Smoothing comparison on a 10,000-sentence sample (illustrative) | Bigram Laplace 629.52, Bigram Kneser-Ney 119.87, Trigram Kneser-Ney 99.90 | `notebooks/b1_ngram_smoothing.ipynb`, cell 9 output | the notebook |
| 15 | Old interpolation lost probability | sums of 0.667 (unseen trigram context) and 0.333 (unseen 6-gram context) | *audit*: probe of the pre-fix n-gram code (then `src/ngram.py`) | `WORKLOG.md`, 2026-09-21 |
| 16 | Lookalike letters | 5,026 lines of the previous unified training split; "ɖe" 46,110 vs "ðe" 2,073 | *audit* | `WORKLOG.md`, 2026-09-22 |
| 17 | Independent check of the new Kneser-Ney | 512.0, 115.1, 77.8, 72.4, 71.6, 72.1 on the old Dataset 1 split | *audit*: scratch implementation vs `src/section_b_ngram/ngram.py` | `WORKLOG.md`, 2026-09-22 |

## Section C: English LLM, distilgpt2 + LoRA, `src/section_c_llm/`

| # | Claim | Value | Source (file → key) | Written by |
|---|---|---|---|---|
| 18 | Corpus size after deduplication | 22,615 rows, 2,212 distinct questions; 1,769 / 221 / 222 pairs | `data/processed/domain_english/stats.json` | `src/section_c_llm/prepare_data.py` |
| 19 | No test question in training or validation | 0 | `results/section_c_llm/lora_results.json` → `test_questions_seen_in_train_or_val` | `src/section_c_llm/train_lora.py` |
| 20 | Trainable parameters | 147,456 of 82,060,032 (0.18%) | `results.standard.trainable_params`, `results.standard.total_params` | `src/section_c_llm/train_lora.py` |
| 21 | Perplexities: full / answer-only / WikiText-2 | base 56.08 / 37.77 / 73.19; standard 28.13 / 30.00 / 78.44; masked 51.39 / 29.09 / 77.24 | `results.{base,standard,masked}.{full_ppl,answer_ppl,wikitext_ppl}` | `src/section_c_llm/train_lora.py` |
| 22 | Relative changes | standard: full -49.8%, answer -20.6%, WikiText +7.2%; masked: answer -23.0%, full -8.4%, WikiText +5.5% | computed from row 21 | computed from the JSON |
| 23 | Validation loss per epoch | standard (full text) 3.3963, 3.3072, 3.2874; masked (answers only) 3.3395, 3.3005, 3.2902 | `results.{standard,masked}.val_loss_per_epoch` | `src/section_c_llm/train_lora.py` |
| 24 | Seeded completions quoted in the reports | e.g. "The fall armyworm in maize affects the development of mites, insects and other insects." | `results.{base,standard,masked}.samples` | `src/section_c_llm/train_lora.py` |
| 25 | Decoding: mean Distinct-3 | unpenalized 78.9%, penalty 1.3: 100.0%, 3-gram block 99.5%, low temperature + penalty + block 100.0%, greedy + penalty + block 100.0% | `results/section_c_llm/decoding_benchmark.json` → `strategy_averages` | `src/section_c_llm/benchmark_decoding.py` |
| 26 | Leakage in the superseded split | 16 of 100 test pairs identical to training pairs; 18 of 100 test "pairs" were answer fragments | *audit* | `WORKLOG.md`, 2026-09-21 and 2026-09-22; journal §5, row 10 |
| 27 | Superseded masking comparison | standard 29.92 vs masked 30.08 answer perplexity on the old metric | *audit* | `WORKLOG.md`, 2026-09-22 |

## Section B, Question 2: LSTM baseline, `src/section_b_lstm/`

| # | Claim | Value | Source (file → key) | Written by |
|---|---|---|---|---|
| 28 | LSTM vs Kneser-Ney, Dataset 2 (420 sentences) | LSTM 7,864.84 per word (7,396.3 to 8,144.1, seeds 42 to 44) vs Kneser-Ney 5,806.95; LSTM 479,964 parameters | `results/section_b_lstm/lstm_vs_ngram.json` → `["2"].lstm_per_word_mean`, `lstm_per_word_min`, `lstm_per_word_max`, `kn_bpe.per_word_perplexity`, `runs[].params` | `src/section_b_lstm/run_baseline.py` |
| 29 | LSTM vs Kneser-Ney, unified corpus | LSTM 166.04 per word (164.88 to 167.54) vs Kneser-Ney 189.07; LSTM 3,964,276 parameters; 2.40 to 3.12 hours per seed | same file → `["unified"]`, plus `runs[].train_seconds` | `src/section_b_lstm/run_baseline.py` |
| 30 | Unified LSTM still improving at the 10-epoch budget | best epoch 10 of 10 for every seed; last two validation perplexities 9.200 → 9.157, 9.162 → 9.100, 9.249 → 9.147 | `["unified"].runs[].best_epoch`, `runs[].val_perplexity_per_epoch` | `src/section_b_lstm/run_baseline.py` |
| 31 | Parameters per training word | about 50 (479,964 / 9,642) and about 2 (3,964,276 / 1,874,130) | row 28/29 parameters divided by `train_words` in `data/processed/{dataset_2_json,unified}/stats.json` | computed |
| 32 | Same token stream as the n-gram | vocabulary, predicted-token count and spelling charge equal the BPE n-gram row | assertions in `src/section_b_lstm/run_baseline.py` against `results/section_b_ngram/*.json` | `src/section_b_lstm/run_baseline.py` |
| 33 | Kneser-Ney time for all six BPE orders (unified) | 425 s | *audit*: sweep log | `WORKLOG.md`, 2026-09-22/24 |
| 34 | First Dataset 2 LSTM run was under-trained | 9,243.16 per word with a 20-epoch cap, still improving | *audit* | `WORKLOG.md`, 2026-09-22/24 |
