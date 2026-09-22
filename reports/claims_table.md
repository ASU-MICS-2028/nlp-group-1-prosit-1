# Claims & Traceability Table: Prosit 1

Every number in the technical report, the slides and the learning journal must be copied from a file
that a committed script writes. This table names that file, the exact key, and the script. Files under
`data/` are gitignored; rebuild them with the commands in `README.md`. Rows marked *audit* are
measurements from the 2026-09-21/22 verification, recorded in `WORKLOG.md`, of code or data that no
longer exists in that form.

---

## Section B: Ewe Statistical Language Model (Èʋegbe)

| # | Claim | Value | Source (file → key) | Written by |
|---|---|---|---|---|
| 1 | Unified corpus size and split | 123,511 sentences; 98,808 / 12,351 / 12,352 | `data/processed/unified/stats.json` → `total_unique_sentences`, `train_sentences`, `val_sentences`, `test_sentences` | `scripts/build_ewe_datasets.py` |
| 2 | Training words (unified) | 1,874,130 | same file → `train_words` | `scripts/build_ewe_datasets.py` |
| 3 | Religious share | unified 5.1% mention Yehowa, 6.2% chapter:verse; Dataset 1: 10.6% and 6.8% | `data/processed/{unified,dataset_1_csv}/stats.json` → `pct_mentioning_yehowa`, `pct_with_chapter_verse` | `scripts/build_ewe_datasets.py` |
| 4 | Unicode Word test perplexity, $N=1..6$ | 534.7, 121.2, 77.7, 70.5, 69.3, 69.7 | `reports/results_unified_all_tokenizers.json` → `results["Unicode Word"][N-1].perplexity` | `scripts/run_multi_tokenizer_ablation.py` |
| 5 | Flat from $N=4$ | validation perplexities for $N=4..6$ within 3% on every dataset | all five `reports/results_*.json` → `val_perplexity` | `scripts/run_multi_tokenizer_ablation.py` |
| 6 | Best order and per-word perplexity per tokenizer (unified) | BPE 189.1 ($N=6$), Stemmer 196.9, Unicode Word 202.2, Whitespace 261.6 ($N=5$), Character 447.1 ($N=6$) | `results_unified_all_tokenizers.json` → `best_order_by_val`, `results[...].per_word_perplexity` | `scripts/run_multi_tokenizer_ablation.py` |
| 7 | Per-word perplexity with `<unk>` free (journal only) | Whitespace 120.1, Unicode Word 132.1, Stemmer 136.7, BPE 188.7, Character 446.2 | derived from the same rows: $\exp(\ln \text{PPL} \times (\text{test\_tokens} + 4000) / \text{test\_words})$ | computed from the JSON |
| 8 | Kneser-Ney beats equal-weight interpolation | 77.7 vs 93.6 at $N=3$ (Unicode Word); true at every $N \ge 2$ on all datasets | `results[...].perplexity` vs `results[...].interpolation_perplexity` | `scripts/run_multi_tokenizer_ablation.py` |
| 9 | Ney discount best on validation | unified: Ney 66.75 vs 0.5: 83.3, 0.75: 68.24, 0.9: 67.41 | `discount_check_val_unicode_word` in each results JSON | `scripts/run_multi_tokenizer_ablation.py` |
| 10 | Sparsity of test 6-grams (Unicode Word) | 80.5% | `results["Unicode Word"][5].sparsity_pct` | `scripts/run_multi_tokenizer_ablation.py` |
| 11 | Vocabulary, `<unk>` rate, tokens per word (unified) | e.g. Unicode Word 26,489 types, 1.93%, 1.16 tokens per word | `results[...][0]`: `vocab_size`, `oov_rate_pct`, `test_tokens / (test_words - 4000)` | `scripts/run_multi_tokenizer_ablation.py` |
| 12 | Cross-dataset table (journal §5) | per-dataset best $N$, `<unk>` rate, sparsity, per-word perplexity | `reports/results_dataset_{1,2,3,4}_all_tokenizers.json` | `scripts/run_multi_tokenizer_ablation.py` |
| 13 | Seeded sample reproducing Jonah 1:1 | "2 eye yehowa ƒe gbe va na yona , amitai vi ," | `results["Unicode Word"][3].sample_generation` | `scripts/run_multi_tokenizer_ablation.py` |
| 14 | Smoothing comparison on a 10,000-sentence sample (illustrative) | Bigram Laplace 629.52, Bigram Kneser-Ney 119.87, Trigram Kneser-Ney 99.90 | `notebooks/01_low_resource_ngram_lm.ipynb`, cell 9 output | the notebook |
| 15 | Old interpolation lost probability | sums of 0.667 (unseen trigram context) and 0.333 (unseen 6-gram context) | *audit*: probe of the pre-fix `src/ngram.py` | `WORKLOG.md`, 2026-09-21 |
| 16 | Lookalike letters | 5,026 lines of the previous unified training split; "ɖe" 46,110 vs "ðe" 2,073 | *audit* | `WORKLOG.md`, 2026-09-22 |
| 17 | Independent check of the new Kneser-Ney | 512.0, 115.1, 77.8, 72.4, 71.6, 72.1 on the old Dataset 1 split | *audit*: scratch implementation vs `src/ngram.py` | `WORKLOG.md`, 2026-09-22 |

## Section C: Agro-Extension Domain Adaptation (LoRA)

| # | Claim | Value | Source (file → key) | Written by |
|---|---|---|---|---|
| 18 | Corpus size after deduplication | 22,615 rows, 2,212 distinct questions; 1,769 / 221 / 222 pairs | `data/processed/domain_english/stats.json` | `src/prepare_domain_data.py` |
| 19 | No test question in training or validation | 0 | `reports/domain_adaptation_results.json` → `test_questions_seen_in_train_or_val` | `src/train_domain_lora.py` |
| 20 | Trainable parameters | 147,456 of 82,060,032 (0.18%) | `results.standard.trainable_params`, `results.standard.total_params` | `src/train_domain_lora.py` |
| 21 | Perplexities: full / answer-only / WikiText-2 | base 56.08 / 37.77 / 73.19; standard 28.13 / 30.00 / 78.44; masked 51.39 / 29.09 / 77.24 | `results.{base,standard,masked}.{full_ppl,answer_ppl,wikitext_ppl}` | `src/train_domain_lora.py` |
| 22 | Relative changes | standard: full -49.8%, answer -20.6%, WikiText +7.2%; masked: answer -23.0%, full -8.4%, WikiText +5.5% | computed from row 21 | computed from the JSON |
| 23 | Validation loss per epoch | standard (full text) 3.3963, 3.3072, 3.2874; masked (answers only) 3.3395, 3.3005, 3.2902 | `results.{standard,masked}.val_loss_per_epoch` | `src/train_domain_lora.py` |
| 24 | Seeded completions quoted in the reports | e.g. "The fall armyworm in maize affects the development of mites, insects and other insects." | `results.{base,standard,masked}.samples` | `src/train_domain_lora.py` |
| 25 | Decoding: mean Distinct-3 | unpenalized 78.9%, penalty 1.3: 100.0%, 3-gram block 99.5%, low temperature + penalty + block 100.0%, greedy + penalty + block 100.0% | `reports/decoding_strategies_benchmark.json` → `strategy_averages` | `scripts/benchmark_decoding_strategies.py` |
| 26 | Leakage in the superseded split | 16 of 100 test pairs identical to training pairs; 18 of 100 test "pairs" were answer fragments | *audit* | `WORKLOG.md`, 2026-09-21 and 2026-09-22 |
| 27 | Superseded masking comparison | standard 29.92 vs masked 30.08 answer perplexity on the old metric | *audit* | `WORKLOG.md`, 2026-09-22 |
