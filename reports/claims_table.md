# Claims & Traceability Table — Prosit 1

Every quantitative claim, metric, or architectural statistic in the technical report or presentation slides must be directly traceable to a specific cell in a clean-running notebook and benchmark artifact.

This table provides the authoritative evidence base for your group presentation and the individual Automated AI Viva Quiz on `clenam.ai`.

---

## Section B: Ewe Statistical Language Model (Èʋegbe)

| # | Slide / Report Claim | Exact Figure / Value | Source / Benchmark Artifact | Justification / Methodology |
|---|---|---|---|---|
| 1 | Grand Unified Mega-Corpus Scale | **124,396 sentences / 2,349,941 tokens** | `data/processed/unified/stats.json` | 4 sources (Folklore, Bios, Waxal Speech, Web), 2,260 cross-duplicates removed |
| 2 | Training / Test Partition Scale | **99,516 Train (1.88M words)** / 12,441 Test | `reports/results_unified_all_tokenizers.json` | Leak-free 80/10/10 split; evaluated on 4,000 held-out test sentences |
| 3 | Rightward Word Breaking Point Shift | **$N=3 \to N=4$** (PPL: $150.1 \to 147.8$) | `reports/results_unified_all_tokenizers.json` | At 1.88M words, 4-gram contexts recur with sufficient frequency to beat Trigram |
| 4 | Ewe Morphological Stemmer Optimum | **4-gram ($N=4$): PPL = 134.0** | `reports/results_unified_all_tokenizers.json` | Peeling affixes (`-wo`, `mí-`, `wó-`) pools inflections, achieving 9.3% error reduction |
| 5 | Byte-Pair Encoding (BPE) Peak Context | **6-gram ($N=6$): PPL = 13.8** | `reports/results_unified_all_tokenizers.json` | Subwords eliminate OOV crashes (0.0% unigram sparsity) and sustain $N=6$ context |
| 6 | Whitespace Glued Punctuation Penalty | **3.2x degradation** (PPL 470.2 vs 147.8) | `reports/results_unified_all_tokenizers.json` | Punctuation attached to words pollutes surface forms, inflating $|V|$ to 100,707 |
| 7 | Character Model Branching Baseline | **6-gram ($N=6$): PPL = 7.6** | `reports/results_unified_all_tokenizers.json` | Compact character alphabet ($|V|=123$); proves Viva Perplexity Invariance Rule |
| 8 | Combining Tone Mark Bug Resolution | **100% diacritic preservation** | `src/ewe_tokenizers.py` | Overcomes Python `isalnum()` failure on `\u0303` via regex `^[\w\u0300-\u036f]+$` |
| 9 | Bigram Kneser-Ney vs Laplace Advantage | **PPL: 327.7 vs 1,879.6 (82.6% drop)** | `notebooks/01_low_resource_ngram_lm.ipynb` Cell 9 | Continuation probabilities discount fixed-idiom words on held-out test split |

---

## Section C: Agro-Extension Domain Adaptation (LoRA)

| # | Slide / Report Claim | Exact Figure / Value | Source / Benchmark Artifact | Justification / Methodology |
|---|---|---|---|---|
| 10 | Base model zero-shot domain perplexity | **62.38** (Loss: 4.1332) | `reports/domain_adaptation_results.json` | Pretrained DistilGPT2 evaluated on 100 held-out agricultural Q&A test pairs |
| 11 | Post-LoRA domain test perplexity | **29.33** (Loss: 3.3785) | `reports/domain_adaptation_results.json` | 3 epochs, $r=8, \alpha=32$, AdamW ($\text{lr} = 5 \times 10^{-4}$), 189 steps |
| 12 | Relative perplexity reduction | **52.99% improvement** | `reports/domain_adaptation_results.json` | $\frac{62.38 - 29.33}{62.38} = 52.99\%$ drop in prediction uncertainty |
| 13 | Trainable parameter percentage | **0.18%** (147,456 / 82,060,032) | `reports/domain_adaptation_results.json` | LoRA attached exclusively to attention projections (`c_attn` Conv1D) |
| 14 | Training compute efficiency | **288.6 seconds** (~4.8 min) | `reports/domain_adaptation_results.json` | Parameter efficiency enables rapid fine-tuning on standard local CPU |
| 15 | Repetition suppression via decoding | **Distinct-3: 49.1% $\to$ 100.0%** | `reports/decoding_strategies_benchmark.json` | Repetition penalty ($r=1.25\dots 1.3$) and $N=3$ block eliminates phrase looping |
| 16 | Prompt-loss masking answer perplexity | **38.45 $\to$ 30.08 (21.8% drop)** | `reports/prompt_masking_ablation_results.json` | Masking prompt tokens (`-100`) dedicates all gradient updates to answer tokens |

