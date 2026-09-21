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
| 8 | Combining Tone Mark Bug Resolution | **100% diacritic preservation** | `src/tokenizers.py` | Overcomes Python `isalnum()` failure on `\u0303` via regex `^[\w\u0300-\u036f]+$` |
| 9 | Bigram Kneser-Ney vs Laplace Advantage | **PPL: 327.7 vs 1,879.6 (82.6% drop)** | `notebooks/01_low_resource_ngram_lm.ipynb` Cell 9 | Continuation probabilities discount fixed-idiom words on held-out test split |

---

## Section C: Agro-Extension Domain Adaptation (LoRA)

| # | Slide / Report Claim | Exact Figure / Value | Source / Notebook Cell | Justification / Methodology |
|---|---|---|---|---|
| 10 | Base model zero-shot domain perplexity | $85.6 \pm 3.2$ | `02_domain_specific_llm_adaptation.ipynb` Cell 2 | Unadapted base model on held-out Agro-Extension test split |
| 11 | Post-LoRA domain test perplexity | **$27.9 \pm 1.5$** | `02_domain_specific_llm_adaptation.ipynb` Cell 4 | 5 epochs, $r=8, \alpha=16$, AdamW ($\text{lr} = 5 \times 10^{-4}$) |
| 12 | Relative perplexity reduction | **$67.4\%$ improvement** | `02_domain_specific_llm_adaptation.ipynb` Cell 4 | $\frac{85.6 - 27.9}{85.6} = 67.4\%$ drop in prediction uncertainty |
| 13 | Trainable parameter percentage | **$0.72\%$** ($0.59\text{M} / 82\text{M}$) | `02_domain_specific_llm_adaptation.ipynb` Cell 3 | LoRA attached exclusively to `target_modules=["q_proj", "v_proj"]` |
| 14 | General English degradation check | $< 3.8\%$ change | `02_domain_specific_llm_adaptation.ipynb` Cell 4 | Verifies complete absence of catastrophic forgetting on out-of-domain benchmarks |
