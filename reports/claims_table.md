# Claims & Traceability Table — Prosit 1

Every quantitative claim, statistic, or comparison appearing in the technical report or presentation slides must be traceable to a specific cell in a clean-running notebook.

A panel examiner will pick claims at random during the presentation and viva quiz.

---

## Low-Resource African Language LM (Section B)

| # | Slide / Report Claim | Exact Figure / Value | Source / Notebook Cell | Justification / Methodology |
|---|---|---|---|---|
| 1 | Raw unigram test perplexity | $245.8 \pm 12.4$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Laplace smoothed unigram ($k=1.0$) |
| 2 | Maximum Likelihood Estimation failure | $\text{PPL} = \infty$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Unsmoothed bigram fails on $38\%$ unseen transitions |
| 3 | Bigram Laplace test perplexity | $134.2 \pm 6.1$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Add-1 smoothing redistributes mass uniformly |
| 4 | Trigram Linear Interpolation perplexity | $88.4 \pm 4.2$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Tuned validation weights: $\lambda = [0.1, 0.3, 0.6]$ |
| 5 | Interpolated Kneser-Ney perplexity | $79.1 \pm 3.8$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Uses continuation probabilities for backoff |
| 6 | Out-of-vocabulary (OOV) rate on test set | $4.8\%$ | `01_low_resource_ngram_lm.ipynb` Cell 2 | Closed vocabulary threshold ($min\_freq=2$) on train split |

---

## Domain-Adapted English Model (Section C)

| # | Slide / Report Claim | Exact Figure / Value | Source / Notebook Cell | Justification / Methodology |
|---|---|---|---|---|
| 7 | Zero-shot base model perplexity on domain test set | $85.6 \pm 3.2$ | `02_domain_specific_llm_adaptation.ipynb` Cell 3 | Evaluated on held-out domain test partition |
| 8 | Post-LoRA domain test perplexity | $27.9 \pm 1.5$ | `02_domain_specific_llm_adaptation.ipynb` Cell 5 | 5 epochs, $r=8, \alpha=32$, AdamW learning rate $5 \times 10^{-4}$ |
| 9 | Relative perplexity reduction | $67.4\%$ improvement | `02_domain_specific_llm_adaptation.ipynb` Cell 5 | $(85.6 - 27.9) / 85.6$ |
| 10 | Trainable parameter percentage | $0.72\%$ ($0.59\text{M} / 82\text{M}$) | `02_domain_specific_llm_adaptation.ipynb` Cell 4 | LoRA adapters applied to query and value projections |
| 11 | General English perplexity degradation | $< 3.8\%$ change | `02_domain_specific_llm_adaptation.ipynb` Cell 6 | Verifies absence of catastrophic forgetting on general test |
