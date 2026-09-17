# Claims & Traceability Table — Prosit 1

Every quantitative claim, metric, or architectural statistic in the technical report or presentation slides must be directly traceable to a specific cell in a clean-running notebook.

This table provides the authoritative evidence base for your group presentation and the individual Automated AI Viva Quiz on `clenam.ai`.

---

## Section B: Twi Statistical Language Model (`ghana-nlp/abena-twi-corpus`)

| # | Slide / Report Claim | Exact Figure / Value | Source / Notebook Cell | Justification / Methodology |
|---|---|---|---|---|
| 1 | Unigram Laplace test perplexity | $245.8 \pm 12.4$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Laplace smoothed unigram ($k=1.0$) with closed vocabulary |
| 2 | Maximum Likelihood Estimation failure | $\text{PP} = \infty$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Zero-count dilemma: unsmoothed bigram fails on $38\%$ unseen transitions |
| 3 | Bigram Laplace test perplexity | $134.2 \pm 6.1$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Add-One smoothing redistributes probability mass uniformly across $|V|$ |
| 4 | Bigram Lidstone test perplexity | $112.6 \pm 5.0$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Add-$0.1$ smoothing shaves smaller probability mass than Add-1 |
| 5 | Trigram Linear Interpolation perplexity | $88.4 \pm 4.2$ | `01_low_resource_ngram_lm.ipynb` Cell 4 | Linear combination of unigram, bigram, trigram ($\lambda = [0.1, 0.3, 0.6]$) |
| 6 | Interpolated Kneser-Ney perplexity | **$79.1 \pm 3.8$** | `01_low_resource_ngram_lm.ipynb` Cell 4 | Absolute discount $d=0.75$, backing off to continuation probabilities |
| 7 | Stream sampling scale factor | `SCALE_FACTOR = 0.05` | `01_low_resource_ngram_lm.ipynb` Cell 2 | Iterative streaming prevents Google Colab / local RAM crashes |
| 8 | Out-of-vocabulary (OOV) test rate | $4.8\%$ | `01_low_resource_ngram_lm.ipynb` Cell 3 | Vocabulary induced strictly on training split; unseen words mapped to `<unk>` |

---

## Section C: Agro-Extension Domain Adaptation (LoRA)

| # | Slide / Report Claim | Exact Figure / Value | Source / Notebook Cell | Justification / Methodology |
|---|---|---|---|---|
| 9 | Base model zero-shot domain perplexity | $85.6 \pm 3.2$ | `02_domain_specific_llm_adaptation.ipynb` Cell 3 | Unadapted base model on held-out Agro-Extension test split |
| 10 | Post-LoRA domain test perplexity | **$27.9 \pm 1.5$** | `02_domain_specific_llm_adaptation.ipynb` Cell 5 | 5 epochs, $r=8, \alpha=16$, AdamW ($\text{lr} = 5 \times 10^{-4}$) |
| 11 | Relative perplexity reduction | **$67.4\%$ improvement** | `02_domain_specific_llm_adaptation.ipynb` Cell 5 | $\frac{85.6 - 27.9}{85.6} = 67.4\%$ drop in surprise |
| 12 | Trainable parameter percentage | **$0.72\%$** ($0.59\text{M} / 82\text{M}$) | `02_domain_specific_llm_adaptation.ipynb` Cell 4 | LoRA attached exclusively to `target_modules=["q_proj", "v_proj"]` |
| 13 | General English degradation check | $< 3.8\%$ change | `02_domain_specific_llm_adaptation.ipynb` Cell 6 | Verifies complete absence of catastrophic forgetting |
