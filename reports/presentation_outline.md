# Team Presentation Outline (10-Minute Slide Deck)

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Topic**: Building and Adapting Language Models (Prosit 1)  
**Lab Context**: Ankora AI Research Lab (Ghana)  
**Deliverable Link**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  
*(Must appear on Slide 1 as required by assignment guidelines)*

---

### Slide 1: Title & Overview (0:00 - 1:00)
- **Title**: Specialized Language Modeling for Low-Resource Ewe (Èʋegbe) & Agro-Extension Domain Adaptation
- **Team**: MICS 2028 · Group 1
- **GitHub Repository**: `https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git` *(Direct link on slide)*
- **Context & Motivation**: Ankora AI Research Lab internship deliverables:
  1. Low-resource statistical language model for Ewe speech recognition decoders.
  2. Parameter-efficient fine-tuning (PEFT / LoRA) for specialized agricultural advisory.

---

### Slide 2: Problem Formulation & The Statistical Imperative (1:00 - 2:30)
- **The African NLP Dilemma**: Under data scarcity (<100k sentences), deep neural models overfit, memorize noise, and suffer high latency.
- **The Classical Solution**: Smoothed n-gram models compile into **Weighted Finite-State Transducers (WFSTs)** with $O(1)$ table-lookup decoding, running with microsecond latency on edge CPUs.
- **Handling Data Sparsity**: Why Maximum Likelihood Estimation fails ($\text{PP}=\infty$ on 38% unseen transitions); the mathematical progression from Laplace and Lidstone to Interpolation and Kneser-Ney.

---

### Slide 3: Ewe Low-Resource Language Model & 5-Tokenizer Ablation (2:30 - 4:30)
- **The Grand Unified Ewe Mega-Corpus**: 124,396 clean sentences / 2.35M words harmonized across 4 registers (Folklore, Biographies, Waxal Spoken Speech, and Web Crawl).
- **Linguistic & Engineering Innovations**:
  - Unicode NFC normalization and remediation of Python's combining tone mark bug (`\u0303` regex fix).
  - Multi-tokenizer spectrum benchmark: Whitespace, Unicode Word, Ewe Stemmer, BPE Subwords, and Character.
- **Key Empirical Discoveries**:
  - **The Punctuation Penalty**: Glued punctuation degrades Whitespace perplexity by 3.2x (470.2 vs 147.8).
  - **The Stemming Advantage**: Peeling affixes (`-wo`, `mí-`) reduces perplexity across all orders ($147.8 \to 134.0$).
  - **The Rightward Shift**: Corpus scaling pushes the word breaking point from $N=3$ to $N=4$.
  - **Subword Headroom**: BPE subwords eliminate OOV and sustain peak context all the way to $N=6$ (PPL = 13.8).

---

### Slide 4: Domain-Specific LLM Adaptation (Agro-Extension) (4:30 - 6:30)
- **Domain Selection**: Authentic KisanVaani Agricultural Extension Advisory Q&A Corpus (`KisanVaani/agriculture-qa-english-only`, 22,615 extension pairs).
- **Methodological Evaluation**:
  - *From Scratch*: Computationally prohibitive ($>10^5$ GPU hours).
  - *RAG*: High retrieval latency and index maintenance overhead.
  - *PEFT / LoRA (Selected)*: Parameter-efficient, freezes 99.82% of base weights, adapts low-rank matrices ($r=8, \alpha=32$) on attention projections (`c_attn` Conv1D).
- **Safety & Efficiency**: Trains only **147,456 parameters (0.18%)** in under 5 minutes on standard CPU, completely avoiding catastrophic forgetting.

---

### Slide 5: Experimental Results & Claims Traceability (6:30 - 8:30)
- **Quantitative Benchmark Highlights**:
  - Ewe Statistical LM: 4-gram Stemmer achieves **PPL = 134.0**; BPE 6-gram achieves **PPL = 13.8**.
  - Agro-Extension LoRA: Domain test perplexity drops by **53.0%** (**62.38 $\to$ 29.33**; Loss: $4.13 \to 3.38$) with only **0.18%** trainable parameters.
  - Repetition Suppression: Repetition penalty ($r=1.25-1.3$) boosts Distinct-3 unique trigrams from **49.1% $\to$ 100.0%**, eliminating self-reinforcement loops.
  - Prompt-Loss Masking: Focusing gradients on response tokens reduces answer-token PPL from **38.45 $\to$ 30.08** (21.8% drop).
- **Traceability Table (`reports/claims_table.md`)**:
  - Every single number on our slides is cross-referenced to reproducible benchmark JSON artifacts and notebook cells.
- **Qualitative Generation Samples**:
  - Base model repetitively parrots questions; adapted model immediately generates actionable agronomic guidance naming specific crops (`maize`, `cassava`, `soybeans`).

---

### Slide 6: Key Takeaways & Viva Readiness (8:30 - 10:00)
- **Core Insights**:
  1. Statistical N-grams with morphological stemming provide the optimal compute-accuracy frontier for low-resource ASR decoding.
  2. LoRA enables cost-effective domain adaptation for specialized African enterprise applications.
- **Reproducibility**: Complete open-source pipeline with automated linting, tests, and reflective learning journals.
- **Q&A / Panel Defense**: Ready for automated Viva Quiz on `clenam.ai` and panel questions.
