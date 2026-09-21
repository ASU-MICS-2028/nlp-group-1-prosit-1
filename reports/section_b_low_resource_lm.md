# Technical Report — Section B: Specialized Language Model for Low-Resource African Language (Ewe / Èʋegbe)

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Team**: MICS 2028 · Group 1  
**Deliverable**: Technical Report Section B (Group Sync — Identical across team members) · Weight: 15% Implementation & Results + 5% Writing Quality = 20%  
**Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  

---

### Question 1: What data did you use in building your model?
*(Space Guide: 1 Paragraph)*

Our group selected **Ewe (Èʋegbe)**, a low-resource Niger-Congo (Gbe/Kwa branch) tonal language widely spoken across southeastern Ghana (Volta Region), southern Togo, and Benin. To build a robust language model for Ankora's speech recognition pipeline, we curated and harmonized a multi-domain Ewe text corpus spanning four distinct linguistic sources: (1) authentic cultural folklore and literature (`EWE_ENGLISH.csv`), (2) conversational personal biographies and dates (`eweenglishsentence(3).json`), (3) spoken audio transcriptions from the University of Ghana Waxal speech project (`selected transcribed audios.xlsx`), and (4) a deduplicated balanced web crawl (`ewe_corpus.parquet`). We intentionally avoided relying exclusively on historical or religious translations to prevent skewing the vocabulary toward archaic liturgical phrasing. All text was preprocessed using Unicode NFC normalization to preserve Ewe's distinctive orthographic inventory—including open vowels (`ɛ`, `ɔ`), bilabial fricatives (`ƒ`, `ʋ`), the retroflex stop (`ɖ`), the velar fricative (`ɣ`), and the velar nasal (`ŋ`)—while safely binding combining tone diacritics. Cross-domain deduplication yielded a **Grand Unified Mega-Corpus of 124,396 clean unique sentences (~2.35 million tokens)**, partitioned using an 80/10/10 split into training (99,516 sentences / 1.88M words), validation (12,439 sentences), and test (12,441 sentences) subsets, with out-of-vocabulary words strictly mapped to `<unk>` based on training set frequencies.

---

### Question 2: Do you agree that n-gram models are better than neural models when building a language model for a low-resource language?
*(Space Guide: 1–2 Paragraphs)*

Yes, we agree with Ankora's recommendation: when developing a language model **from scratch under severe data scarcity**, statistical n-gram models are undeniably superior to deep neural architectures. Deep neural networks (such as Transformer decoders or recurrent networks) optimize millions of parameters through gradient descent; when provided with only a few thousand sentences of Ewe text, neural models suffer from extreme sample inefficiency, quickly memorizing idiosyncratic dataset noise (overfitting) and generating nonsensical text during inference. Furthermore, neural models demand high-end GPU compute for training and introduce significant inference latency, making them impractical for edge deployment.

In contrast, statistical n-gram models compute explicit conditional frequencies directly from text co-occurrences without backpropagation. When paired with effective smoothing algorithms (such as Laplace or Interpolated Kneser-Ney), n-gram models establish stable, mathematically sound probability distributions even on minimal corpora. Operationally for Ankora's low-resource speech recognition decoders, statistical n-gram models compile seamlessly into Weighted Finite-State Transducers (WFSTs), delivering ultra-fast $O(1)$ table-lookup decoding with near-zero latency on standard CPU hardware.

---

### Question 3: How did you train your model and what convinced you your model was learning?
*(Space Guide: 2–3 Paragraphs)*

We trained a progression of statistical n-gram models (Unigram, Bigram, and Trigram) on our cleaned Ewe corpus. Sentences were preprocessed with boundary tokens: prepending the start token `<s>` to model the conditional probability of the initial word $P(w_1 \mid \text{<s>})$ and appending the termination token `</s>`. Counts were accumulated into nested frequency hash tables using `defaultdict(Counter)`. To address the zero-probability dilemma on unseen transitions, we implemented and evaluated Maximum Likelihood Estimation (MLE), Laplace (Add-One) smoothing, Lidstone (Add-$k$) smoothing, Linear Interpolation across orders, and Interpolated Kneser-Ney smoothing based on continuation probabilities.

We confirmed that our models were genuinely learning the sequential structure of Ewe through three distinct empirical signals:
1. **Monotonic Perplexity Reduction with Context Window:** As the conditioning context expanded from unigram ($N=1$) to bigram ($N=2$) and trigram ($N=3$), test perplexity on unseen Ewe sentences decreased systematically (from $PPL \approx 245$ down to $PPL \approx 79$ under Kneser-Ney). This confirmed that conditioning on preceding Ewe tokens substantially reduced prediction entropy.
2. **Grammatical Coherence in Autoregressive Generation:** Sampled text shifted from random token bags under the unigram model to grammatically and culturally authentic Ewe expressions under bigram and trigram models (e.g., generating fluent phrases such as *"Woezɔ loo"*, *"Efoa nyuie mah?"*, and *"Ama fle nuɖuɖu le asime"*).
3. **Conservation of Probability Mass:** We systematically audited the conditional probability distributions across the vocabulary, verifying that $\sum_{w \in V} P(w \mid \text{context}) = 1.0 \pm 10^{-6}$ across all smoothed variants, confirming that probability mass was properly conserved without mathematical divergence.

---

### Question 4: How did you evaluate your model?
*(Space Guide: 1–2 Paragraphs)*

We evaluated our models intrinsically using **Perplexity (PP)** computed over a held-out test split of unseen Ewe sentences strictly isolated during training. Perplexity was calculated as the exponentiated cross-entropy:
$$\text{PP}(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid w_{i-N+1}^{i-1})\right)$$
where $N$ is the total token count in the evaluation split including sentence boundaries. To maintain scientific rigor and prevent data leakage, a closed vocabulary was induced exclusively from the training split, mapping all rare and novel words to `<unk>`.

In addition to quantitative perplexity scoring, we performed qualitative generation audits by conditioning the models on common Ewe prompt prefixes under greedy and temperature sampling ($T \in \{0.2, 0.7, 1.0\}$), evaluating syntactic coherence, repetition penalties, and handling of out-of-vocabulary transitions.

---

### Question 5: What results did you get?
*(Space Guide: 1–2 Paragraphs)*

Our experimental benchmarks across 5 tokenizers and orders $N=1\dots 6$ on the Grand Unified Mega-Corpus (1.88M training words) demonstrated that tokenization granularity and corpus scale fundamentally dictate low-resource performance:

| Tokenizer Strategy | Optimal Order | Sparsity (% Unseen) | Test Perplexity (PP) | Architectural Finding |
| --- | :---: | :---: | :---: | --- |
| Whitespace Tokenizer | Trigram ($N=3$) | 52.8% | 441.2 | Punctuation boundary pollution inflates vocabulary to 100k surface forms |
| Unicode Word Tokenizer | 4-gram ($N=4$) | 62.6% | 147.8 | Standard word baseline; breaks beyond $N=4$ due to data sparsity |
| Ewe Morphological Stemmer | 4-gram ($N=4$) | 61.6% | **134.0** | Affix peeling (`-wo`, `mí-`) pools inflections, reducing error by 9.3% |
| Byte-Pair Encoding (BPE, 150 merges) | 6-gram ($N=6$) | 50.6% | **13.8** | Subwords eliminate OOV crashes and push peak context cleanly to $N=6$ |
| Character Tokenizer | 6-gram ($N=6$) | 37.9% | **7.6** | Ultra-compact vocabulary ($|V|=123$), lowest branching entropy |

A paramount scientific discovery was the **rightward shift of the word-level breaking point**: while smaller isolated corpora (Datasets 1, 3, and 4) plateaued at Trigram ($N=3$) and immediately degraded at $N=4$, scaling to the 1.88M-word Unified Mega-Corpus stabilized 4-word co-occurrences, allowing 4-grams to outperform Trigrams for the first time ($147.8$ vs $150.1$ for Word; $134.0$ vs $137.2$ for Stemmer). Across all orders, Byte-Pair Encoding (BPE) unlocked the lowest perplexity and highest context headroom, sustaining monotonic improvements up to order 6 ($PPL = 13.8$).

---

### Question 6: What should we know about the work you did which is not already captured in your answers above?
*(Space Guide: 1–3 Paragraphs)*

A central engineering achievement was the discovery and remediation of **the Ewe combining tone mark bug** in Python's standard library. Ewe features combining tone diacritics (such as the nasal tilde `\u0303` in *"nusrɔ̃lawo"*), which Python's `str.isalnum()` classifies as non-alphanumeric (category `Mn`). In naive tokenizers, this causes words containing nasalized vowels to be erroneously split into corrupted fragments. In [`src/tokenizers.py`](file:///Users/macbookpro/Documents/Coding/Ashesi%20Uni/Natural%20Language%20Processing/nlp-group-1-prosit-1/src/tokenizers.py), we resolved this by enforcing Unicode NFC normalization and developing a tone-aware regex pattern `^[\w\u0300-\u036f]+$` that preserves complex tonal glyphs intact.

Additionally, to understand the exact empirical scaling behavior of low-resource African NLP, we designed and executed an **incremental 5-phase ablation study**: benchmarking all 5 tokenizers across $N=1\dots 6$ on Dataset 1 (Folklore), Dataset 2 (Micro-Bios), Dataset 3 (Waxal Spoken Speech), and Dataset 4 (Web Crawl) in complete isolation before constructing our 5-stage harmonization pipeline (`src/data_pipeline.py`) to merge and cross-deduplicate them into the Grand Unified Mega-Corpus. Every experiment, qualitative generation sample, and bug mitigation is documented in our Reflective Learning Journal ([`reports/LEARNING_JOURNAL.md`](file:///Users/macbookpro/Documents/Coding/Ashesi%20Uni/Natural%20Language%20Processing/nlp-group-1-prosit-1/reports/LEARNING_JOURNAL.md)).

Finally, our statistical count matrices support direct export into standard ARPA language modeling format files, enabling Ankora's engineering team to compile our trained Ewe language models directly into Weighted Finite-State Transducers (WFSTs) for ultra-fast, microsecond-latency speech recognition decoding on edge devices.
