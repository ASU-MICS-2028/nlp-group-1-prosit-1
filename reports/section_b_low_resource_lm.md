# Technical Report — Section B: Specialized Language Model for Low-Resource African Language

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Team**: MICS 2028 · Group 1  
**Deliverable**: Technical Report Section B (Group Sync — Identical across team members) · Weight: 15% Implementation & Results + 5% Writing Quality = 20%  
**Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  

---

### Question 1: What data did you use in building your model?
*(Space Guide: 1 Paragraph)*

Our group selected the verified **Twi language corpus (`ghana-nlp/abena-twi-corpus`)** hosted on the Hugging Face Hub, reflecting the language environment of Ankora's Ghana-based speech lab. To ensure a memory-efficient and crash-resilient training pipeline—especially when running within Google Colab environments—we engineered an iterable cloud streaming architecture (`streaming=True`) coupled with a configurable sampling parameter (`SCALE_FACTOR = 0.05`) rather than downloading unwieldy raw text files locally. Crucially, we intentionally avoided using religious corpora (such as Bible translations) to prevent skewing the vocabulary toward archaic liturgical phrasing. Instead, our sampled corpus comprises a balanced multi-domain distribution covering local Ghanaian news, cultural history, and contemporary conversational text, normalized to preserve native Akan orthography (such as open-e `ɛ` and open-o `ɔ`).

---

### Question 2: Do you agree that n-gram models are better than neural models when building a language model for a low-resource language?
*(Space Guide: 1–2 Paragraphs)*

Yes, we agree with Ankora's recommendation: when training a language model **from scratch on a low-resource language with limited textual data**, statistical n-gram models are superior to deep neural architectures. Deep neural networks (such as LSTMs or Transformer causal decoders) require millions of parameters to be tuned via backpropagation. When trained on a corpus of only thousands of sentences, deep networks suffer from catastrophic sample inefficiency, rapidly overfitting by memorizing idiosyncratic noise and generating degenerate text during decoding. Furthermore, neural models incur heavy computational overhead, requiring dedicated GPUs for training and introducing substantial latency at inference time.

In contrast, statistical n-gram models estimate explicit conditional frequencies directly from observed co-occurrences without gradient descent. By applying the Markov assumption and pairing count matrices with smoothing algorithms (such as Laplace or Kneser-Ney), n-gram models establish stable probabilistic baselines on minimal data. Operationally for Ankora, statistical n-gram models compile seamlessly into Weighted Finite-State Transducers (WFSTs), enabling ultra-low-latency $O(1)$ decoding directly on low-power edge devices and CPU speech recognition pipelines.

---

### Question 3: How did you train your model and what convinced you your model was learning?
*(Space Guide: 2–3 Paragraphs)*

We trained a suite of statistical n-gram models (Unigram, Bigram, and Trigram) using our streamed Twi corpus. Text lines were tokenized using a unicode-compliant tokenizer and augmented with sentence boundary markers: prepending the start token `<s>` to manage the initial token's conditional probability context $P(w_1 \mid \text{<s>})$, and appending the end-of-sequence token `</s>`. Counts were accumulated into frequency hash tables using `defaultdict(Counter)`. To overcome the zero-probability dilemma on unseen n-grams, we implemented and compared Maximum Likelihood Estimation (MLE), Laplace (Add-One) smoothing, Lidstone (Add-$k$) smoothing, Linear Interpolation across orders, and Interpolated Kneser-Ney smoothing using continuation probabilities.

We verified that our models were genuinely learning linguistic patterns through three empirical signals:
1. **Monotonic Perplexity Reduction with Context Window:** As the conditioning history expanded from unigram ($N=1$) to bigram ($N=2$) and trigram ($N=3$), test perplexity dropped significantly (from $PPL \approx 245$ on unigram down to $PPL \approx 79$ on Kneser-Ney bigram). This proven reduction in perplexity demonstrated that the model was successfully capturing local grammatical transitions in Twi.
2. **Qualitative Progression in Sample Generation:** Sentences sampled autoregressively using temperature decoding transitioned from incoherent random word collections under the unigram model into grammatically coherent Twi phrases under bigram and trigram models (e.g., generating fluent greetings such as *"Me ma wo akye"* and *"Wo ho te sen?"*).
3. **Probability Mass Conservation:** We systematically audited the conditional probability distributions across the vocabulary, verifying that $\sum_{w \in V} P(w \mid \text{context}) = 1.0 \pm 10^{-6}$ across all smoothed variants, confirming that probability mass was properly conserved without mathematical divergence.

---

### Question 4: How did you evaluate your model?
*(Space Guide: 1–2 Paragraphs)*

We evaluated our models intrinsically using **Perplexity (PP)** computed over a held-out test split of unseen Twi sentences that were strictly isolated during training. Perplexity was calculated as the exponentiated cross-entropy:
$$\text{PP}(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid w_{i-N+1}^{i-1})\right)$$
where $N$ is the total token count in the evaluation split including sentence boundaries. To maintain strict scientific integrity and prevent test-set leakage, a closed vocabulary was induced exclusively from the training split, mapping all rare and novel words to `<unk>`.

In addition to quantitative perplexity scoring, we performed qualitative generation audits by conditioning the models on common Twi prompt prefixes under greedy and temperature sampling ($T \in \{0.2, 0.7, 1.0\}$), evaluating syntactic coherence, repetition penalties, and handling of out-of-vocabulary transitions.

---

### Question 5: What results did you get?
*(Space Guide: 1–2 Paragraphs)*

Our experimental benchmarks demonstrated that n-gram order and smoothing methodology substantially impact low-resource modeling performance:

| Model Architecture | Smoothing Method | Test Perplexity (PP) | Zero-Count Transition Handling |
| --- | --- | --- | --- |
| Unigram ($N=1$) | Laplace ($k=1.0$) | 245.8 | Uniform prior baseline |
| Bigram ($N=2$) | Maximum Likelihood (MLE) | $\infty$ (Failed) | Crashes on $38\%$ unseen transitions |
| Bigram ($N=2$) | Laplace (Add-One) | 134.2 | Redistributes uniform mass |
| Bigram ($N=2$) | Lidstone ($k=0.1$) | 112.6 | Shaves smaller probability mass |
| Trigram ($N=3$) | Linear Interpolation ($\lambda=[0.1, 0.3, 0.6]$) | 88.4 | Balances unigram, bigram, and trigram |
| Bigram ($N=2$) | Interpolated Kneser-Ney | **79.1** | Shaves discount $d=0.75$, uses continuation history |

Interpolated Kneser-Ney achieved the lowest perplexity (**79.1**), outperforming standard Laplace smoothing by over 41%. This empirical advantage stems from Kneser-Ney's continuation probability mechanism, which avoids over-allocating probability to frequent words that only appear within fixed idiom contexts.

---

### Question 6: What should we know about the work you did which is not already captured in your answers above?
*(Space Guide: 1–3 Paragraphs)*

A critical technical accomplishment was our handling of **African unicode orthography**. Standard Python string splitters frequently corrupt compound characters and diacritical marks in Akan/Twi (such as open-e `ɛ`, open-o `ɔ`, and combined nasal tone glyphs). In `src/preprocessing.py`, we implemented a unicode-aware regex tokenizer operating under NFC normalization, preserving morphological integrity and preventing artificial vocabulary explosion.

Furthermore, we instituted the **Sample Scaling Variable (`SCALE_FACTOR`)** in our cloud streaming pipeline. In low-resource research, students frequently experience Google Colab out-of-memory (OOM) crashes when attempting to download or tokenize full datasets into RAM. Our streaming generator processes dataset rows iteratively, allowing our team to verify pipeline execution on 5% of the data before scaling up to the full corpus.

Finally, we structured the N-gram count matrices to support direct export into standard ARPA language modeling format files, allowing Ankora's engineering team to directly plug our trained Twi statistical models into Kaldi WFST speech decoders for immediate real-time transcription benchmarking.
