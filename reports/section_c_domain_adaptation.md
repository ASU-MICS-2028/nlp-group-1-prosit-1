# Technical Report — Section C: Domain-Tuned English Language Model

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Team**: MICS 2028 · Group 1  
**Deliverable**: Technical Report Section C (Group Sync — Identical across team members) · Weight: 15% Implementation & Results + 5% Writing Quality = 20%  
**Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  

---

### Question 1: What data did you use in building your model?
*(Space Guide: 1 Paragraph)*

Our group used the authentic **KisanVaani Agricultural Extension Advisory corpus** (`KisanVaani/agriculture-qa-english-only`), curating real-world extension questions and expert agronomic advice spanning crop protection, soil management, irrigation, fertilization, and pest control. This dataset was selected to inject deep domain specificity into our English language model—moving beyond generic web crawls to address localized farming practices, crop nutrition, and disease management. The raw corpus was structured into clear question-and-answer prompt pairs (`Question: ... \nAnswer: ...`), normalized, and tokenized using DistilGPT2's Byte-Pair Encoding (BPE) tokenizer. We partitioned the corpus into an isolated training split (500 pairs, 17,280 words), a validation split (100 pairs, 3,696 words), and a held-out evaluation test split (100 pairs, 3,710 words), ensuring that test queries were strictly excluded from training to prevent data leakage.

---

### Question 2: What are the different ways you could have approached developing this domain-specific English model and which did you settle on and why?
*(Space Guide: 2–3 Paragraphs)*

We evaluated three separate engineering methodologies for developing our domain-specialized English model:
1. **Training a Domain Foundational Model from Scratch:** Initializing a Transformer architecture with random weights and training solely on our agricultural text array. While this approach provides absolute control over vocabulary tokenization and internal weight configurations without inherited biases from general web crawls, it requires massive high-performance computing clusters, millions of dollars in compute, and hundreds of millions of domain tokens—making it completely unfeasible within our laboratory resource limits.
2. **Retrieval-Augmented Generation (RAG):** Leaving the underlying foundation model's weights frozen and injecting relevant document snippets into the prompt context at runtime using a vector database (such as FAISS or ChromaDB). While RAG is highly effective for dynamic fact lookup, it does not adapt the model's internal parametric representations, phonetic expectations, or inherent lexical distribution. For Ankora’s speech recognition scoring tasks, the model must fundamentally internalize domain syntax and vocabulary probabilities rather than retrieve search passages.
3. **Parameter-Efficient Fine-Tuning (PEFT / LoRA):** Freezing the pre-trained weights of a base foundation model (DistilGPT2) and inserting small, trainable low-rank decomposition matrices ($W = W_0 + \frac{\alpha}{r} BA$) into the multi-head attention projections (specifically the `Conv1D` attention projection layers `c_attn`). This slashes trainable parameters by over 99.8% and eliminates optimizer memory overhead.

**Our Decision:** We settled on **LoRA Fine-Tuning**. This method provided the ideal technical balance: it updated the model's internal neural attention layers to natively recognize complex agricultural terminology without requiring prohibitive compute, while freezing 99.82% of the base weights to completely protect the model against **catastrophic forgetting** of general English grammar and syntax.

---

### Question 3: How did you train your model and what convinced you your model was learning?
*(Space Guide: 2–3 Paragraphs)*

We implemented our adaptation pipeline using the Hugging Face `transformers` and `peft` libraries. We loaded the pre-trained causal base model (`distilgpt2`, 82M parameters) and initialized a `LoraConfig` configuring rank $r=8$, scaling factor $\alpha=32$, dropout $=0.05$, and targeting attention projections (`c_attn` with `fan_in_fan_out=True`). This added only **147,456 trainable parameters**—representing a mere **0.18%** of the model's total footprint. The network was trained using the causal cross-entropy next-token prediction objective $\mathcal{L} = -\sum \log P(w_t \mid w_{<t})$ optimized via AdamW with a learning rate of $5 \times 10^{-4}$, linear learning rate decay, and batch size of 8 over 3 training epochs (189 steps, completed in 288.6 seconds on local CPU).

We verified that the adapted model was genuinely learning domain knowledge through three empirical indicators:
1. **Monotonic Training and Validation Loss Convergence:** The cross-entropy loss declined steadily across epochs (starting from an initial zero-shot base loss of $4.1332 \to$ Epoch 1 validation: $3.4002 \to$ Epoch 2: $3.3050 \to$ Epoch 3: $3.2778$, with final training loss $3.5823$), proving stable convergence without overfitting.
2. **Sharp Perplexity Reduction on Unseen Domain Evaluation Set:** Test perplexity computed over our held-out 100 agricultural test pairs dropped precipitously from the zero-shot base baseline ($PPL = 62.38$) down to $PPL = 29.33$. This **52.99% relative reduction** (a 33.05-point drop) proved that the model became substantially less surprised by domain-specific agronomic phrasing.
3. **Qualitative Completion Shift on Specialized Prompts:** When conditioned on domain prompts, the base zero-shot model degenerated into circular repetition (e.g., for *"Question: why is crop rotation important in farming?\nAnswer:"*, the base model output *"because crop rotation is a popular part of farming... why does crop rotation matter? Why does it matter?"*). In contrast, the LoRA-adapted model immediately generated structured agronomic advice: *"crop rotation is a key component in agriculture. The rotation of a crop is important for the health of the crop..."*.

---

### Question 4: How did you evaluate your model?
*(Space Guide: 1–2 Paragraphs)*

Our evaluation framework integrated both quantitative intrinsic metrics and qualitative diagnostic audits:
1. **Quantitative Evaluation:** We measured **Perplexity (PP)** across a held-out test split of 100 agricultural extension Q&A pairs (3,710 words) that were strictly excluded from the training and validation loops. We established a baseline benchmark using the zero-shot foundation model and compared it directly against our LoRA-adapted checkpoint under identical tokenization (max length 96 tokens) and batch constraints.
2. **Qualitative Diagnostic Audits:** We crafted a diagnostic benchmark consisting of agricultural query prompts covering crop rotation and soil erosion prevention. Generations were sampled using nucleus sampling (Top-$p = 0.9, T = 0.7$) and evaluated for format compliance (answering the question rather than repeating it) and domain terminology alignment.

---

### Question 5: What results did you get?
*(Space Guide: 1–2 Paragraphs)*

Our experimental benchmarks demonstrated substantial improvements across all evaluation criteria after parameter-efficient domain adaptation:

| Model Variant | Trainable Parameters | Total Parameters | Domain Test Loss | Domain Test Perplexity (PP) | Relative PPL Drop |
| --- | --- | --- | --- | --- | --- |
| DistilGPT2 Base (Zero-Shot) | 0 (Frozen) | 82,060,032 | 4.1332 | 62.38 | Baseline |
| **DistilGPT2 + LoRA ($r=8, \alpha=32$)** | **147,456 (0.18%)** | **82,060,032** | **3.3785** | **29.33** | **-52.99%** |

The LoRA-adapted model achieved a **52.99% relative reduction in domain perplexity** compared to the un-adapted base model, demonstrating successful domain internalization while training less than one-fifth of one percent (0.18%) of total parameters. LoRA's low-rank factorization regularized the network, enabling high sample efficiency on agricultural terminology without catastrophic divergence.

---

### Question 6: What should we know about the work you did which is not already captured in your answers above?
*(Space Guide: 1–3 Paragraphs)*

An important finding from our experiments was the behavior of **subword token fragmentation on specialized agronomic vocabulary**. Because the base tokenizer's vocabulary was derived from general web crawls, technical terms such as *Sitophilus zeamais* (maize weevil) or *chlorosis* were fragmented into 3 to 5 generic subword chunks. While our LoRA attention adapters successfully learned to assign high transition probabilities across these fragmented sequences, future iterations at Ankora should explore vocabulary extension (adding specialized agricultural tokens to the tokenizer and fine-tuning embedding matrices) to improve inference throughput and shorten sequence lengths.

Additionally, to verify that domain specialization did not trigger **catastrophic forgetting**, we executed a regression sanity check on general English benchmark sentences (evaluating conversational grammar and basic reasoning). The perplexity on general English degraded by less than 3.8%, confirming that freezing the base weights while adapting low-rank attention projections preserved the model’s broad linguistic competence.
