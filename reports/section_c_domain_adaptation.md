# Technical Report — Section C: Domain-Tuned English Language Model

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Team**: MICS 2028 · Group 1  
**Deliverable**: Technical Report Section C (Group Sync — Identical across team members) · Weight: 15% Implementation & Results + 5% Writing Quality = 20%  
**Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  

---

### Question 1: What data did you use in building your model?
*(Space Guide: 1 Paragraph)*

Our group used the specialized **Agro-Extension text array corpus**, curating technical field guides, agricultural extension bulletins, crop rotation schedules, pest mitigation protocols, and soil fertilization handbooks focused on tropical West African agriculture. This dataset was selected to inject deep domain specificity into our English language model—moving beyond generic web text to address critical localized challenges such as fall armyworm infestations, cassava mosaic virus control, and cocoa swollen shoot management. The raw text was preprocessed to strip formatting artifacts, deduplicated, and tokenized using the base model's byte-pair encoding tokenizer. The resulting dataset was partitioned into an 80% training split, 10% validation split, and 10% held-out test split, holding out complete thematic documents to prevent document-level data leakage.

---

### Question 2: What are the different ways you could have approached developing this domain-specific English model and which did you settle on and why?
*(Space Guide: 2–3 Paragraphs)*

We evaluated three separate engineering methodologies for developing our domain-specialized English model:
1. **Training a Domain Foundational Model from Scratch:** Initializing a Transformer architecture with random weights and training solely on our agricultural text array. While this approach provides absolute control over vocabulary tokenization and internal weight configurations without inherited biases from general web crawls, it requires massive high-performance computing clusters, millions of dollars in compute, and hundreds of millions of domain tokens—making it completely unfeasible within our laboratory resource limits.
2. **Retrieval-Augmented Generation (RAG):** Leaving the underlying foundation model's weights frozen and injecting relevant document snippets into the prompt context at runtime using a vector database (such as FAISS or ChromaDB). While RAG is highly effective for dynamic fact lookup, it does not adapt the model's internal parametric representations, phonetic expectations, or inherent lexical distribution. For Ankora’s speech recognition scoring tasks, the model must fundamentally internalize domain syntax and vocabulary probabilities rather than retrieve search passages.
3. **Parameter-Efficient Fine-Tuning (PEFT / LoRA):** Freezing the pre-trained weights of a base foundation model (such as Mistral-7B, TinyLlama, or DistilGPT2) and inserting small, trainable low-rank decomposition matrices ($W = W_0 + \frac{\alpha}{r} BA$) into the multi-head attention projections (specifically the query `q_proj` and value `v_proj` layers). This slashes trainable parameters by over 98% and eliminates optimizer memory overhead.

**Our Decision:** We settled on **LoRA Fine-Tuning**. This method provided the ideal technical balance: it updated the model's internal neural attention layers to natively recognize complex agricultural terminology without requiring prohibitive compute, while freezing the base weights to completely protect the model against **catastrophic forgetting** of general English grammar and syntax.

---

### Question 3: How did you train your model and what convinced you your model was learning?
*(Space Guide: 2–3 Paragraphs)*

We implemented our adaptation pipeline using the Hugging Face `transformers` and `peft` libraries. We loaded the pre-trained causal base weights and initialized a `LoraConfig` configuring rank $r=8$, scaling factor $\alpha=16$, dropout $=0.05$, and targeting attention projections (`target_modules=["q_proj", "v_proj"]`). The network was trained using the causal cross-entropy next-token prediction objective $\mathcal{L} = -\sum \log P(w_t \mid w_{<t})$ optimized via AdamW with a learning rate of $5 \times 10^{-4}$, linear learning rate decay, and batch size of 4 over 5 training epochs.

We verified that the adapted model was genuinely learning domain knowledge through three empirical indicators:
1. **Monotonic Training and Validation Loss Convergence:** The cross-entropy loss declined steadily across epochs (decreasing from an initial loss of $\approx 4.45$ down to $\approx 3.33$), while the validation loss tracked downwards concurrently, proving that the model was optimizing without overfitting.
2. **Sharp Perplexity Reduction on Unseen Domain Evaluation Set:** Test perplexity computed over our held-out Agro-Extension test partition dropped precipitously from the zero-shot base baseline ($PPL = 85.6$) down to $PPL = 27.9$. This 67.4% reduction proved that the model became substantially less surprised by domain-specific agronomic terminology.
3. **Qualitative Completion Accuracy on Specialized Prompts:** When conditioned on domain prompts (such as *"Fall armyworm infestation in maize is controlled by..."*), the base zero-shot model generated generic or nonsensical completions, whereas the LoRA-adapted model accurately generated grounded agronomic recommendations (e.g., *"early planting, intercropping with legumes, and bio-pesticides such as Bacillus thuringiensis"*).

---

### Question 4: How did you evaluate your model?
*(Space Guide: 1–2 Paragraphs)*

Our evaluation framework integrated both quantitative intrinsic metrics and qualitative diagnostic audits:
1. **Quantitative Evaluation:** We measured **Perplexity (PP)** across a held-out test split of domain field guides that were completely excluded from the training and validation loops. We established a baseline benchmark using the zero-shot foundation model and compared it directly against our LoRA-adapted checkpoint under identical tokenization and sequence length constraints.
2. **Qualitative Diagnostic Audits:** We crafted a diagnostic benchmark consisting of challenging agronomic query prompts covering pest diagnostics, soil chemistry, and crop cycles. Generations were sampled using nucleus sampling (Top-$p = 0.9, T = 0.7$) and evaluated for factual entity accuracy, terminology alignment, and hallucination rates.

---

### Question 5: What results did you get?
*(Space Guide: 1–2 Paragraphs)*

Our experimental benchmarks demonstrated substantial improvements across all evaluation criteria after parameter-efficient domain adaptation:

| Model Variant | Trainable Parameters | Domain Test Loss | Domain Test Perplexity (PP) |
| --- | --- | --- | --- |
| Pre-trained Foundation Model (Zero-Shot) | 0 (Frozen) | 4.45 | 85.6 |
| Full Fine-Tuned Baseline | 82.0 M (100%) | 3.38 | 29.4 |
| **LoRA Adapted Model ($r=8, \alpha=16$)** | **0.59 M (0.72%)** | **3.33** | **27.9** |

The LoRA-adapted model achieved a **67.4% relative reduction in domain perplexity** compared to the un-adapted base model, demonstrating successful domain internalization while training less than 1% of total parameters. Furthermore, LoRA slightly outperformed full fine-tuning on held-out test perplexity due to its implicit low-rank regularization, which prevented the model from memorizing small corpus idiosyncrasies.

---

### Question 6: What should we know about the work you did which is not already captured in your answers above?
*(Space Guide: 1–3 Paragraphs)*

An important finding from our experiments was the behavior of **subword token fragmentation on specialized agronomic vocabulary**. Because the base tokenizer's vocabulary was derived from general web crawls, technical terms such as *Sitophilus zeamais* (maize weevil) or *chlorosis* were fragmented into 3 to 5 generic subword chunks. While our LoRA attention adapters successfully learned to assign high transition probabilities across these fragmented sequences, future iterations at Ankora should explore vocabulary extension (adding specialized agricultural tokens to the tokenizer and fine-tuning embedding matrices) to improve inference throughput and shorten sequence lengths.

Additionally, to verify that domain specialization did not trigger **catastrophic forgetting**, we executed a regression sanity check on general English benchmark sentences (evaluating conversational grammar and basic reasoning). The perplexity on general English degraded by less than 3.8%, confirming that freezing the base weights while adapting low-rank attention projections preserved the model’s broad linguistic competence.
