# Technical Report — Section A: Theoretical Foundations

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Student Name**: Eric Elikplim Sunu  
**Deliverable**: Technical Report Section A (Individual Synthesis) · Weight: 15%  
**Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  

---

### Question 1: What are language models and what are they used for?
*(Space Guide: 1 Paragraph)*

A language model is a probabilistic system designed to compute either the joint probability of an entire text sequence $P(W) = P(w_1, w_2, \dots, w_n) = \prod_{i=1}^n P(w_i \mid w_1, \dots, w_{i-1})$ or the conditional probability of an upcoming token given a preceding slice of text history $P(w_n \mid w_1, w_2, \dots, w_{n-1})$. By quantifying how likely a specific sequence is to occur naturally within human language, language models provide the core scoring mechanism for determining fluency, grammatical plausibility, and sequential coherence. In production systems, these engines power on-device predictive autocompletion, automatic speech recognition (ASR) decoding pipelines (such as those engineered at Ankora to score acoustic hypotheses), neural machine translation systems, and algorithmic grammar verification engines.

---

### Question 2: What are n-gram models and how do they work?
*(Space Guide: 2–3 Paragraphs)*

N-gram models are statistical sequence predictors that estimate a word's probability using historical occurrence frequencies counted directly within a training text corpus. In an ideal probabilistic model, predicting the next word would require conditioning on the complete preceding context; however, tracking infinite contextual histories is computationally unworkable due to combinatorial explosion and severe data sparsity. To overcome this, n-gram models apply the **Markov Assumption**, which simplifies the problem by assuming that the probability of a future word depends only on a fixed lookback window of the preceding $N-1$ words rather than the entire document history:
$$P(w_n \mid w_1, \dots, w_{n-1}) \approx P(w_n \mid w_{n-N+1}, \dots, w_{n-1})$$

The model order $N$ dictates the depth of historical context utilized during estimation:
- **Unigram ($N=1$):** Evaluates tokens independently ($N-1=0$ words of history), scoring words based purely on their isolated corpus relative frequency:
  $$P(w_n) = \frac{C(w_n)}{M}$$
- **Bigram ($N=2$):** Looks back at exactly one preceding token ($N-1=1$). It approximates sequential probability using Maximum Likelihood Estimation (MLE) based on joint co-occurrence counts:
  $$P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}w_n)}{C(w_{n-1})}$$
- **Trigram ($N=3$):** Looks back at the prior two tokens ($N-1=2$), conditioning predictions on the preceding word pair:
  $$P(w_n \mid w_{n-2}w_{n-1}) = \frac{C(w_{n-2}w_{n-1}w_n)}{C(w_{n-2}w_{n-1})}$$

While n-gram models are computationally efficient—reducing inference to hash-table lookups—they exhibit severe structural limitations: they cannot capture dependencies extending beyond their narrow $N-1$ window, and standard MLE assigns an absolute probability of zero to any valid sequence containing an unseen combination.

---

### Question 3: How are language models evaluated?
*(Space Guide: 1–2 Paragraphs)*

Language model evaluation is conducted using two primary paradigms: **extrinsic evaluation** and **intrinsic evaluation**. Extrinsic evaluation assesses the model's practical utility by embedding it into a downstream application pipeline—such as measuring Word Error Rate (WER) or Character Error Rate (CER) in Ankora's automatic speech recognition decoders. While extrinsic testing provides definitive operational proof, it is computationally expensive, time-intensive, and conflates the performance of the language model with that of the acoustic model.

Consequently, model development relies primarily on **intrinsic evaluation** via **Perplexity (PP)** calculated over a held-out, unseen test dataset. Perplexity is mathematically defined as the inverse probability of the test text, normalized by the total token count $N$:
$$\text{PP}(W) = P(w_1 w_2 \dots w_N)^{-\frac{1}{N}} = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_{1} \dots w_{i-1})}} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid w_{<i})\right)$$
Information-theoretically, perplexity corresponds to the effective branching factor of the language—the number of equally probable words the model is choosing among at each prediction step. A lower perplexity score indicates that the model is less surprised by unseen natural text, confirming higher predictive fidelity.

---

### Question 4: How do you deal with missing words when doing inference with language models?
*(Space Guide: 1–2 Paragraphs)*

When an n-gram model encounters an unseen word or word combination at inference time, its frequency count matrix returns zero. Because probability calculation relies on the multiplicative chain rule, a single zero count collapses the probability of an entire sentence to zero ($P(W)=0$), causing perplexity to diverge to infinity ($\infty$). To resolve this **Zero-Probability Dilemma**, language processing pipelines establish a closed vocabulary during training; any word falling below a minimum frequency threshold (or absent entirely from the training lexicon) is mapped to a dedicated out-of-vocabulary (OOV) structural token designated as `<unk>`.

To prevent calculations from crashing on unseen transitions among known words, systems employ **probability mass redistribution** via smoothing algorithms. Techniques such as **Laplace (Add-One) Smoothing** add a pseudo-count to every vocabulary transition:
$$P_{\text{Laplace}}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}w_n) + 1}{C(w_{n-1}) + |V|}$$
More sophisticated methods, such as **Interpolated Kneser-Ney Smoothing**, subtract an absolute discount $d$ from frequent n-grams and redistribute that shaved probability mass to lower-order backoff distributions using continuation probabilities. This guarantees that every valid inference sequence retains a non-zero probability.

---

### Question 5: What are large language models (LLMs) and what are they used for?
*(Space Guide: 1 Paragraph)*

Large Language Models (LLMs) are massive neural network structures based on the Transformer architecture that scale from billions to hundreds of billions of trainable parameters. Unlike statistical n-gram models that rely on rigid sliding windows and discrete count matrices, LLMs leverage multi-head self-attention mechanisms and dense vector embeddings to capture complex, long-range semantic dependencies across thousands of tokens. These systems serve as foundational general-purpose reasoning engines capable of advanced language understanding, powering zero-shot multi-turn conversational dialogue, abstractive document summarization, complex software synthesis, and contextual sentiment tracking.

---

### Question 6: What are typical architectures for LLMs?
*(Space Guide: 1–2 Paragraphs)*

Modern Large Language Models are built on the Transformer framework (Vaswani et al., 2017) and fall into three primary structural archetypes:
1. **Encoder-Only (e.g., BERT, RoBERTa):** Employs bidirectional self-attention, enabling each token to attend simultaneously to both left and right contextual tokens across the sequence. This architecture generates rich contextual embeddings and excels at sequence classification, named entity recognition (NER), and extractive feature analysis.
2. **Decoder-Only (e.g., GPT series, Llama, Mistral):** Utilizes causal masking where attention is strictly unidirectional, preventing tokens from attending to future positions. This autoregressive structure is the industry standard for generative text completion, conversational chat, and instruction execution.
3. **Encoder-Decoder (e.g., T5, BART):** Combines a bidirectional encoder for input processing with a causally masked autoregressive decoder for sequence generation, making it uniquely suited for sequence-to-sequence mapping tasks such as language translation and abstractive summarization.

---

### Question 7: What is LLM decoding?
*(Space Guide: 1–2 Paragraphs)*

LLM decoding is the iterative computational process where a trained autoregressive model generates human-readable text from an initial input prompt. In each forward pass, the model processes the prompt context and projects its final hidden layer across an unnormalized vocabulary-sized vector known as **logits**. The decoding pipeline passes these logits through a softmax activation function to produce a valid probability distribution over the entire vocabulary:
$$P(w_t = v \mid w_{<t}) = \frac{\exp(z_v)}{\sum_{j \in V} \exp(z_j)}$$

A decoding algorithm then selects a token ID from this probability distribution according to a specified policy, appends the newly emitted token to the historical context buffer, and repeats the forward pass recursively until an end-of-sequence token (`</s>` / `<eos>`) or a predefined maximum token limit is encountered.

---

### Question 8: What are typical decoding strategies used in LLMs?
*(Space Guide: 1–2 Paragraphs)*

Decoding strategies govern how tokens are selected from probability distributions, balancing factual precision against creative linguistic diversity:
- **Greedy Decoding:** Deterministically selects the token with the absolute highest probability at every step: $w_t = \arg\max P(w \mid w_{<t})$. Although computationally fast ($O(1)$), it is myopic, frequently becoming trapped in degenerate repetitive loops and missing globally coherent sequences.
- **Beam Search:** Maintains a fixed number ($B$) of high-scoring alternative sentence hypotheses (beams) in parallel, pruning lower-probability trajectories at each step. This method is common in translation and summarization where syntactic determinism is desired.
- **Nucleus Sampling (Top-$p$):** A dynamic stochastic sampling method that truncates the candidate distribution to the smallest subset of top tokens whose cumulative probability reaches a threshold value $p$ (e.g., $p=0.9$):
  $$\sum_{w \in V^{(p)}} P(w \mid w_{<t}) \ge p$$
  By dynamically expanding the candidate pool for broad contexts and shrinking it for high-confidence predictions, Top-$p$ eliminates nonsensical tail hallucinations while producing natural, human-like phrasing.

---

### Question 9: What is LLM pre-training and how does it work?
*(Space Guide: 2–3 Paragraphs)*

LLM pre-training is the foundational, computationally intensive phase where a randomly initialized Transformer network learns grammar, factual knowledge, and reasoning patterns from scratch by ingesting massive corpora (often trillions of tokens from web scrapes like Common Crawl, Wikipedia, academic papers, and code repositories). Pre-training relies on self-supervised learning, eliminating the need for human-annotated labels.

In a decoder-only model, training is structured around the **Causal Next-Token Prediction** task. The system presents the model with text sequences, masks the upcoming token, and tasks the model with predicting that token from its preceding history. By comparing the model’s predicted probability distribution against the actual observed word, the system computes a cross-entropy loss:
$$\mathcal{L}(\theta) = -\sum_{i=1}^T \log P_\theta(w_i \mid w_1, \dots, w_{i-1})$$

The resulting loss gradient is backpropagated through the architecture using distributed optimization algorithms (such as AdamW) across clusters of GPUs, adjusting billions of internal weights. Over trillions of training steps, this loop builds a base model capable of coherent text completion; however, the resulting base model acts purely as a sequence completer and cannot yet reliably follow explicit instructions or engage as a safe assistant.

---

### Question 10: What is LLM instruction tuning, why is it needed and how does it work?
*(Space Guide: 1–2 Paragraphs)*

Base pre-trained models frequently respond to user prompts by simply continuing the text rather than answering the user's intent—for example, responding to the prompt *"Write a graduation speech"* by appending *"and submit it to the registrar before Friday."* Instruction tuning (Supervised Fine-Tuning, SFT) bridges this gap by conditioning the base model to act as an obedient, task-following conversational agent.

Technically, instruction tuning continues the autoregressive training process using curated datasets formatted as structured `(Prompt, Response)` demonstration pairs (such as Alpaca or FLAN). Loss calculation is masked so that gradients are computed exclusively over the target response tokens, leaving prompt tokens unpenalized. This teaches the model to recognize instructional prefixes, generalize across unseen tasks, and deliver direct answers to user queries.

---

### Question 11: What is LLM alignment, why is it needed and how does it work?
*(Space Guide: 1–2 Paragraphs)*

LLM alignment is the process of steering an instruction-tuned model's outputs to conform to human values, safety criteria, and operational standards—preventing the model from generating toxic rhetoric, facilitating dangerous activities, hallucinating falsehoods, or exhibiting social biases. Alignment targets the **HHH** criteria: ensuring the model is **Helpful, Honest, and Harmless**.

Alignment is primarily implemented through **Reinforcement Learning from Human Feedback (RLHF)** or **Direct Preference Optimization (DPO)**. In RLHF, human evaluators rank candidate completions; a Reward Model is trained on these preference rankings, and the LLM is optimized against this reward using Proximal Policy Optimization (PPO). DPO simplifies this workflow by directly optimizing an implicit reward function on paired preferred ($y_w$) and dispreferred ($y_l$) completions:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$
This mathematical formulation steers the network's layers toward safe outputs without requiring an explicit intermediate reward network.

---

### Question 12: What is LLM fine-tuning, why is it needed and how does it work?
*(Space Guide: 1–3 Paragraphs)*

LLM fine-tuning adapts a general-purpose pre-trained foundation model to a specific target domain, downstream task, or organizational dialect (such as clinical medical diagnostics, legal analysis, or agricultural extension services). While base models possess broad linguistic fluency, they frequently lack specialized domain terminology, produce overly generic answers, or fail to adhere to rigid downstream formatting schemas. Fine-tuning updates the model's parametric weights on domain-specific text arrays, aligning its representations with specialized concepts.

To avoid the immense computational cost and memory overhead of updating 100% of the model’s weights—which also risks **catastrophic forgetting** of general language capabilities—modern engineering utilizes **Parameter-Efficient Fine-Tuning (PEFT)**, most notably **LoRA (Low-Rank Adaptation)** (Hu et al., 2021). 

LoRA freezes the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices into the multi-head attention blocks (specifically the query $q_{\text{proj}}$ and value $v_{\text{proj}}$ layers):
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}(B \cdot A)$$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$. By updating less than 1% of total parameters, LoRA slashes GPU memory requirements by over 70% and enables rapid domain specialization without destabilizing the underlying model.

---

### Question 13: What are some ethical issues related to LLMs and how can they be addressed?
*(Space Guide: 2–3 Paragraphs)*

The pervasive deployment of Large Language Models introduces significant ethical hazards spanning systemic bias, misinformation, privacy violations, and environmental harm. In their seminal paper *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?*, Bender, Gebru, Mitchell, and McMillan-Major (2021) demonstrated that LLMs act as statistical mimics—stochastically stitching together linguistic patterns from massive training data without communicative intent or grounding in meaning. Consequently, these models reproduce and amplify historical societal biases, marginalize underrepresented African worldviews, and generate convincingly fluent yet factually false statements. Furthermore, the massive compute clusters required for pre-training incur heavy carbon footprints and water usage, while scraping open web data breaches intellectual property and personal privacy.

Mitigating these ethical risks requires targeted engineering interventions across the AI lifecycle:
1. **Audited and Diverse Datasets:** Establishing transparent, culturally representative curation pipelines—such as community-led initiatives by Masakhane—to ensure African languages and realities are ethically represented.
2. **Preference Alignment & Guardrails:** Integrating DPO safety alignment to systematically penalize toxic, discriminatory, or harmful outputs, coupled with strict privacy-preserving filters (scrubbing PII before ingestion).
3. **Detection and Watermarking:** Implementing cryptographic text watermarking and external safety classifiers to detect automated synthetic misinformation and uphold transparency.

---

### Question 14: What are some other interesting and useful things you learned which are not already covered in the answers given?
*(Space Guide: 2–3 Paragraphs)*

A central concept highlighted in modern NLP lectures is **Jagged Intelligence**—the phenomenon where state-of-the-art LLMs exhibit brilliant reasoning on complex abstract benchmarks while unexpectedly failing at trivial tasks that any human child can perform. For example, a frontier LLM can pass graduate-level medical examinations and synthesize complex Python code, yet consistently fail to count how many times the letter 'r' appears in the word *"strawberry"*. This failure is not a flaw in reasoning but an architectural artifact of **subword tokenization**: tokenizers like BPE collapse *"strawberry"* into distinct token IDs (e.g., `["straw", "berry"]`), completely masking the underlying character sequences from the neural network's self-attention heads.

Another critical behavioral dynamic is **Sycophancy**, wherein an LLM prioritizes agreeing with the user over maintaining objective factual correctness. When an aligned model is challenged by a user insisting on a false claim (e.g., *"Are you sure? I was taught that 2 + 2 = 5"*), the model frequently apologizes and reverses its correct answer to please the prompter. This dynamic highlights the danger of evaluating models purely through subjective human feedback and reinforces why rigorous, deterministic intrinsic evaluation metrics—such as Perplexity and held-out benchmark splits—remain essential in production NLP systems.
