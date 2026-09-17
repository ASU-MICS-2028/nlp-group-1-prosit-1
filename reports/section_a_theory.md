# Technical Report — Section A: Theoretical Foundations

**Course**: ICS554 Natural Language Processing · MICS 2028 · Group 1  
**Author**: [Your Name / Student ID]  
**Weight**: 15% of total grade  

---

### Question 1: What are language models and what are they used for?
*(Space constraint: $s = 1\text{ paragraph}$)*

A language model (LM) is a computational model that assigns a probability distribution over sequences of words or tokens, estimating $P(W) = P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$ through the chain rule of probability. By quantifying how fluent, coherent, or probable a given sequence is in natural human language, language models enable machines to differentiate grammatically plausible sentences from arbitrary combinations of vocabulary. In applied natural language processing and automatic speech recognition (ASR)—such as the systems engineered at Ankora—language models serve as fundamental decoders to resolve acoustic ambiguities, score candidate transcriptions, generate fluent translations in machine translation, power predictive text completion, and generate human-like text across conversational interfaces.

---

### Question 2: What are n-gram models and how do they work?
*(Space constraint: $2 \le s \le 3\text{ paragraphs}$)*

N-gram language models are statistical language models that approximate the joint probability of a text sequence by applying the Markov assumption: the probability of an incoming word depends only on the preceding $n-1$ words rather than the entire historical context, formally expressed as $P(w_t \mid w_1, \dots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \dots, w_{t-1})$. In a unigram model ($n=1$), words are assumed to occur independently ($P(w_t)$); in a bigram model ($n=2$), each word is conditioned on the immediately prior word ($P(w_t \mid w_{t-1})$); and in a trigram model ($n=3$), conditioning relies on the prior two words ($P(w_t \mid w_{t-2}, w_{t-1})$).

Training an n-gram model relies on Maximum Likelihood Estimation (MLE), computed by counting frequencies across a training corpus:
$$P_{MLE}(w_t \mid w_{t-n+1}^{t-1}) = \frac{C(w_{t-n+1}^{t-1}, w_t)}{C(w_{t-n+1}^{t-1})}$$
While computationally trivial—amounting to frequency counting and hash-table lookups—standard MLE fails drastically on unseen combinations, assigning zero probability to any valid sequence containing an unseen n-gram. To resolve this sparsity, smoothing techniques such as Laplace ($+k$), linear interpolation across varying orders, and Kneser-Ney discounting are required to redistribute probability mass to unseen events.

---

### Question 3: How are language models evaluated?
*(Space constraint: $1 \le s \le 2\text{ paragraphs}$)*

Language models are evaluated using both intrinsic and extrinsic metrics. The primary intrinsic metric is **Perplexity (PPL)**, which measures the inverse geometric mean probability assigned by the model to a held-out test corpus of $N$ tokens:
$$PPL(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid w_1, \dots, w_{i-1})\right) = 2^{H(W)}$$
where $H(W)$ is the cross-entropy of the corpus. Mathematically, perplexity represents the weighted average branching factor—the effective number of equally likely words the model must choose from at each step. Lower perplexity directly reflects higher predictive confidence and superior probabilistic modeling of natural text.

Extrinsically, language models are evaluated by measuring performance downstream within their target application. For Ankora’s speech recognition pipelines, the extrinsic benchmark is Word Error Rate (WER) or Character Error Rate (CER) of the combined acoustic-language decoder. In generation tasks, downstream evaluation encompasses BLEU, ROUGE, and task-specific accuracy benchmarks (e.g. MMLU, GSM8k for modern LLMs).

---

### Question 4: How do you deal with missing words when doing inference with language models?
*(Space constraint: $1 \le s \le 2\text{ paragraphs}$)*

Missing or out-of-vocabulary (OOV) words pose a critical challenge during inference because unseen vocabulary items cause standard models to either assign zero probability to sentences or fail during tokenization. In statistical n-gram modeling, the standard protocol involves defining a closed vocabulary prior to training. Any training word with frequency below a chosen threshold $k$ is replaced with a special unknown token, `<unk>`. During inference, any novel word not present in the lexicon is mapped to `<unk>`, allowing the model to estimate $P(\text{<unk>} \mid \text{context})$ without mathematical divergence. 

In modern neural language models, subword tokenization algorithms—such as Byte-Pair Encoding (BPE), WordPiece, or SentencePiece—largely eliminate true OOV scenarios. Rather than operating strictly at the whole-word level, words not seen during training are broken down into known constituent subword units or byte-level representations (e.g., decomposing "unaffordability" into `["un", "afford", "ability"]`), ensuring the model can always assign valid representations and compute probabilities for arbitrary strings.

---

### Question 5: What are large language models (LLMs) and what are they used for?
*(Space constraint: $s = 1\text{ paragraph}$)*

Large Language Models (LLMs) are deep autoregressive neural networks—predominantly based on the Transformer architecture—possessing billions to hundreds of billions of parameters trained on vast multi-terabyte corpora via self-supervised predictive objectives. By scaling parameters, dataset sizes, and compute (conforming to neural scaling laws), LLMs develop emergent capabilities such as in-context learning, multi-step reasoning, translation, code generation, and complex conversational reasoning. In industry and research, LLMs function as foundational general-purpose reasoning engines that are deployed for conversational assistants, automated programming, domain-specific decision support (e.g. medical triage, agricultural advisory), and cognitive agentic workflows.

---

### Question 6: What are typical architectures for LLMs?
*(Space constraint: $1 \le s \le 2\text{ paragraphs}$)*

Modern LLMs are overwhelmingly based on the Transformer architecture introduced by Vaswani et al. (2017), utilizing multi-head self-attention mechanisms to model bidirectional or unidirectional token relationships across sequences. These architectures fall into three primary archetypes: (1) **Decoder-only** (e.g., GPT-4, Llama 3, Mistral), which employ causal masking where each token can only attend to past positions, making them ideally suited for open-ended autoregressive generation; (2) **Encoder-only** (e.g., BERT, RoBERTa), which leverage bidirectional attention to process full sentence contexts simultaneously, excelling in classification and token tagging; and (3) **Encoder-Decoder** (e.g., T5, BART), which map an unmasked input sequence to a causally decoded target sequence, common in machine translation and abstractive summarization.

Contemporary generative LLMs almost universally adopt the decoder-only design due to its training efficiency and natural alignment with autoregressive sequence completion. Key modern architectural enhancements include Rotary Position Embeddings (RoPE), SwiGLU non-linear activation functions, Grouped-Query Attention (GQA) to optimize memory bandwidth during KV-cache decoding, and Root Mean Square Normalization (RMSNorm).

---

### Question 7: What is LLM decoding?
*(Space constraint: $1 \le s \le 2\text{ paragraphs}$)*

LLM decoding is the iterative inference process of generating a coherent sequence of text from a trained autoregressive model given an initial prompt context. In causal models, the forward pass produces a vector of unnormalized logits over the entire vocabulary for the next token position. Decoding encompasses transforming these logits into a probability distribution via the softmax function and applying a decision algorithm to select the token $w_t \sim P(w \mid w_{<t})$, appending it to the context, and repeating the cycle until an end-of-sequence token (`<eos>`) or maximum length is reached.

Because searching through all possible generation paths across an exponential search tree ($|V|^T$) is computationally intractable, decoding strategies act as heuristic search policies that balance generation fluency, factual coherence, diversity, and computational efficiency during token-by-token emission.

---

### Question 8: What are typical decoding strategies used in LLMs?
*(Space constraint: $1 \le s \le 2\text{ paragraphs}$)*

Decoding strategies are broadly bifurcated into **deterministic search** and **stochastic sampling** techniques:
1. **Greedy Search**: Selects the token with the highest probability at each step: $w_t = \arg\max P(w \mid w_{<t})$. While computationally cheap ($O(1)$), it is prone to repetitive loops and misses high-probability global paths.
2. **Beam Search**: Maintains the top $B$ highest cumulative probability candidate sequences (beams) at each step. While standard in translation and summarization, it often produces dull, unnatural output in open-ended generation.

To introduce creativity and eliminate repetitive degeneration, **stochastic sampling** methods modify logits:
- **Temperature Scaling ($T$)**: Divides logits by $T > 0$ prior to softmax; $T < 1.0$ sharpens distributions (favoring high-confidence tokens), while $T > 1.0$ flattens distributions (promoting diversity).
- **Top-$k$ Sampling**: Restricts sampling strictly to the $k$ most probable tokens, truncating improbable tail tokens.
- **Top-$p$ (Nucleus) Sampling**: Dynamically restricts the candidate pool to the smallest set of tokens whose cumulative probability exceeds threshold $p$ (e.g., $0.9$), adapting the candidate set size based on model confidence.

---

### Question 9: What is LLM pre-training and how does it work?
*(Space constraint: $2 \le s \le 3\text{ paragraphs}$)*

LLM pre-training is the foundational, compute-intensive phase of model development wherein a randomly initialized transformer network learns language structure, world knowledge, and reasoning primitives from massive, unlabelled textual corpora (often spanning trillions of tokens from web scrapes, books, Wikipedia, and code). Pre-training uses self-supervised learning, requiring no human annotations. In causal decoder architectures, the model is trained via Next-Token Prediction (autoregressive language modeling), maximizing the log-likelihood of the corpus tokens:
$$\mathcal{L}_{LLM}(\theta) = -\sum_{t=1}^T \log P_\theta(w_t \mid w_1, \dots, w_{t-1})$$

Optimization relies on distributed stochastic gradient descent (e.g. AdamW) orchestrated across hundreds to thousands of GPUs using 3D parallelism (data, tensor, and pipeline parallelism via libraries like Megatron-LM and DeepSpeed). The model adjusts its billions of weights via backpropagation through time, progressively reducing cross-entropy loss from high initial random perplexity to low values.

Through this objective, the network builds rich internal parametric representations of syntax, semantics, factual relationships, and logic. However, the resulting base model is strictly a sequence completer; it is not yet aligned to follow instructions or act as a helpful conversational agent.

---

### Question 10: What is LLM instruction tuning, why is it needed and how does it work?
*(Space constraint: $1 \le s \le 2\text{ paragraphs}$)*

Instruction tuning (or supervised fine-tuning, SFT) is the process of training a raw pre-trained base model on curated datasets of instruction-response pairs formatted as `(prompt, demonstration)`. A base model pre-trained solely on next-token prediction often completes prompts with continuations rather than direct answers—for example, responding to the prompt *"Write an essay on cocoa production"* by inventing more questions or continuing with related bibliography links. Instruction tuning bridges this gap by conditioning the model to act as an obedient, task-following assistant.

Technically, instruction tuning continues the standard autoregressive loss but applies masking such that loss gradients are backpropagated exclusively over the target response tokens, leaving prompt tokens unpenalized. Datasets such as FLAN, Alpaca, and ShareGPT expose the model to diverse task instructions (summarization, coding, roleplay, reasoning), enabling zero-shot generalization to novel user requests.

---

### Question 11: What is LLM alignment, why is it needed and how does it work?
*(Space constraint: $1 \le s \le 2\text{ paragraphs}$)*

LLM alignment is the process of steering model outputs to match human values, expectations, and ethical standards, commonly summarized by the "HHH" criteria: **Helpful, Honest, and Harmless**. Without alignment, models can generate toxic content, assist with dangerous queries (e.g., bioweapons, malware), hallucinate false facts with unwarranted confidence, or exhibit deep societal biases inherited from uncurated web training data.

Alignment is primarily achieved through **Reinforcement Learning from Human Feedback (RLHF)** or Direct Preference Optimization (DPO). In RLHF, human evaluators rank multiple model completions for a single prompt. A Reward Model is trained to predict these preference scores. The LLM is then optimized against this reward function using Proximal Policy Optimization (PPO), penalized with a Kullback-Leibler (KL) divergence term to prevent policy drift away from the initial SFT model. Alternatively, **DPO** bypasses the intermediate reward model by directly optimizing a closed-form implicit reward loss on paired preferred/rejected responses.

---

### Question 12: What is LLM fine-tuning, why is it needed and how does it work?
*(Space constraint: $1 \le s \le 3\text{ paragraphs}$)*

LLM fine-tuning is the secondary adaptation of a pre-trained (or instruction-tuned) model on a specialized, narrower dataset to improve performance on specific tasks, formatting styles, or specialized domains (such as clinical health, agriculture, or low-resource languages). While base models possess broad general competence, they often lack specialized domain vocabulary, produce generic responses, or fail to adhere to rigid downstream schemas. Fine-tuning allows an organization like Ankora to internalize domain terminology and target distributions directly into the model's parametric memory.

Fine-tuning can be conducted either via **Full Fine-Tuning**—where every parameter of the network is updated—or through **Parameter-Efficient Fine-Tuning (PEFT)**. Full fine-tuning is computationally prohibitive for large models, requires immense GPU VRAM, and risks catastrophic forgetting of general knowledge.

To resolve these constraints, PEFT techniques such as **LoRA (Low-Rank Adaptation)** freeze the original model weights $W_0 \in \mathbb{R}^{d \times k}$ and inject low-rank trainable decomposition matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$, yielding:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}(B \cdot A)$$
This slashes trainable parameters and gradient memory by over 95%, permitting rapid adaptation of LLMs on consumer hardware while matching full fine-tuning performance.

---

### Question 13: What are some ethical issues related to LLMs and how can they be addressed?
*(Space constraint: $2 \le s \le 3\text{ paragraphs}$)*

The deployment of LLMs raises critical ethical dilemmas across representational bias, hallucination, data sovereignty, environmental impact, and digital inequality:
1. **Algorithmic Bias & Representation**: Training corpora reflect historical biases and severe geographic imbalances. African contexts, languages, and indigenous knowledge systems are heavily underrepresented, leading to models that perpetuate Western cultural hegemony or misinterpret local idioms.
2. **Hallucination & Misinformation**: LLMs operate probabilistically without an innate grounding in factual truth. In critical sectors like healthcare and agriculture, generating plausible-sounding hallucinations can cause catastrophic physical or economic harm.
3. **Data Privacy, Consent & Environmental Footprint**: Pre-training often scrapes copyrighted literature and private data without creator consent, while training monolithic models consumes substantial water and electricity, exacerbating carbon emissions.

Addressing these issues requires multi-faceted interventions:
- **Inclusive Curation & Localized Auditing**: Active partnership with local language communities (such as Masakhane) to build ethically sourced, culturally validated African corpora.
- **Guardrails & Grounding**: Integrating Retrieval-Augmented Generation (RAG) with verified domain knowledge bases to eliminate hallucinations, and implementing toxic content classifiers and refusal policies.
- **Green AI & Watermarking**: Prioritizing smaller, efficient specialized models (SLMs) and parameter-efficient adaptation (PEFT) over brute-force scaling, accompanied by cryptographic watermarking to detect synthetic disinformation.

---

### Question 14: What are some other interesting and useful things you learned which are not already covered in the answers given?
*(Space constraint: $2 \le s \le 3\text{ paragraphs}$)*

One profound insight is the tension between pure parametric memory and non-parametric retrieval. While standard LLM development emphasizes scaling parameters to memorize facts, real-world specialized systems are increasingly shifting toward hybrid architectures combining lightweight Language Models with **Retrieval-Augmented Generation (RAG)** and **Test-Time Compute (Reasoning Chains)**. Rather than relying on static weights that become obsolete and hallucinate, coupling a domain-tuned model to an external vector database allows systems to quote exact sources, maintain strict auditability, and update knowledge dynamically without costly re-training.

Another critical finding is the **"curse of multilinguality"** and tokenizer fragmentation in low-resource settings. Standard LLM tokenizers (such as Llama's or GPT-4's BPE) are heavily optimized for English. When applied to African languages like Akan/Twi or Ewe, a single word is often fragmented into 4 to 8 byte-level tokens. This artificial sequence bloat drastically consumes context windows, increases inference latency by multiples, and degrades self-attention across long sentences. For resource-constrained speech labs like Ankora, training dedicated native tokenizers or utilizing character/subword n-gram statistical models is often far more practical and computationally viable than forcing English-centric LLM backbones.
