# Technical Report, Section C: Domain-Tuned English Language Model

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Team**: MICS 2028 · Group 1  
**Deliverable**: Technical Report Section C (Group Sync, identical across team members) · Weight: 15% Implementation & Results + 5% Writing Quality = 20%  
**Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  
**Numbers**: every figure below is copied from `results/section_c_llm/lora_results.json` or `results/section_c_llm/decoding_benchmark.json` (see `reports/claims_table.md`).

---

### Question 1: What data did you use in building your model?
*(Space Guide: 1 Paragraph)*

We used the **KisanVaani agricultural question-answering corpus** (`KisanVaani/agriculture-qa-english-only` on Hugging Face): English farming questions with written answers on crops, pests, soil and livestock. The raw corpus has 22,615 rows but only **2,212 distinct questions**, because most questions are repeated. We therefore kept one row per question *before* shuffling, so no test question can also appear in training, and split the 2,212 pairs 80/10/10 (seed 42) into 1,769 training, 221 validation and 222 test pairs (`src/section_c_llm/prepare_data.py`); the training script re-checks this and finds 0 test questions in training or validation. Each pair is written as `Question: ...` followed by `Answer: ...` on the next line. To fit a laptop CPU budget we trained on the first 500 training pairs; validation and test use all of theirs. An earlier version of this experiment split the raw rows without deduplication, and 16 of its 100 test pairs were word-for-word copies of training pairs, so all results below replace that version.

---

### Question 2: What are the different ways you could have approached developing this domain-specific English model and which did you settle on and why?
*(Space Guide: 2–3 Paragraphs)*

We considered three approaches:
1. **Training a model from scratch** on agricultural text. This gives full control over the vocabulary and weights, but 2,212 question-answer pairs are far too little text to learn English itself, let alone a domain; it would need orders of magnitude more data and compute than we have.
2. **Retrieval-Augmented Generation (RAG)**: keep a pretrained model frozen and paste relevant passages from a document index into the prompt at inference time. RAG is good at looking up facts, but it does not change the model's own probabilities for domain words and phrasing. For Ankora's speech recognition use, where the language model scores candidate transcripts, those probabilities are exactly what must change.
3. **Parameter-efficient fine-tuning with LoRA**: freeze the pretrained weights $W_0$ of distilgpt2 and learn a low-rank update $\Delta W = \frac{\alpha}{r} BA$ in the attention layer `c_attn`, GPT-2's single projection that produces the query, key and value vectors. With rank $r=8$ this trains **147,456 parameters, 0.18% of the model's 82,060,032**.

**We chose LoRA.** It adapts the model's probabilities to the domain with a parameter count small enough to train on a CPU, and because the pretrained weights stay frozen it changes general English less than full fine-tuning would. It does not leave general English untouched, though: we measured the cost (Question 5). We trained LoRA two ways on identical data: a **standard** adapter with the usual causal-LM loss on every token, and a **prompt-masked** adapter whose loss covers only the answer tokens.

---

### Question 3: How did you train your model and what convinced you your model was learning?
*(Space Guide: 2–3 Paragraphs)*

`src/section_c_llm/train_lora.py` loads `distilgpt2`, attaches LoRA ($r=8$, $\alpha=32$, dropout 0.05, target `c_attn`, with `fan_in_fan_out=True` because GPT-2 stores that layer as a `Conv1D`), and trains with the Hugging Face `Trainer`: next-token cross-entropy $\mathcal{L} = -\sum_t \log P(w_t \mid w_{<t})$, AdamW at learning rate $5 \times 10^{-4}$ with linear decay, batch size 8, 3 epochs, sequences truncated at 96 tokens, seed 42. The masked variant sets the label of every token up to and including `Answer:` to `-100`, which the loss ignores.

Three things convinced us the models were learning the domain:
1. **Validation loss fell every epoch**: standard (full text) 3.3963, 3.3072, 3.2874; masked (answer tokens only, so not comparable with the standard numbers) 3.3395, 3.3005, 3.2902.
2. **Held-out perplexity fell on the answers, not just on the template.** On the 222 test pairs, answer-token perplexity dropped from 37.77 (base) to 30.00 (standard) and 29.09 (masked). Scoring the answers alone matters: the standard model's full-text perplexity (28.13) is lower than its answer perplexity because, after training, the template tokens and the question phrasing are easier to predict than the answers.
3. **The samples changed shape.** Given `Question: why is crop rotation important in farming?`, the base model loops "crop rotation is important in farming." after every `Answer:`, while both adapters produce an answer-shaped explanation. The content is another matter (Question 6).

---

### Question 4: How did you evaluate your model?
*(Space Guide: 1–2 Paragraphs)*

**Quantitatively**, we computed token-weighted perplexity (total negative log-likelihood of the scored tokens divided by their number) for the base model and both adapters on the same 222 held-out pairs, in three views: the **full** question-and-answer text, the **answer tokens only** given the question, and **general English**, 200 paragraphs from the WikiText-2 test split, to measure what the adaptation costs outside the domain. An earlier version of our evaluation averaged per-batch losses and treated fragments of split answers as pairs; both are fixed.

**Qualitatively**, we sampled each model on three agricultural prompts with a fixed seed (42), temperature 0.7 and top-p 0.9, and read the answers for correctness as well as fluency. A separate benchmark (`src/section_c_llm/benchmark_decoding.py`, 4 prompts, 5 seeds per sampled strategy) measures how decoding settings change repetition, using Distinct-3: unique word trigrams divided by all word trigrams in an answer.

---

### Question 5: What results did you get?
*(Space Guide: 1–2 Paragraphs)*

| Model | Trainable parameters | Full Q&A perplexity | Answer-only perplexity | WikiText-2 perplexity |
| --- | ---: | ---: | ---: | ---: |
| distilgpt2 base | 0 | 56.08 | 37.77 | 73.19 |
| + LoRA, loss on all tokens (standard) | 147,456 (0.18%) | **28.13** | 30.00 | 78.44 |
| + LoRA, loss on answers only (masked) | 147,456 (0.18%) | 51.39 | **29.09** | 77.24 |

The standard adapter cuts full-text perplexity by 49.8% and answer perplexity by 20.6%. The answer-only figure is the cleanest measure of domain knowledge; much of the extra full-text gain comes from the question side, the template tokens and the way the questions are phrased. The masked adapter cuts answer perplexity by 23.0% but full-text perplexity by only 8.4%, because it was never trained to predict questions. Its 3% edge over the standard adapter on answers comes from a single seed and 500 training pairs, so we treat the two as comparable on answers. Both adapters cost general English: WikiText-2 perplexity rises 7.2% (standard) and 5.5% (masked). An earlier result of 62.38 to 29.33 was measured on the leaky test set described in Question 1 and is superseded, not comparable.

---

### Question 6: What should we know about the work you did which is not already captured in your answers above?
*(Space Guide: 1–3 Paragraphs)*

**Fluent is not the same as correct.** Of the nine seeded answers (three prompts, three models), at best one, the masked adapter's generic reply on crop rotation, is roughly right; the others loop, contradict themselves or are wrong. Asked how to control fall armyworm in maize, the standard adapter answers "The fall armyworm in maize affects the development of mites, insects and other insects." and the masked adapter "The fall armyworm in maize can occur through a variety of diseases, such as arachnoid, cloverworm, and coca." An 82-million-parameter model trained on 500 examples learns how an extension answer sounds long before it learns agronomy. A model like this can help score speech recognition hypotheses in the domain, but it must not generate advice for farmers.

**Decoding settings change repetition, not correctness.** Unpenalized sampling (temperature 0.7) sometimes loops; averaged over 4 prompts and 5 seeds its Distinct-3 is 78.9%. A repetition penalty of 1.3 or blocking repeated 3-grams raises it to 99.5 to 100%, but 3-gram blocking guarantees that by construction, and the penalty pushes text off topic: one penalized answer about armyworm begins "The common disease, the malaria parasite-borne chikungunya virus, is transmitted by soil moisture." We therefore make no production recommendation for a decoding strategy.

**Which objective to use depends on the job.** Prompt-loss masking is the standard choice for training an assistant to answer questions, but for Ankora's speech recognition setting the model scores whole transcripts, including farmers' spoken questions, and only the standard objective learns those (full-text perplexity 28.13 against 51.39).
