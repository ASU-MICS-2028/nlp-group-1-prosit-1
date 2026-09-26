# Team Presentation Outline (10-Minute Slide Deck)

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Topic**: Building and Adapting Language Models (Prosit 1)  
**Lab Context**: Ankora AI Research Lab (Ghana)  
**Deliverable Link**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  
*(Must appear on Slide 1 as required by assignment guidelines)*  
**Every number below is in `reports/claims_table.md` with the file and script that produce it.**  
**The deck**: `presentation/Prosit1_Language_Models.pptx` (11 slides on the Ashesi Presentation Red template), built from this plan by `python presentation/build_deck.py`, which reads every number from the result files. Its speaker notes hold what to say on each slide and where each number comes from.

---

### Slide 1: Title & Overview (0:00 - 1:00)
- **Title**: Specialized Language Modeling for Low-Resource Ewe (Èʋegbe) & Agro-Extension Domain Adaptation
- **Team**: MICS 2028 · Group 1
- **GitHub Repository**: `https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git` *(Direct link on slide)*
- **Context & Motivation**: Ankora AI Research Lab internship deliverables:
  1. Low-resource statistical language model for Ewe speech recognition decoders.
  2. Parameter-efficient fine-tuning (PEFT / LoRA) for specialized agricultural advisory.

---

### Slide 2: Problem Formulation & the N-Gram Approach (1:00 - 2:30)
- **N-grams vs neural, measured**: an LSTM trained on the same tokens loses to the n-gram on 420 sentences (7,864.84 vs 5,806.95 per word) but wins by 12% on the full 1.9M-word corpus (166.04 vs 189.07). The n-gram trains in minutes on a CPU and fits speech recognition decoders; each LSTM run took about 3 hours. So: n-grams first, neural once there is enough text.
- **The sparsity problem, in our own numbers**: on the Ewe test set, 10.5% of bigrams and 80.5% of 6-grams never occur in training. Maximum likelihood gives those probability 0.
- **The fix**: smoothing, from Laplace to interpolation to **interpolated Kneser-Ney**, which moves probability from rare events down to shorter contexts.

---

### Slide 3: The Ewe Model & 5-Tokenizer Study (2:30 - 4:30)
- **Data**: 123,511 unique sentences (1.87M training words) from four sources: sentence pairs, dictionary examples, University of Ghana Waxal speech transcriptions, and a large aligned corpus. A large part is Bible and Jehovah's Witnesses text (5.1% of sentences mention Yehowa), and that skews the model.
- **Ewe orthography**: NFC normalization, tone marks kept as part of the word (ɔ̃ has no single Unicode character), and lookalike letters fixed (Ð typed for Ɖ had split "ɖe" into two words).
- **Result 1, context length**: test perplexity falls 534.7 → 121.2 → 77.7 → 70.5 from unigram to 4-gram, then stays flat (69.3, 69.7). Longer context never hurts with Kneser-Ney; it stops helping at about 4 words.
- **Result 2, tokenizers (perplexity per word, the fair unit)**: BPE 189.1 < Ewe affixes kept 196.9 < words 202.2 < whitespace 261.6 < characters 447.1. Same ranking on every source dataset.
- **Visual**: `results/section_b_ngram/order_ablation.png`.

---

### Slide 4: Domain-Specific LLM Adaptation (Agro-Extension) (4:30 - 6:30)
- **Data**: KisanVaani agricultural Q&A, 22,615 rows but only **2,212 distinct questions**. We deduplicated before splitting: 1,769 train / 221 validation / 222 test, with 0 test questions seen in training.
- **Options considered**: training from scratch (far too little data), RAG (does not change the model's probabilities, which is what transcript scoring needs), **LoRA (chosen)**.
- **LoRA setup**: distilgpt2, rank 8, $\alpha=32$, on `c_attn` (the fused query/key/value projection): **147,456 trainable parameters, 0.18%** of the model. Two variants on identical data: loss on all tokens, and loss on answers only.

---

### Slide 5: Results (6:30 - 8:30)

| Model | Full Q&A | Answer only | General English (WikiText-2) |
|---|---:|---:|---:|
| distilgpt2 base | 56.08 | 37.77 | 73.19 |
| LoRA, loss on all tokens | 28.13 | 30.00 | 78.44 |
| LoRA, loss on answers only | 51.39 | 29.09 | 77.24 |

- **Domain knowledge**: answer perplexity down 20.6% (standard). The larger full-text drop also includes learning the question format.
- **Cost**: general-English perplexity up 7.2%. LoRA limits forgetting; it does not prevent it.
- **Masking** helps answers about as much (within noise) but never learns to predict questions, which speech recognition needs.
- **Fluent is not correct**: "The fall armyworm in maize affects the development of mites, insects and other insects." Decoding tricks remove loops (Distinct-3 78.9% → 100%, partly by construction) but not errors.
- **Visual**: `results/section_c_llm/lora_perplexity.png`.

---

### Slide 6: Key Takeaways (8:30 - 10:00)
1. **Smoothing decides everything.** With correct Kneser-Ney, Ewe n-grams improve up to about 4 words of context and then plateau; subword (BPE) units model the text best per word.
2. **LoRA adapts cheaply but does not make the model reliable.** 0.18% of the parameters cut answer perplexity by a fifth, at a measurable cost to general English, and the answers are fluent rather than correct.
3. **N-grams vs neural is a data-size question.** The n-gram wins on micro-data; an LSTM wins by 12% once there are 1.9M words, at hours of CPU instead of minutes.
4. **Verify before you write.** Our first draft reported a "breaking point" at $N=4$ and a tokenizer ranking that both came from bugs (probability lost for unseen contexts, padding counted as a word, per-token comparisons across tokenizers). Unit tests and committed scripts now guard every number.
- **Q&A / Panel Defense**: ready for the automated Viva Quiz on `clenam.ai` and panel questions.
