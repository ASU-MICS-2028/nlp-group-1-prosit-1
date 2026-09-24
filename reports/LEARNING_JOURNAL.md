# Individual Reflective Learning Journal: Natural Language Processing (ICS554)

**Student Name**: Eric Elikplim Sunu  
**Degree**: Master's in Intelligent Computing Systems (MICS 2028)  
**Course**: ICS554 Natural Language Processing · Ashesi University  
**Project**: Prosit 1 (Ankora AI Research Lab: Language Modeling & Domain Adaptation)  
**Branch**: `eric` · **Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  

> **Status, 2026-09-22.** Sections 2 to 11 were rewritten after a verification audit of this repository. Section 12 records what we first believed, what it turned out to be, and why. Every number here is copied from a file in `reports/` written by a committed script; `reports/claims_table.md` lists which.

---

## 1. Problem Formulation & Epistemic Objectives

### 1.1 The Learning Purpose
In Problem-Based Learning (PBL), the objective of an engineering exploration is not merely to write functional code, but to understand how the algorithms behave, document where our understanding broke down under empirical pressure, and build mental models we can defend.

For this prosit, we set out to test the interaction between:
1. **Tokenization Granularity**: how the choice of unit (Character, Whitespace, Unicode Word, a rule-based Ewe stemmer, and Byte-Pair Encoding) changes vocabulary size $|V|$, unknown-word rate and sequence length.
2. **N-Gram Conditioning Horizon ($N \in [1, 2, 3, 4, 5, 6]$)**: how a longer Markov context changes perplexity and data sparsity, and whether longer contexts eventually stop helping.
3. **Multi-Source Data Harmonization**: how to clean, normalize (Unicode NFC), deduplicate, and merge four disparate text sources in a low-resource tonal language (**Ewe / Èʋegbe**) into one leak-free training corpus.

---

## 2. The Two Core Experiments (Unified Corpus)

All numbers in this section come from `reports/results_unified_all_tokenizers.json`, written by `scripts/run_multi_tokenizer_ablation.py`. Setup: 98,808 training sentences (1,874,130 words), scored on 4,000 validation and 4,000 test sentences; interpolated Kneser-Ney smoothing; tokens seen only once in training become `<unk>`; the best order $N$ is chosen on validation, never on test.

*The first draft of this section (2026-09-17) was written before any Ewe data was in the repository and contained illustrative tables (a vocabulary of about 1,250 words, a Kneser-Ney trigram perplexity of 71.4, and so on). Those numbers were never measured and have been removed; see section 12.*

### 2.1 Experiment Series A: The Tokenization Spectrum

| Tokenizer | What one token is | Vocabulary $|V|$ | Test tokens that are `<unk>` | Tokens per word |
|---|---|---:|---:|---:|
| Whitespace | text between spaces, punctuation attached (`loo!`) | 37,560 | 3.93% | 1.00 |
| Unicode Word | words and punctuation separately, tone marks kept | 26,489 | 1.93% | 1.16 |
| Ewe Stemmer (affixes kept) | Unicode words with one prefix and one suffix split off (`nu+ srɔ̃la +wo`) | 23,297 | 1.55% | 1.27 |
| BPE (150 merges) | subword pieces learned from the training split | 371 | 0.01% | 2.38 |
| Character | single characters, spaces kept as `▁` | 227 | 0.00% | 4.82 |

#### Key insights from Series A
- **Attached punctuation multiplies word types.** Whitespace splitting makes `loo`, `loo!` and `loo,` three different words: its vocabulary is 42% larger than Unicode Word's (37,560 vs 26,489) and its unknown-token rate twice as high (3.93% vs 1.93%).
- **Characters and BPE almost never meet an unknown token,** but pay with longer sequences (4.82 and 2.38 tokens per word), so the same $N$ covers much less text.
- **Per-token perplexity cannot rank tokenizers.** A character model chooses among 227 symbols per step, a word model among 26,489. Section 3, Discovery 4 compares them per word instead.

### 2.2 Experiment Series B: The N-Gram Lookback Horizon (Unicode Word)

| Order $N$ | Test n-grams never seen in training | KN validation PPL | KN test PPL | Equal-weight interpolation, test PPL | Superseded code, test PPL |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0% | 527.7 | 534.7 | 534.7 | 782.1 |
| 2 | 10.5% | 117.6 | 121.2 | 151.6 | 225.9 |
| 3 | 36.9% | 74.9 | 77.7 | 93.6 | 150.1 |
| 4 | 60.9% | 68.0 | 70.5 | 80.6 | 147.8 |
| 5 | 74.5% | 66.8 | **69.3** | 77.0 | 163.6 |
| 6 | 80.5% | 67.1 | 69.7 | 75.9 | 185.8 |

The last column is the old code's result (from the committed JSON before this rewrite, on slightly different data), kept only to show the shape that misled us.

- **Longer context never hurts once smoothing is correct.** Test perplexity falls from 534.7 to 70.5 by $N=4$, then stays flat (69.3 at $N=5$, 69.7 at $N=6$) even though 80.5% of test 6-grams never occur in training: Kneser-Ney hands the probability of an unseen long context down to the shorter ones.
- **The chosen order is within noise.** Validation picks $N=5$, but $N=4$, 5 and 6 differ by less than 2% on validation. The honest summary is "flat from $N=4$".
- **Kneser-Ney beats equal-weight interpolation at every $N \ge 2$** (77.7 vs 93.6 at $N=3$). Interpolation, now a proper distribution, also keeps improving with $N$; it is simply worse at deciding how much to trust a rare context.
- **The old curve turned upward after $N=4$ because of bugs**, not because of the language (Discovery 2).

---

## 3. Deep Technical Discoveries

### Discovery 1: The Combining Diacritic Trap
- **The phenomenon**: the punctuation-aware pattern `\w+|[^\w\s]` split `"nusrɔ̃lawo"` (students) into three tokens, `['nusrɔ', '̃', 'lawo']`, with the tilde alone in the middle; plain `\w+` silently dropped the tilde.
- **The root cause**: Python's `\w` matches what `str.isalnum()` accepts, letters and digits. A combining tilde (U+0303, Unicode category Mn) is neither. NFC normalization cannot rescue it, because Unicode has no single precomposed character for ɔ with a tilde: ɔ̃ stays two code points. This is documented behaviour, not a Python bug.
- **The fix**: the token pattern also accepts the combining-mark block U+0300 to U+036F: `r"[\w\u0300-\u036f]+|[^\w\s]"`. NFC still matters, so that letters that *do* have a precomposed form are always stored the same way.

### Discovery 2: The "Breaking Point" Was Our Smoothing, Not the Language
- **What we believed**: on each dataset, word n-grams peak at $N=3$ and get worse from $N=4$; with 1.9M words the peak "shifts rightward" to $N=4$. Both claims were in the report and the slides.
- **What it was**: three bugs in `src/ngram.py`, each of which grows with $N$:
  1. *Equal-weight interpolation threw probability away.* An order whose context never occurred in training contributed 0 but kept its weight, so the probabilities summed to less than 1: 0.667 for an unseen trigram context, 0.333 for an unseen 6-gram context.
  2. *The `<s>` padding was counted as a word.* A 6-gram model pads each sentence with five `<s>` and counted them in the unigram table (in our unit-test corpus, 10 of 18 unigram tokens), so the distribution everything backs off to got worse as $N$ grew.
  3. *"Kneser-Ney" above bigrams was not Kneser-Ney.* For $N \ge 3$ it backed off to a uniform $1/|V|$ instead of the continuation-count distribution.

  On top of that, every training word was in the vocabulary, so `<unk>` never occurred in training and each unknown test word got probability about $10^{-12}$, roughly 7% of the old total test loss.
- **Why it looked like a law**: each bug hurts more as $N$ grows, which is exactly the shape of a "breaking point".
- **How we know the fix is right**: the new Kneser-Ney reproduces, to one decimal at every order, an independent implementation written during the audit (on the old Dataset 1 split: 512.0, 115.1, 77.8, 72.4, 71.6, 72.1). `tests/test_pipeline.py` now checks that probabilities sum to 1 for seen and unseen contexts; that one-line check would have caught bug 1 on the first day.

### Discovery 3: Lookalike Letters Split Ewe Words in Two
Capital eth Ð (U+00D0) looks identical to the Ewe capital Ɖ (U+0189) but lowercases to ð, not ɖ. The corpus used it in 5,026 lines of the previous unified training split, so "ðe" (2,073 times) and "ɖe" (46,110 times) were counted as different words, as were "ðasefowo" and "ɖasefowo". The cleaner now maps Ð and ð to Ɖ and ɖ (and Greek ε to ɛ) before anything is counted.

### Discovery 4: Comparing Tokenizers Needs a Common Unit (a Correction of a Correction)
1. **The draft claim**: "Across all orders, Byte-Pair Encoding (BPE) unlocked the lowest perplexity", comparing BPE's per-token perplexity (13.8) with Unicode Word's (147.8). Per-token perplexities of different tokenizers are not comparable.
2. **The first correction (review of 2026-09-21)**: re-expressed per word, the old results said Unicode Word 320 against BPE 449, so "BPE is worse". That was also wrong. It used the broken models above (where each unknown word cost an arbitrary 27.6 nats), a character model trained on only 4,000 sentences, and BPE merges learned from 3,000.
3. **The corrected comparison**: every tokenizer now trains on the same 98,808 sentences, all lowercase the same way, the character model keeps word boundaries, and each `<unk>` pays the cost of spelling its word with a small character model trained on the words seen once in training. A model that predicts `<unk>` has only said "some rare word"; it has not finished predicting the sentence.

| Tokenizer | Best $N$ (val) | Test PPL per token | Per word, `<unk>` free | Per word, `<unk>` spelled |
|---|:---:|---:|---:|---:|
| Whitespace | 5 | 120.1 | 120.1 | 261.6 |
| Unicode Word | 5 | 69.3 | 132.1 | 202.2 |
| Ewe Stemmer (affixes kept) | 5 | 50.7 | 136.7 | 196.9 |
| BPE (150 merges) | 6 | 9.7 | 188.7 | **189.1** |
| Character | 6 | 3.7 | 446.2 | 447.1 |

Without the spelling charge the ranking runs backwards: the tokenizer with the most unknown tokens (Whitespace, 3.93%) looks best, because `<unk>` is an easy, frequent token. With it, **BPE is best per word**, keeping affixes as tokens beats plain words by 2.6% (196.9 vs 202.2), attached punctuation costs 29% (261.6 vs 202.2), and characters trail because six characters cover only about 1.2 words of context.

The old stemmer *deleted* the affixes it found. That made each prediction easier (fewer, coarser types), so its lower perplexity measured an easier task, not a better model. It now keeps them as tokens (`nu+ srɔ̃la +wo`), so it models the same text as the others. Its rules are still crude: `megbe` (behind) becomes `me+ gbe`.

---

## 4. Problems Encountered & Proven Circumventions

| # | Problem | Symptom | Root cause | Fix (and where) |
|---|---|---|---|---|
| 1 | Combining diacritic splitting | `nusrɔ̃lawo` split into three tokens | `\w` excludes combining marks; ɔ̃ has no precomposed form | token regex accepts U+0300 to U+036F (`src/ewe_tokenizers.py`) |
| 2 | Zero probability under MLE | unseen n-gram gives $P=0$ | MLE has no mass for unseen events | interpolated Kneser-Ney; `perplexity()` now returns infinity for MLE instead of hiding it behind a $10^{-12}$ floor |
| 3 | Probability thrown away at high $N$ | perplexity rose from $N=4$ | equal-weight interpolation kept weight on orders with unseen contexts | renormalize over orders whose context was seen (`src/ngram.py`) |
| 4 | `<s>` counted as a word | unigram mass on `<s>` grew with $N$ | counting started at the padding | count only positions that are predicted |
| 5 | Trigram "Kneser-Ney" was uniform backoff | lower orders never used | wrong lookup table for the lower order | recursive Kneser-Ney with continuation counts, checked against an independent implementation |
| 6 | Unknown words | each got $P \approx 10^{-12}$ | `<unk>` never occurred in training | tokens seen once become `<unk>`; per-word comparisons charge the spelling |
| 7 | Lookalike letters | `ðe` and `ɖe` counted separately | Ð (eth) typed for Ɖ | mapped in `src/data_pipeline.py` |
| 8 | Corrupted rows | 15 binary lines in Dataset 1 | broken rows in the CSV | dropped by the cleaner |
| 9 | Irreproducible splits | nobody else could rebuild the data | build steps were never committed | `scripts/build_ewe_datasets.py` |
| 10 | Section C answers split into fragments | 18 of 100 test "pairs" had no question | answers contain blank lines; files were split on blank lines | JSONL splits (`src/prepare_domain_data.py`) |
| 11 | Section C test questions seen in training | 16 of 100 test pairs were copies of training pairs | 22,615 rows but only 2,212 distinct questions, split without deduplication | one row per question before the split |

(An earlier version of this table listed a Colab memory problem; it never happened in this project and has been removed.)

---

## 5. Multi-Source Dataset Harmonization Protocol & Empirical Sweep

`scripts/build_ewe_datasets.py` builds every split from the raw files; `data/README.md` records their origin.

1. **Dataset 1 (`EWE_ENGLISH.csv`)**: 28,614 English/Ewe rows. Earlier described as cultural folklore; in fact a large share is Jehovah's Witnesses publications and Bible verses (10.6% of kept sentences mention Yehowa, 6.8% carry chapter:verse references). Kept: 25,837 sentences.
2. **Dataset 2 (`eweenglishsentence(3).json`)**: 600 rows from a dictionary database: 477 Glosbe example sentences and 123 sentences from peterlin.pl, including personal introductions that name real people. Kept: 526.
3. **Dataset 3 (`selected transcribed audios.xlsx`)**: 19,152 rows, 19,151 of them with a transcription of a spoken image description (University of Ghana, Waxal project, 2023, 539 speakers). Kept: 19,150.
4. **Dataset 4 (`ewe_corpus.parquet`)**: the first 200,000 rows of a 4,408,322-row English/Ewe pair file sorted by alignment score; at least 10 of 15 randomly sampled kept sentences are Bible or Jehovah's Witnesses text. Kept: 80,183.
5. **Unified**: all four, deduplicated across sources: 123,511 sentences (98,808 train / 12,351 validation / 12,352 test).

### Cross-dataset comparison
Perplexities are measured on each dataset's own test set, so compare *within* a column, not across columns.

| | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 | Unified |
|---|---:|---:|---:|---:|---:|
| Training sentences | 20,669 | 420 | 15,320 | 64,146 | 98,808 |
| Training words | 518,219 | 9,642 | 508,630 | 866,411 | 1,874,130 |
| Unicode Word: test `<unk>` rate | 2.47% | 22.69% | 3.3% | 2.18% | 1.93% |
| Unicode Word: test trigrams unseen in training | 43.59% | 61.94% | 40.37% | 40.8% | 36.91% |
| Unicode Word: $N$ chosen on validation | 5 | 4 | 3 | 4 | 5 |
| Per word, `<unk>` spelled: BPE | **183.2** | **5,806.9** | **187.3** | **201.6** | **189.1** |
| Per word: Ewe Stemmer (affixes kept) | 198.9 | 7,300.0 | 200.9 | 219.0 | 196.9 |
| Per word: Unicode Word | 208.2 | 7,827.7 | 205.1 | 226.0 | 202.2 |
| Per word: Whitespace | 284.5 | 10,555.3 | 237.6 | 295.6 | 261.6 |
| Per word: Character | 375.8 | 8,062.8 | 249.6 | 408.0 | 447.1 |

Dataset 2's per-word numbers are huge because, with only 420 training sentences, 22.69% of Unicode Word test tokens are unknown words that must be spelled out.

### What the corrected sweep shows
1. **No breaking point anywhere.** On every dataset the word-level models flatten from about $N=4$: validation perplexities for $N=4$, 5 and 6 are 3% or less apart. Which of them validation picks (3, 4 or 5) is noise, not a finding.
2. **The tokenizer ranking per word is stable.** BPE is best on all five datasets; keeping affixes as tokens beats plain words on all five; attached punctuation (Whitespace) is worse than Unicode Word on all five; characters are last on four of five (on the 420-sentence Dataset 2 they beat Whitespace).
3. **The smoothing choices hold up on validation.** Kneser-Ney beats equal-weight interpolation at every $N \ge 2$ on every dataset, and the Ney discount estimate is the best of the four discounts we tried on validation, or within 0.3% of it.
4. **The religious skew shows in what the models generate.** The unified 4-gram model's seeded sample is "2 eye yehowa ƒe gbe va na yona , amitai vi ," (the opening of the Book of Jonah). A model for everyday Ewe speech needs more conversational text like Dataset 3.


### Neural baseline (Section B, Question 2)
`scripts/run_lstm_baseline.py` trains an LSTM (`src/lstm_lm.py`) on exactly the BPE tokens of the best n-gram. It asserts that the vocabulary, the number of predicted tokens and the unknown-word spelling charge equal the n-gram run, then scores each test sentence on its own from `<s>`, like the n-gram. Results in `reports/results_lstm_baseline.json`, three seeds each:

| | Dataset 2 (420 training sentences) | Unified (98,808 training sentences) |
|---|---:|---:|
| LSTM size | 479,964 parameters (about 50 per training word) | 3,964,276 parameters (about 2 per training word) |
| LSTM, per word (mean, range) | 7,864.84 (7,396.3 to 8,144.1) | **166.04** (164.88 to 167.54) |
| Kneser-Ney, BPE, per word | **5,806.95** | 189.07 |
| Training time per LSTM seed | under a minute | 2.4 to 3.1 hours |

- **A crossover, not a verdict.** The n-gram wins on micro-data and the LSTM wins at 1.9M words; in both cases every seed falls on the same side of the n-gram number.
- **The unified LSTM is under-trained, so its win is conservative.** Every seed's best epoch was the last of the 10 we could afford, with validation perplexity still falling.
- **A training budget can masquerade as a result.** Our first Dataset 2 run stopped at 20 epochs while all three seeds were still improving and scored 9,243.16 per word; letting early stopping decide (it chose epochs 30 to 34) gave 7,864.84. A cap set for convenience is a hidden hyperparameter.

---

## 6. Synthesis & Viva Exam Readiness (`clenam.ai`)

1. **Why n-grams for a low-resource African language?**  
   *Defense*: We tested it with an LSTM trained on the same BPE tokens and scored per word. On 420 sentences the n-gram wins (5,806.95 against 7,864.84, the LSTM's range 7,396.3 to 8,144.1 over three seeds): about 50 LSTM parameters per training word is more than the data can pin down. On our 1.9M-word corpus the LSTM wins by 12% (166.04, range 164.88 to 167.54, against 189.07), and it was still improving when our 10-epoch budget ran out. The n-gram trains in about 7 minutes where each LSTM took 2.4 to 3.1 hours on a CPU, is fully explainable, and fits speech recognition decoders as a weighted finite-state transducer. So: n-grams for tiny data or tight compute, a neural model once there is enough text and time.
2. **Can you compare the perplexity of your character model with your word model?**  
   *Defense*: Not per token: a character model chooses among 227 symbols per step, a word model among 26,489. Per word it works, as long as every model pays for the whole text; a word model that predicts `<unk>` must also pay to spell the word. On that basis BPE is best (189.1 per word) and characters worst (447.1).
3. **Does a longer context make an n-gram model worse?**  
   *Defense*: Not with correct smoothing. Kneser-Ney passes the probability of an unseen long context down to shorter contexts, so perplexity flattens from $N=4$ even when 80% of test 6-grams are unseen. Our first curve rose after $N=4$ because of three bugs (Discovery 2), and we can name each one.
4. **Did morphological stemming help?**  
   *Defense*: Deleting affixes made the task easier, so its lower perplexity proved nothing. Keeping the affixes as separate tokens gives a small, consistent gain per word on all five datasets (2.6% on the unified corpus). Ewe is usually described as mostly isolating, so a small gain is what we would expect.
5. **What probability does your model give an unknown word?**  
   *Defense*: Words seen only once in training become `<unk>`, so `<unk>` gets a real estimate from those rare-word events. When we compare tokenizers per word we add the cost of spelling the word with a character model.

---

## 7. Section C: Domain-Specific English LLM Adaptation (Agro-Extension)

### 7.1 Problem Motivation
Ankora's agricultural assistants work in English. A pretrained model such as `distilgpt2` writes fluent English but knows little agronomy, and given `Question: ... Answer:` it tends to loop on the prompt. We adapted it with LoRA under two constraints: training must run on a laptop CPU, and the damage to general English should be small, and measured rather than assumed.

### 7.2 Dataset
`KisanVaani/agriculture-qa-english-only` has 22,615 rows but only 2,212 distinct questions. We keep one row per question before shuffling, then split 80/10/10 (seed 42): 1,769 training, 221 validation and 222 test pairs, stored as JSONL (`src/prepare_domain_data.py`). Training uses the first 500 training pairs (a CPU budget). A real test pair:

```text
Question: How does the Botrytis leaf blight pathogen survive during dormant periods?
Answer: The pathogen overwinters as sclerotia, which are produced on infected onion bulbs left in cull piles, on mother bulbs stored for seed production, and on bulbs left in the field. Sclerotia also overwinter directly in the soil and on leaves that persist as debris in commercial onion fields.
```

### 7.3 LoRA Mechanics
LoRA freezes a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and learns $\Delta W = \frac{\alpha}{r} B A$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ and $r \ll \min(d, k)$. In distilgpt2 the target, `c_attn`, is one fused projection per layer that produces the query, key and value vectors: $W_0$ is $768 \times 2304$ (1,769,472 weights). With $r=8$ each layer adds $8 \times (768 + 2304) = 24,576$ weights; over 6 layers that is **147,456 trainable parameters, 0.18% of 82,060,032**. We use $\alpha = 32$ (scale $\alpha / r = 4$), dropout 0.05, and `fan_in_fan_out=True` because GPT-2 stores `c_attn` as a `Conv1D`.

### 7.4 Training
Both adapters train for 3 epochs with AdamW (learning rate $5 \times 10^{-4}$, linear decay), batch size 8, 96-token sequences, seed 42 (`src/train_domain_lora.py`).

| Epoch | Standard adapter: validation loss (full text) | Masked adapter: validation loss (answer tokens only) |
|---|---:|---:|
| 1 | 3.3963 | 3.3395 |
| 2 | 3.3072 | 3.3005 |
| 3 | 3.2874 | 3.2902 |

The two columns measure different tokens, so compare down a column, not across. The mean training loss (3.531 standard, 3.4869 masked) is above the final validation loss because the Trainer averages it over the whole run, early high-loss steps included, and dropout is only active during training. Each adapter took about 20 minutes this time because the CPU was shared with the n-gram sweeps; that is not a statement about LoRA's cost.

### 7.5 Results on 222 Held-Out Questions

| Model | Full Q&A perplexity | Answer-only perplexity | WikiText-2 perplexity |
|---|---:|---:|---:|
| distilgpt2 base | 56.08 | 37.77 | 73.19 |
| LoRA, loss on all tokens | **28.13** | 30.00 | 78.44 |
| LoRA, loss on answers only | 51.39 | **29.09** | 77.24 |

- **Standard adapter**: full-text perplexity down 49.8%, answer perplexity down 20.6%. Most of the extra full-text gain is on the question side (template tokens and question phrasing); the answer-only number is the cleaner measure of domain knowledge.
- **Masked adapter**: answer perplexity down 23.0%, full text only 8.4%, because it never learns to predict questions.
- **Cost**: WikiText-2 perplexity rises 7.2% (standard) and 5.5% (masked). LoRA limits forgetting; it does not prevent it.
- The earlier headline of 62.38 to 29.33 was measured on the leaky test set and is superseded, not comparable.

Seeded samples (temperature 0.7, top-p 0.9, seed 42) for `Question: How can farmers control fall armyworm in maize?`:
- *Base*: "The armyworm has been observed in the north-western part of the country. The main cause of fall armyworm was found in the south-western part of the country. It was found in the"
- *Standard*: "The fall armyworm in maize affects the development of mites, insects and other insects. In addition, the fall armyworm affects the development of mites, insects and other insects. This affects the"
- *Masked*: "The fall armyworm in maize can occur through a variety of diseases, such as arachnoid, cloverworm, and coca. The insect can be introduced to the soil, which is"

The adapters have learned what an extension answer sounds like, not what is true. Of the nine seeded answers in `reports/domain_adaptation_results.json`, at best one (the masked adapter on crop rotation) is roughly right.

---

## 8. Deep Technical Gotchas & Systems Engineering Discoveries

### Gotcha 1: No torch Wheel for Python 3.14 on an Intel Mac
- **Symptom**: `pip install torch` on Python 3.14 on an Intel Mac found no matching distribution.
- **Root Cause**: torch 2.2.2 is the last release with Intel-Mac wheels, and it supports Python only up to 3.12.
- **Resolution**: rebuilt the virtual environment on Python 3.12 and pinned `torch==2.2.2`.

### Gotcha 2: Choosing Library Versions That Work With torch 2.2.2
- **What we did**: pinned `transformers==4.38.2`, `peft==0.10.0`, `accelerate==0.28.0` and `datasets==5.0.1`, a combination that runs every script here on torch 2.2.2. All exact versions are in `requirements.txt`.
- **Correction**: this entry used to say newer transformers releases require torch 2.5 or later. Their PyPI metadata does not say that (4.45 declares no torch floor; 4.50 asks for 2.0 or later), so that explanation was not confirmed.

### Gotcha 3: The NumPy 2 ABI C-Extension Incompatibility
- **Symptom**: torch 2.2.2 with NumPy 2.x failed with `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0`.
- **Resolution**: pinned `numpy==1.26.4` and `scipy==1.12.0`.

### Gotcha 4: Python Module Shadowing of Hugging Face `tokenizers`
- **Symptom**: importing `transformers` failed with `ModuleNotFoundError: No module named 'tokenizers.pre_tokenizers'`.
- **Root Cause**: our file `src/tokenizers.py` was imported instead of the Hugging Face `tokenizers` package.
- **Resolution**: renamed it to `src/ewe_tokenizers.py`.

### Gotcha 5: DistilGPT2 `Conv1D` vs Linear Attention Projections
- **Symptom**: attaching LoRA produced a weight-orientation warning.
- **Root Cause**: GPT-2 stores `c_attn` as a `Conv1D` with transposed weights.
- **Resolution**: `LoraConfig(fan_in_fan_out=True)`.

### Gotcha 6: Blank Lines Inside Answers
- **Symptom**: 18 of the first 100 test "pairs" had no `Question:`.
- **Root Cause**: 790 KisanVaani answers contain blank lines, and the old split files separated pairs with blank lines.
- **Resolution**: one JSON object per pair (`.jsonl`).

---

## 9. Viva Exam Readiness (`clenam.ai`, Section C)

1. **Why LoRA instead of full fine-tuning?**  
   *Defense*: Full fine-tuning updates all 82M parameters and needs optimizer state for each. LoRA trains 147,456 (0.18%) and leaves the pretrained weights untouched, which fits a laptop CPU and limits forgetting. It does not remove it: general-English perplexity still rose 7.2%.
2. **How do you know the model learned the domain rather than the format?**  
   *Defense*: We score the answer tokens separately. Answer perplexity fell from 37.77 to 30.00 (20.6%) on 222 test questions that never appear in training; the larger full-text drop (49.8%) also includes the template and the questions.
3. **Why did the base model repeat the question?**  
   *Defense*: A base causal LM has never been trained on this question-answer format, and once text starts repeating, more repetition becomes the likeliest continuation. After adaptation the model answers in the right shape, but often with wrong content.
4. **How did you prevent leakage?**  
   *Defense*: The raw data repeats each question about ten times. We keep one row per question before splitting, and the training script checks that none of the 222 test questions appear in training or validation (it finds 0). Our first split did not do this, and 16 of its 100 test pairs were copies of training pairs.

---

## 10. Decoding Strategies & Repetition

### 10.1 The phenomenon
Sampling from a small model sometimes loops. From the benchmark: "It helps to control soil erosion, which can lead to soil erosion. It also helps to improve soil drainage. It also helps to reduce soil erosion." Once a phrase is in the context, repeating it becomes more likely; this self-reinforcing repetition is well documented for neural text generation (Holtzman et al., 2020; Xu et al., 2022).

### 10.2 Benchmark (`reports/decoding_strategies_benchmark.json`)
Four prompts; each sampled strategy is run with 5 fixed seeds per prompt, greedy search once. Distinct-3 is unique word trigrams divided by all word trigrams in an answer.

| Strategy | Settings | Mean Distinct-3 |
|---|---|---:|
| Unpenalized sampling | $T=0.7$, top-p 0.9 | 78.9% |
| Repetition penalty | $T=0.7$, top-p 0.9, penalty 1.3 | 100.0% |
| 3-gram blocking | $T=0.7$, top-p 0.9, `no_repeat_ngram_size=3` | 99.5% |
| Low temperature + penalty + block | $T=0.35$, top-p 0.85, penalty 1.25, block 3 | 100.0% |
| Greedy + penalty + block | penalty 1.25, block 3 | 100.0% |

- 3-gram blocking forbids the decoder to repeat any token trigram, so a Distinct-3 near 100% is guaranteed by construction; it shows the setting works, not that the answers are good.
- A penalty buys variety at the cost of topic: one penalized answer about fall armyworm begins "The common disease, the malaria parasite-borne chikungunya virus, is transmitted by soil moisture."
- An earlier version of this benchmark reported 49.1% for unpenalized sampling from one unseeded sample per prompt, dominated by a single answer that repeated `Answer:`; with 5 seeds per prompt it is 78.9%. It also called one setting "optimal" with "zero hallucination"; nothing measured supports that, and no setting here produces reliable advice.

---

## 11. Prompt-Loss Masking vs Standard Causal LM

### 11.1 The idea
With the standard objective the loss covers the question and the answer. Masking sets the question's labels to `-100`, so `CrossEntropyLoss` ignores them and training optimizes only $P(\text{answer} \mid \text{question})$.

### 11.2 A fair comparison
Both adapters: same 500 training pairs, same seed, same hyperparameters, scored on the same 222 test questions.

| Model | Answer-only perplexity | Full Q&A perplexity | WikiText-2 perplexity |
|---|---:|---:|---:|
| Base | 37.77 | 56.08 | 73.19 |
| Standard | 30.00 | **28.13** | 78.44 |
| Masked | **29.09** | 51.39 | 77.24 |

- On answers the two are comparable: masking is 3% better, from a single seed and 500 pairs.
- Masking gives up almost all of the full-text gain, because the model never learns to predict questions.
- **Which to use depends on the job.** For a question-answering assistant, masking is the usual choice. For Ankora's speech recognition setting, where the model scores whole transcripts including farmers' spoken questions, the standard objective is the one that learns them.
- An earlier entry declared masking "the superior training paradigm" after comparing the masked adapter only with the base model, trained on 400 pairs instead of 500. On the repo's own answer metric the standard adapter from the same day already scored as well (29.92 vs 30.08), so that conclusion was not supported.

---

## 12. Verification Audit (2026-09-21/22): What We Believed, What We Found, Why

A review of the whole repository against its code and data, followed by the fixes above. Recorded here because the mistakes are the most useful thing this prosit taught.

**Numbers no code produced.** Several report and journal numbers were written before or without the experiments they describe: the tables first drafted in section 2 (and the same values typed into notebook 03 and quoted in the Section B report), a general-English "forgetting" check reported as under 3.8% (never run; the measured cost is 7.2%), an ARPA-export capability, an example Q&A pair that is not in the corpus, a *Sitophilus zeamais* analysis for a term the corpus never contains, and slide figures from before the real data existed. All removed or replaced by measured values.

**Bugs that produced the headline findings.** The "breaking point" and its "rightward shift" came from three smoothing bugs plus the unknown-word floor (Discovery 2). The tokenizer ranking came from comparing per-token perplexities, then from a per-word comparison built on the broken models (Discovery 4). The "stemming advantage" came from deleting affixes.

**Data that was not what we said.** Dataset 1 was described as folklore and Section B said we had avoided relying on religious text; a large share of Datasets 1 and 4 is Bible and Jehovah's Witnesses material. Dataset 2 contains personal information. Lookalike letters split words (Discovery 3). Section C's test set shared 16 of 100 pairs with training.

**Why it happened.** Text drafted with AI assistance was written ahead of the experiments and never reconciled with the result files. The tone ("Grand Unified Mega-Corpus", "undeniably superior", "ironclad") made the claims harder to question, and there was no test that a probability distribution sums to 1.

**What changed.** Every Ewe split and every result is rebuilt by committed scripts; unit tests check that the smoothed distributions sum to 1; the order $N$ and the discount are chosen on validation; tokenizers are compared per word with unknown words charged; Section C is deduplicated before splitting and reports answer-only and general-English perplexity. `METHODOLOGY_GUIDE.md` gained a sixth pillar, *Verify Before You Write*.

**What the corrected numbers say.** With correct smoothing, longer contexts stop helping from about $N=4$ but never hurt; BPE is the best tokenizer per word on every dataset; an LSTM on the same tokens loses to the n-gram on 420 sentences but beats it by 12% on the full corpus, at hours of CPU instead of minutes; keeping Ewe affixes as tokens helps a little; LoRA adapts distilgpt2's probabilities to agricultural answers (20.6% lower answer perplexity) at a measurable cost to general English, and its answers are fluent but not reliable.
