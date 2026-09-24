# Team Learning Journal: Prosit 1

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Team**: MICS 2028 · Group 1  
**Project**: Prosit 1 (Ankora AI Research Lab: Language Modeling & Domain Adaptation)  
**Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  

> The group's shared record of what we built, what we found, and what we got wrong and fixed. Every number is copied from a file in `results/` (or a `stats.json` under `data/processed/`) written by a committed script; `reports/claims_table.md` says which. Individual reflections are not part of this file.

---

## 1. What We Built: Three Separate Models

| Model | Report | Language and data | What is trained | Code | Results |
|---|---|---|---|---|---|
| **N-gram** (interpolated Kneser-Ney) | Section B | Ewe, 123,511 sentences | nothing is trained: counts plus smoothing | `src/section_b_ngram/` | `results/section_b_ngram/` |
| **LSTM baseline** | Section B, Question 2 | the same Ewe sentences and BPE tokens as the n-gram | a small network, from scratch | `src/section_b_lstm/` | `results/section_b_lstm/` |
| **distilgpt2 + LoRA** | Section C | English agricultural Q&A, 2,212 questions | 147,456 of 82,060,032 parameters (0.18%) | `src/section_c_llm/` | `results/section_c_llm/` |

The LSTM exists only to answer Section B's n-gram-versus-neural question, on Ewe. It is not the LLM of Section C: the two share no code, data or weights. The n-gram and the LSTM are scored per word on the same Ewe test sentences, so their numbers compare; the LLM is scored on English with its own tokenizer, so its perplexities do not compare with Section B's.

---

## 2. Section B: The Ewe N-Gram Models

**Data.** Four existing sources, cleaned, deduplicated within and across sources, then split 80/10/10 with seed 42: 123,511 sentences, of which 98,808 are training sentences (1,874,130 words). A large part is Bible and Jehovah's Witnesses text (5.1% of sentences mention Yehowa, 6.2% carry chapter:verse references), so the models favour that register. Details: `data/README.md` and `reports/datasheet.md`.

**Setup.** Interpolated Kneser-Ney; tokens seen once in training become `<unk>`; the order $N$ is chosen on 4,000 validation sentences and test perplexity is reported once, on 4,000 test sentences.

### 2.1 Context length (Unicode Word tokens, unified corpus)

| Order $N$ | Test n-grams never seen in training | Kneser-Ney, validation | Kneser-Ney, test | Equal-weight interpolation, test |
|---|---:|---:|---:|---:|
| 1 | 0.0% | 527.7 | 534.7 | 534.7 |
| 2 | 10.5% | 117.6 | 121.2 | 151.6 |
| 3 | 36.9% | 74.9 | 77.7 | 93.6 |
| 4 | 60.9% | 68.0 | 70.5 | 80.6 |
| 5 | 74.5% | 66.8 | **69.3** | 77.0 |
| 6 | 80.5% | 67.1 | 69.7 | 75.9 |

Longer context never hurts once smoothing is correct: perplexity falls to 70.5 by $N=4$ and then stays flat, even though 80.5% of test 6-grams were never seen, because Kneser-Ney passes the probability of an unseen long context down to shorter ones. Validation picks $N=5$, but $N=4$, 5 and 6 are within noise of each other. Kneser-Ney beats equal-weight interpolation at every $N \ge 2$.

### 2.2 Tokenizers (unified corpus, best $N$ per tokenizer)

| Tokenizer | Vocabulary | Test tokens that are `<unk>` | Tokens per word | Best $N$ | Perplexity per token | Per word, `<unk>` free | Per word, `<unk>` spelled |
|---|---:|---:|---:|:---:|---:|---:|---:|
| Whitespace | 37,560 | 3.93% | 1.00 | 5 | 120.1 | 120.1 | 261.6 |
| Unicode Word | 26,489 | 1.93% | 1.16 | 5 | 69.3 | 132.1 | 202.2 |
| Ewe Stemmer (affixes kept) | 23,297 | 1.55% | 1.27 | 5 | 50.7 | 136.7 | 196.9 |
| BPE (150 merges) | 371 | 0.01% | 2.38 | 6 | 9.7 | 188.7 | **189.1** |
| Character | 227 | 0.00% | 4.82 | 6 | 3.7 | 446.2 | 447.1 |

- **Per-token perplexity cannot rank tokenizers.** A character model chooses among 227 symbols per step, a word model among 26,489. Per word, every model is scored on the same text.
- **A word model that predicts `<unk>` has not finished the job.** Without charging each `<unk>` for spelling its word (with a small character model trained on the words seen once), the ranking runs backwards: the tokenizer with the most unknown tokens looks best. With the charge, BPE is best, keeping Ewe affixes as tokens beats plain words by 2.6%, attached punctuation costs 29%, and characters trail.

### 2.3 The same comparison on each source dataset

Perplexities are measured on each dataset's own test set, so compare within a column, not across columns.

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

- **No breaking point anywhere.** On every dataset, validation perplexities for $N=4$, 5 and 6 are 3% or less apart.
- **The ranking is stable.** BPE is best on all five; affixes as tokens beat plain words on all five; characters are last on four of five (on the 420-sentence Dataset 2 they beat Whitespace).
- **The smoothing choices hold on validation.** The Ney discount estimate is the best of the four discounts we tried, or within 0.3% of it.
- **The religious skew shows in the output.** The unified 4-gram's seeded sample is "2 eye yehowa ƒe gbe va na yona , amitai vi ," (the opening of the Book of Jonah).

---

## 3. Section B, Question 2: The LSTM Baseline

The LSTM reads exactly the BPE tokens of the best n-gram; the script asserts that the vocabulary, the number of predicted tokens and the `<unk>` spelling charge match the n-gram run. Each test sentence is scored on its own from `<s>`, like the n-gram. Three seeds per dataset.

| | Dataset 2 (420 training sentences) | Unified (98,808 training sentences) |
|---|---:|---:|
| LSTM size | 479,964 parameters (about 50 per training word) | 3,964,276 parameters (about 2 per training word) |
| LSTM, per word (mean, range over seeds) | 7,864.84 (7,396.3 to 8,144.1) | **166.04** (164.88 to 167.54) |
| Kneser-Ney, BPE, per word | **5,806.95** | 189.07 |
| Training time per LSTM seed | under a minute | 2.4 to 3.1 hours |

- **A crossover, not a verdict.** The n-gram wins on micro-data and the LSTM wins by 12% at 1.9M words; in both cases every seed falls on the same side of the n-gram.
- **The unified LSTM is under-trained, so its win is conservative.** Every seed's best epoch was the last of the 10 we could afford.
- **A training cap is a hidden hyperparameter.** Our first Dataset 2 run stopped at 20 epochs while still improving and scored 9,243.16 per word; letting early stopping decide gave 7,864.84.

---

## 4. Section C: distilgpt2 Adapted with LoRA

**Data.** `KisanVaani/agriculture-qa-english-only` has 22,615 rows but only 2,212 distinct questions. We keep one row per question before shuffling, then split 1,769 / 221 / 222; the training script confirms that 0 test questions appear in training or validation. Training uses the first 500 training pairs (a CPU budget).

**Setup.** LoRA on `c_attn`, GPT-2's fused query/key/value projection: rank 8, $\alpha=32$, dropout 0.05, `fan_in_fan_out=True` because GPT-2 stores the layer as a `Conv1D`. Two adapters on identical data: *standard* (loss on every token) and *masked* (loss on the answer tokens only).

| Model | Full Q&A perplexity | Answer-only perplexity | WikiText-2 perplexity |
|---|---:|---:|---:|
| distilgpt2 base | 56.08 | 37.77 | 73.19 |
| LoRA, loss on all tokens (standard) | **28.13** | 30.00 | 78.44 |
| LoRA, loss on answers only (masked) | 51.39 | **29.09** | 77.24 |

- **The answer-only number measures domain knowledge.** The standard adapter cuts it by 20.6%; its larger full-text drop (49.8%) also includes learning the question format.
- **Adaptation costs general English.** WikiText-2 perplexity rises 7.2% (standard) and 5.5% (masked). LoRA limits forgetting; it does not prevent it.
- **Masking is comparable on answers** (one seed, 500 pairs) but never learns to predict questions. For Ankora's speech recognition use, where the model scores whole transcripts including farmers' questions, the standard objective is the one that learns them.
- **Fluent is not correct.** Asked how to control fall armyworm in maize, the standard adapter answers "The fall armyworm in maize affects the development of mites, insects and other insects." Of the nine seeded answers, at best one is roughly right. The model can help score transcripts in the domain; it must not give farmers advice.
- **Decoding settings change repetition, not correctness.** Mean Distinct-3 over 4 prompts (5 seeds per sampled strategy): unpenalized sampling 78.9%, repetition penalty 1.3 100.0%, 3-gram blocking 99.5% (guaranteed by construction), low temperature with penalty and blocking 100.0%, greedy with penalty and blocking 100.0%.

---

## 5. What Went Wrong, and the Fixes

| # | Problem | Symptom | Root cause | Fix (and where) |
|---|---|---|---|---|
| 1 | Combining diacritic splitting | `nusrɔ̃lawo` split into three tokens | `\w` excludes combining marks; ɔ̃ has no precomposed form | token regex accepts U+0300 to U+036F (`src/section_b_ngram/ewe_tokenizers.py`) |
| 2 | Zero probability under MLE | unseen n-gram gives $P=0$ | MLE has no mass for unseen events | interpolated Kneser-Ney; `perplexity()` returns infinity for MLE instead of hiding it behind a $10^{-12}$ floor |
| 3 | Probability thrown away at high $N$ | perplexity rose from $N=4$ | equal-weight interpolation kept weight on orders with unseen contexts | renormalize over orders whose context was seen (`src/section_b_ngram/ngram.py`) |
| 4 | `<s>` counted as a word | unigram mass on `<s>` grew with $N$ | counting started at the padding | count only positions that are predicted |
| 5 | Trigram "Kneser-Ney" was uniform backoff | lower orders never used | wrong lookup table for the lower order | recursive Kneser-Ney with continuation counts, checked against an independent implementation |
| 6 | Unknown words | each got $P \approx 10^{-12}$ | `<unk>` never occurred in training | tokens seen once become `<unk>`; per-word comparisons charge the spelling |
| 7 | Lookalike letters | `ðe` and `ɖe` counted separately | Ð (eth) typed for Ɖ | mapped in `src/section_b_ngram/data_pipeline.py` |
| 8 | Corrupted rows | 15 binary lines in Dataset 1 | broken rows in the CSV | dropped by the cleaner |
| 9 | Irreproducible splits | nobody else could rebuild the data | build steps were never committed | `src/section_b_ngram/build_datasets.py` |
| 10 | Section C answers split into fragments | 18 of 100 test "pairs" had no question | answers contain blank lines; files were split on blank lines | JSONL splits (`src/section_c_llm/prepare_data.py`) |
| 11 | Section C test questions seen in training | 16 of 100 test pairs were copies of training pairs | 22,615 rows but only 2,212 distinct questions, split without deduplication | one row per question before the split |

Rows 3 to 6 together produced our first draft's "breaking point" after $N=4$: each bug hurts more as $N$ grows, so together they looked like a property of the language. The corrected Kneser-Ney matches an independent implementation to one decimal at every order.

**Environment traps.** torch 2.2.2 is the last release with Intel-Mac wheels and supports Python only up to 3.12; it also needs NumPy 1.x (`numpy==1.26.4`). A file named `src/tokenizers.py` shadowed the Hugging Face `tokenizers` package, which is why ours is `ewe_tokenizers.py`. Exact versions are in `requirements.txt`.

---

## 6. Lessons

1. **Test that probabilities sum to 1.** That one check, now in `tests/test_pipeline.py`, would have caught the interpolation bug on the first day.
2. **Compare tokenizers per word, and charge `<unk>` for spelling the word.** Per-token perplexity and free `<unk>` both rank tokenizers wrongly.
3. **Deduplicate before splitting.** Otherwise the test set leaks into training.
4. **Choose settings on validation; touch test once.** $N$, the discount and the LSTM's best epoch were all chosen on validation.
5. **Never write a number that no committed script produced.** `reports/claims_table.md` ties every quoted number to its file, key and script.
6. **Budgets are hyperparameters.** The 20-epoch cap we first used left the Dataset 2 LSTM at 9,243.16 per word instead of 7,864.84.

---

## 7. Open Items

- The licences of the four Ewe sources are unverified, and the origins of Datasets 1 and 4 were not recorded: do not redistribute the Ewe data or models trained on it.
- No Ewe speaker has judged the generated samples.
- Section C rests on one seed and 500 training pairs; the standard and masked adapters are within noise of each other on answers.
