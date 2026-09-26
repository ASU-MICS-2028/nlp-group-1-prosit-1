# Technical Report, Section B: Specialized Language Model for a Low-Resource African Language (Ewe / Èʋegbe)

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Team**: MICS 2028 · Group 1  
**Deliverable**: Technical Report Section B (Group Sync, identical across team members) · Weight: 15% Implementation & Results + 5% Writing Quality = 20%  
**Public Repository**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  
**Numbers**: every figure below is copied from `results/section_b_ngram/unified.json` (or the per-dataset result files) and `data/processed/unified/stats.json`; see `reports/claims_table.md`.

---

### Question 1: What data did you use in building your model?
*(Space Guide: 1 Paragraph)*

Our group chose **Ewe (Èʋegbe)**, a Gbe language of the Kwa branch of Niger-Congo spoken in south-eastern Ghana (Volta Region), southern Togo and Benin. We combined four existing sources (`data/README.md`): (1) English/Ewe sentence pairs (`EWE_ENGLISH.csv`), a large share of which are Jehovah's Witnesses publications and Bible verses; (2) 526 dictionary example sentences and short personal texts from a database export (Glosbe and peterlin.pl); (3) 19,150 transcriptions of spoken image descriptions from the University of Ghana's Waxal project, our only conversational speech; and (4) the first 200,000 rows of a large English/Ewe sentence-pair file, mostly Bible-aligned text. Religious text is therefore a large part of the corpus (5.1% of unified sentences mention Yehowa and 6.2% carry chapter:verse references), which skews the model toward that register. All text is Unicode NFC normalized so that Ewe letters (`ɖ`, `ƒ`, `ɣ`, `ŋ`, `ɔ`, `ɛ`, `ʋ`) and tone marks are stored one way; lookalike letters typed in their place (capital eth Ð for Ɖ) are mapped back; corrupted rows are dropped; and duplicates are removed within and across sources before splitting. That leaves **123,511 sentences**, split 80/10/10 into 98,808 training (1,874,130 words), 12,351 validation and 12,352 test sentences (`src/section_b_ngram/build_datasets.py`). The vocabulary comes from the training split only, and words seen just once in training become `<unk>`.

---

### Question 2: Do you agree that n-gram models are better than neural models when building a language model for a low-resource language?
*(Space Guide: 1–2 Paragraphs)*

Partly, and we tested it. We trained an LSTM language model (`src/section_b_lstm/lstm_lm.py`) on exactly the tokens our best n-gram uses, BPE subwords, and compared test perplexity per word over three seeds (`src/section_b_lstm/run_baseline.py` asserts that the vocabulary, token count and unknown-word charge match the n-gram run). **With very little data the n-gram wins.** On Dataset 2's 420 training sentences (9,642 words), interpolated Kneser-Ney scores 5,806.95 per word against 7,864.84 for the LSTM (7,396.3 to 8,144.1 across seeds), 26% lower. Even this small LSTM has 479,964 parameters, about 50 per training word, far more than the evidence can pin down, while the n-gram's estimates come straight from counts. That, together with training in minutes on a CPU, full explainability and a direct route into speech recognition decoders as weighted finite-state transducers, is why n-grams are the right first model for a truly low-resource language.

**With our full corpus the LSTM wins.** On the unified corpus (98,808 training sentences, 1.9M words), a larger LSTM (3,964,276 parameters, about 2 per training word) scores 166.04 per word (164.88 to 167.54 across seeds) against 189.07 for the best n-gram, 12% lower. Every seed was still improving when it reached our 10-epoch budget, so this understates it. The LSTM conditions on the whole sentence so far rather than the last five tokens and, given enough data, learns representations shared by similar tokens; we did not separate the two effects. The price is compute: each LSTM run took 2.4 to 3.1 hours on a CPU, against about 7 minutes for all six n-gram orders. So we agree for very small corpora and for tight compute and latency budgets, but not in general: somewhere between a few hundred and a hundred thousand sentences, the neural model overtakes the n-gram. The n-gram's order $N$ and discount were chosen on validation and the LSTM's size was set by the amount of data; neither model saw the test set before it was scored.

---

### Question 3: How did you train your model and what convinced you your model was learning?
*(Space Guide: 2–3 Paragraphs)*

`src/section_b_ngram/ngram.py` counts, for every order $k \le N$, how often each word follows each $(k-1)$-word context, over padded sentences (`<s>` before, `</s>` after), counting only the positions the model actually has to predict. On those counts we implemented Maximum Likelihood Estimation, Laplace and Lidstone (add-$k$) smoothing, linear interpolation (with the weights of orders whose context was never seen redistributed to the others, so probabilities still sum to 1), and **interpolated Kneser-Ney**: each order's counts are discounted by $D_k$, estimated from the training counts as $D = n_1 / (n_1 + 2 n_2)$, and the freed probability goes to the next-lower order, which uses continuation counts (in how many different contexts a word has appeared). The main sweep (`src/section_b_ngram/run_sweep.py`) trains Kneser-Ney models for five tokenizers and $N = 1 \dots 6$, and picks each tokenizer's best $N$ on the validation split.

Three things convinced us the models were learning:
1. **Held-out perplexity falls as context grows, then levels off.** For Unicode Word tokens, test perplexity goes 534.7 (unigram), 121.2 (bigram), 77.7 (trigram), 70.5 (4-gram), then stays flat (69.3 at $N=5$, 69.7 at $N=6$).
2. **Every smoothed model is a proper probability distribution.** Unit tests (`tests/test_pipeline.py`) check that probabilities sum to 1 over the vocabulary for seen and unseen contexts. An earlier version of our interpolation failed this (the probabilities for an unseen trigram context summed to 0.667), and finding that failure is how we traced our first, wrong conclusion that longer contexts "break" the model (Question 6).
3. **Samples become longer runs of real Ewe, and eventually copies of training text.** With a fixed seed, the unified corpus model moves from punctuation and function words at $N=1$ to, from $N=4$ on, reproducing a Bible verse: "2 eye yehowa ƒe gbe va na yona , amitai vi ," (Jonah 1:1). That shows both memorization and the corpus's religious skew. Whether the shorter samples are grammatical has to be judged by an Ewe speaker, so we record them rather than grade them.

---

### Question 4: How did you evaluate your model?
*(Space Guide: 1–2 Paragraphs)*

We measured **perplexity** on held-out sentences, the exponentiated average negative log-probability per predicted token:
$$\text{PP}(W) = \exp\left(-\frac{1}{M} \sum_{i=1}^{M} \ln P(w_i \mid w_{i-N+1}^{i-1})\right)$$
where $M$ counts every predicted token including `</s>`. The order $N$ and the Kneser-Ney discount were chosen on 4,000 validation sentences; test perplexity on 4,000 test sentences is reported once. Vocabularies come from the training split only.

Per-token perplexity cannot compare tokenizers, because a character model picks among 227 symbols per step and a word model among 26,489. For that comparison we use **perplexity per word**: the same total test log-probability divided by the number of words, which is identical for every tokenizer because they all model the same text. To be fair, every model must pay for the whole text: a word model that predicts `<unk>` has not said which word it was, so each `<unk>` is also charged the cost of spelling the word with a small character model trained on the words seen once in training. We also report **sparsity**, the share of test n-grams never seen in training.

---

### Question 5: What results did you get?
*(Space Guide: 1–2 Paragraphs)*

Best order per tokenizer on the unified corpus (chosen on validation):

| Tokenizer | Best $N$ | Vocabulary | Test perplexity per token | Test perplexity per word |
| --- | :---: | ---: | ---: | ---: |
| Whitespace (punctuation attached) | 5 | 37,560 | 120.1 | 261.6 |
| Unicode Word | 5 | 26,489 | 69.3 | 202.2 |
| Ewe Stemmer (affixes kept as tokens) | 5 | 23,297 | 50.7 | 196.9 |
| Byte-Pair Encoding (150 merges) | 6 | 371 | 9.7 | **189.1** |
| Character | 6 | 227 | 3.7 | 447.1 |

**Longer context never hurts, but stops helping from about $N=4$.** Word-level test perplexity levels off at $N=4$ even though 80.5% of test 6-grams were never seen in training, because Kneser-Ney passes the probability of an unseen long context down to shorter ones; on every dataset, validation perplexities for $N = 4$, 5 and 6 are 3% or less apart. **Per word, subwords win**: BPE is best (189.1), keeping Ewe affixes as separate tokens beats plain words by 2.6%, attaching punctuation to words costs 29%, and a character 6-gram, which sees only about 1.2 words of context, is far behind. The same ranking holds on each of the four source datasets, except that on the 420-sentence Dataset 2 characters beat whitespace tokens. Kneser-Ney beats equal-weight interpolation at every $N \ge 2$ (77.7 against 93.6 at $N=3$ for Unicode Word).

---

### Question 6: What should we know about the work you did which is not already captured in your answers above?
*(Space Guide: 1–3 Paragraphs)*

**Our first conclusions were wrong, and finding out why taught us the most.** An earlier draft of this report said word n-grams "break" beyond $N=3$, that more data "shifts" the breaking point to $N=4$, and that BPE reached a perplexity of 13.8 against 147.8 for words. All three came from errors: our interpolation threw away the probability of unseen contexts (more of it as $N$ grew), the `<s>` padding was counted as a word, trigram "Kneser-Ney" backed off to a uniform distribution, unknown words got an arbitrary probability of about $10^{-12}$, and per-token perplexities of different tokenizers were compared directly. We found these by checking that probabilities sum to 1 and by re-implementing Kneser-Ney independently; the corrected code matches that implementation exactly, and the unit tests now guard it.

**Ewe orthography needs care at the character level.** Nasalized vowels such as ɔ̃ have no precomposed Unicode character, so they remain a letter plus a combining tilde even after NFC normalization, and Python's `\w` does not match the tilde; our tokenizers accept the combining-mark range explicitly. We also found that 5,026 lines of an earlier training split used capital eth Ð, which looks identical to Ewe Ɖ but lowercases to ð instead of ɖ, splitting words such as "ɖe" into two vocabulary entries; the cleaner now maps these lookalikes back.

**The data has limits we should state plainly.** A large share of it is religious text, so the model will favour that register over everyday speech; one source contains personal introductions naming real people; and the licences of the sources are unverified, so neither the data nor the models should be redistributed. All splits, results and figures can be rebuilt from the raw files with the commands in `README.md`.
