# Comprehensive Quiz Revision Self-Check & Viva Defense Guide

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Target Assessment**: Automated AI Viva Quiz via `clenam.ai` & Faculty Oral Defense (35% of Total Grade)  
**Primary References**: 
- Jurafsky & Martin (2026), *Speech and Language Processing (3rd ed.)*, Chapter 3.
- Bender, Gebru, Mitchell, & McMillan-Major (2021), *On the Dangers of Stochastic Parrots*.

---

### 1. Sparsity Constraints & Subword Tokenization
- **The Problem**: In strict word-level statistical language models, vocabulary elements are treated as atomic, orthogonal symbols in a discrete set. The model treats morphological variants like `"walk"`, `"walks"`, and `"walked"` as completely independent, unrelated tokens. If `"walked"` does not appear in the training corpus, it is mapped to `<unk>` and its morphological relation to `"walk"` is lost. This leads to massive vocabulary explosion and acute data sparsity.
- **The Solution (Subword Tokenization)**: Algorithms like Byte-Pair Encoding (BPE), WordPiece, and SentencePiece break words down into frequent morphemes and character n-grams (e.g., `["walk", "##ed"]`). Unseen words are decomposed into known constituent subwords, virtually eliminating `<unk>` occurrences while allowing the model to share learned syntactic representations across related words.

---

### 2. Mathematical Derivation of Bigram Maximum Likelihood Estimation (MLE)
**Question**: Derive $P(\text{gave} \mid \text{friend}) = \frac{C(\text{friend gave})}{C(\text{friend})}$ from first principles.

1. **Definition of Conditional Probability**:
   $$P(w_n \mid w_{n-1}) = \frac{P(w_{n-1}, w_n)}{P(w_{n-1})}$$
2. **Estimating Probabilities via Relative Frequencies**:
   Let $M$ be the total number of words (or tokens) in the training corpus.
   - The joint probability $P(w_{n-1}, w_n)$ is estimated as the count of the bigram $C(w_{n-1}, w_n)$ divided by the total bigrams in the corpus ($M$):
     $$P(w_{n-1}, w_n) = \frac{C(w_{n-1}, w_n)}{M}$$
   - The marginal probability $P(w_{n-1})$ is estimated as the count of the unigram $C(w_{n-1})$ divided by $M$:
     $$P(w_{n-1}) = \frac{C(w_{n-1})}{M}$$
3. **Substituting into Conditional Probability**:
   $$P(w_n \mid w_{n-1}) = \frac{\frac{C(w_{n-1}, w_n)}{M}}{\frac{C(w_{n-1})}{M}} = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$$
4. **Conclusion**:
   For the specific bigram `"friend gave"`:
   $$P(\text{gave} \mid \text{friend}) = \frac{C(\text{friend gave})}{C(\text{friend})}$$

---

### 3. Evaluation Constraints: Why Counting Whole Sentences Fails
- **The Question**: Why can't we evaluate a language model by counting whole sentences?
- **The Mathematical Reality**: Natural language has infinite combinatoric variety (Chomsky's linguistic productivity). A sentence of average length (15–20 words) has an astronomically small probability of ever being repeated verbatim in any standard-sized evaluation corpus, unless it is a fixed proverb or title.
- **Consequence**: If we attempted to evaluate $P(S)$ by computing $\frac{C(S)}{N_{\text{sentences}}}$, almost every valid grammatical sentence in the test set would have count $C(S) = 0$. This would assign $P(S) = 0$, falsely claiming the sentence is impossible.
- **The Solution**: We apply the **chain rule** and the **Markov assumption** to decompose sentence probabilities into the product of local token conditional probabilities:
  $$P(W) = \prod_{i=1}^n P(w_i \mid w_{i-N+1}^{i-1})$$
  and evaluate via **Perplexity (PP)**.

---

### 4. Hyperparameter Identification
- **Definition**: A **hyperparameter** is a configuration variable external to the model whose value is set before the learning process begins and cannot be directly estimated or updated from the training data via standard optimization.
- **N-Gram Statistical Model Example**: The choice of n-gram order **$N$** (e.g., deciding whether to train a unigram $N=1$, bigram $N=2$, or trigram $N=3$), or the smoothing parameter **$k$** in Lidstone smoothing.
- **Neural Network Example**: The **learning rate ($\eta$)** in AdamW gradient descent, the LoRA rank **$r$**, or the batch size.

---

### 5. The Lookback Anomaly: Why Bigram Produces "I live TV" While Trigram Avoids It
- **The Scenario**: Consider the generation of the ungrammatical sequence: *"I live TV"*.
- **Bigram Lookback Limit ($N=2$, history $N-1=1$)**:
  - Probability decomposition: $P(\text{I} \mid \text{<s>}) \times P(\text{live} \mid \text{I}) \times P(\text{TV} \mid \text{live}) \times P(\text{</s>} \mid \text{TV})$.
  - In a training corpus, *"I live"* appears frequently (e.g. *"I live in Accra"*).
  - Additionally, *"live TV"* appears frequently (e.g. *"watch live TV"*).
  - Because the bigram model evaluates each word conditioned **only on the single immediately preceding word**, it evaluates $P(\text{TV} \mid \text{live})$ in isolation! Because *"live TV"* is common, $P(\text{TV} \mid \text{live})$ is high. The bigram model has forgotten that *"I"* came before *"live"*, resulting in the nonsensical sentence *"I live TV"*.
- **Trigram Solution ($N=3$, history $N-1=2$)**:
  - The trigram model evaluates: $P(\text{TV} \mid \text{I live})$.
  - In the corpus, the trigram `"I live TV"` never occurs ($C(\text{I live TV}) = 0$).
  - Therefore, $P(\text{TV} \mid \text{I live}) \approx 0$, preventing the model from making this grammatical error.

---

### 6. Structural Tokens: Role of Start Token `<s>`
- **The Mathematical Role**: To represent a valid probability distribution over sequences of variable lengths, the language model must compute the probability of the sentence beginning with a specific word $w_1$. Without a start token, a bigram model would have no preceding conditioning context for $w_1$.
- **Formula**: Prepending the start boundary token `<s>` allows the model to treat the initial word as a standard conditional bigram:
  $$P(w_1 \mid \text{<s>}) = \frac{C(\text{<s>}, w_1)}{C(\text{<s>})}$$
  Similarly, the end token `</s>` represents the event of sentence termination, ensuring the sum of probabilities over all possible sentence lengths equals 1.0.
