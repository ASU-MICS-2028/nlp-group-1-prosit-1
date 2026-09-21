# %% [markdown]
"""
# 01_low_resource_ngram_lm
"""

# %% [markdown]
"""
# 01. Specialized Language Model for Low-Resource African Language (Ewe / Èʋegbe)

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Project**: Prosit 1 (Ankora AI Research Lab)  
**Target Language**: **Ewe (Èʋegbe)** — spoken in Ghana, Togo, and Benin  
**Objective**: Develop and evaluate a statistical n-gram language model for Ewe, addressing data scarcity, unique orthographic characters (`ɖ`, `ƒ`, `ɣ`, `ŋ`, `ɔ`, `ɛ`, `ʋ`), out-of-vocabulary words (`<unk>`), and smoothing techniques.

---
### Pipeline Overview
1. **Data Ingestion**: Load Ewe text from a local file (`data/raw/low_resource/`) or Hugging Face dataset.
2. **Ewe Unicode Tokenization**: NFC normalization to preserve tone diacritics and distinct Ewe alphabetic characters.
3. **Sentence Boundaries & OOV Protocol**: Prepend `<s>` to evaluate initial token conditional probability $P(w_1 \mid \text{<s>})$, append `</s>`, and replace rare words with `<unk>` strictly using training frequencies.
4. **Model Training**: Unigram, Bigram, and Trigram count estimation.
5. **Smoothing & Evaluation**: Compare MLE, Laplace, Lidstone, Linear Interpolation, and Interpolated Kneser-Ney via Perplexity (PP).
"""

# %%
import sys
import random
from pathlib import Path

# Ensure repo root is on python path
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import (
    basic_tokenize,
    build_vocabulary,
    replace_oov_tokens,
    load_corpus_from_file_or_hf,
)
from src.ngram import NGramLM
from src.viz import plot_ngram_frequency, plot_perplexity_comparison

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# %% [markdown]
"""
## 1. Load Ewe Corpus
You can drop your Ewe text file into `data/raw/low_resource/ewe.txt` (or provide a Hugging Face dataset ID).
If the file is not yet placed, the pipeline automatically falls back to curated multi-domain Ewe sentences for seamless local execution.
"""

# %%
# Load from Grand Unified Mega-Corpus, raw low-resource file, or curated fallback
UNIFIED_TRAIN = REPO_ROOT / "data" / "processed" / "unified" / "train.txt"
UNIFIED_TEST = REPO_ROOT / "data" / "processed" / "unified" / "test.txt"
LOCAL_RAW = REPO_ROOT / "data" / "raw" / "low_resource" / "ewe.txt"

if UNIFIED_TRAIN.exists() and UNIFIED_TEST.exists():
    print(f"Loading Grand Unified Ewe Corpus from {UNIFIED_TRAIN.parent}...")
    with open(UNIFIED_TRAIN, 'r', encoding='utf-8') as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(UNIFIED_TEST, 'r', encoding='utf-8') as f:
        test_lines = [l.strip() for l in f if l.strip()]
    # For responsive notebook execution, select a clean 10,000-sentence sample (or use all)
    SAMPLE_SIZE = 10000
    train_lines = train_lines[:SAMPLE_SIZE]
    test_lines = test_lines[:SAMPLE_SIZE // 10]
    print(f"Loaded {len(train_lines):,} training sentences and {len(test_lines):,} test sentences.")
elif LOCAL_RAW.exists():
    all_lines = load_corpus_from_file_or_hf(str(LOCAL_RAW))
    split_idx = int(0.8 * len(all_lines))
    train_lines = all_lines[:split_idx]
    test_lines = all_lines[split_idx:]
    print(f"Loaded {len(train_lines):,} train and {len(test_lines):,} test sentences from {LOCAL_RAW}.")
else:
    print("Using representative curated Ewe sentences...")
    train_lines = [
        "Woezɔ loo, miawo katã míedi ŋutifafa le dukɔa me",
        "Efoa nyuie mah? Nyee, mefo nyuie, akpe kaka",
        "Kofi yi suku le Keta egbe ŋdi",
        "Ama fle nuɖuɖu vivi le asime le Ho",
        "Míeyi aƒeme kaba elabena tsi le dzadzam le Aflao",
        "Nufiala fia nu nusrɔ̃lawo nyuie le suku me",
        "Mia dogo le etsɔ me ne Mawu lɔ̃",
        "Devi sia nya nu ŋutɔ le eƒe nusɔsrɔ̃ me",
        "Míedi be míawɔ dɔ le ɖekawɔwɔ me",
        "Ɖo to nyuie ne nàse nya si gblɔm wole"
    ]
    test_lines = [
        "Ŋutsu la kple nyɔnu la woyi agble me",
        "Mía kplɔlawo le dɔ wɔm be dukɔa nade ŋgɔ"
    ]
    print(f"Curated sentences: {len(train_lines)} train, {len(test_lines)} test.")


# %% [markdown]
"""
## 2. Unicode Tokenization & Leak-Free Train/Test Split
Ewe contains distinctive phonemes and glyphs (`ɖ`, `ƒ`, `ɣ`, `ŋ`, `ɔ`, `ɛ`, `ʋ`). Our tokenizer uses Unicode NFC normalization to preserve tone markers and prevent morphological fragmentation.
"""

# %%
# Tokenize using Unicode NFC normalization preserving Ewe characters
train_tokens = [basic_tokenize(line) for line in train_lines if line.strip()]
test_tokens = [basic_tokenize(line) for line in test_lines if line.strip()]

# Induce closed vocabulary STRICTLY from training partition to prevent leakage
vocab, freqs = build_vocabulary(train_tokens, min_freq=1)
train_clean = replace_oov_tokens(train_tokens, vocab)
test_clean = replace_oov_tokens(test_tokens, vocab)

total_train_words = sum(len(s) for s in train_clean)
total_test_words = sum(len(s) for s in test_clean)
print(f"Vocabulary size (|V|): {len(vocab):,} unique word types")
print(f"Training tokens: {total_train_words:,} words across {len(train_clean):,} sentences")
print(f"Test tokens: {total_test_words:,} words across {len(test_clean):,} sentences")


# %% [markdown]
"""
## 3. Training Ewe N-Gram Models with Smoothing
We train statistical models with prepended `<s>` start tokens to evaluate conditional probabilities: $P(w_1 \mid \text{<s>})$.
"""

# %%
# 1. Unigram with Laplace (Add-1.0)
unigram = NGramLM(n=1, smoothing="laplace", k=1.0).fit(train_clean, vocab=vocab)

# 2. Bigram with Laplace (Add-1.0)
bigram_laplace = NGramLM(n=2, smoothing="laplace", k=1.0).fit(train_clean, vocab=vocab)

# 3. Bigram with Lidstone (Add-0.1)
bigram_lidstone = NGramLM(n=2, smoothing="laplace", k=0.1).fit(train_clean, vocab=vocab)

# 4. Trigram with Linear Interpolation (Weights: 0.1, 0.3, 0.6)
trigram_interp = NGramLM(n=3, smoothing="interpolation").fit(train_clean, vocab=vocab)
trigram_interp.set_interpolation_weights([0.1, 0.3, 0.6])

# 5. 4-Gram with Linear Interpolation (Weights: 0.05, 0.15, 0.3, 0.5) - Sweet Spot at Scale!
fourgram_interp = NGramLM(n=4, smoothing="interpolation").fit(train_clean, vocab=vocab)
fourgram_interp.set_interpolation_weights([0.05, 0.15, 0.30, 0.50])

# 6. Bigram with Interpolated Kneser-Ney (Continuation Probabilities)
bigram_kn = NGramLM(n=2, smoothing="kneser_ney").fit(train_clean, vocab=vocab)

print("All Ewe n-gram models trained successfully.")


# %% [markdown]
"""
## 4. Intrinsic Evaluation: Perplexity on Unseen Ewe Test Split
"""

# %%
models = {
    "Unigram (Laplace)": unigram,
    "Bigram (Laplace)": bigram_laplace,
    "Bigram (Lidstone k=0.1)": bigram_lidstone,
    "Trigram (Interpolation)": trigram_interp,
    "4-Gram (Interpolation)": fourgram_interp,
    "Bigram (Kneser-Ney)": bigram_kn,
}

results = {}
print(f"{'Model Architecture / Smoothing':<30} | {'Test Perplexity':<15}")
print("-" * 50)
for name, model in models.items():
    ppl = model.perplexity(test_clean)
    results[name] = ppl
    print(f"{name:<30} | {ppl:<15.2f}")

# Plot comparison chart and save to figures/
fig_path = REPO_ROOT / "figures" / "ngram_perplexity_comparison.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
plot_perplexity_comparison(
    list(results.keys()),
    list(results.values()),
    title="Ewe Language Model - Test Perplexity Benchmark",
    save_path=fig_path,
)
print(f"\nBenchmark plot saved to: {fig_path}")


# %% [markdown]
"""
## 5. Text Generation in Ewe with Sampling Temperature
"""

# %%
print("=== Autoregressive Ewe Text Generation (T=0.7) ===\n")
for name, model in models.items():
    sample = model.generate(max_length=12, temperature=0.7)
    print(f"[{name}]:\n  -> '{sample}'\n")

