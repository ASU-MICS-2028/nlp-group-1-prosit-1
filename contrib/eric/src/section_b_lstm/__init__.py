"""
Section B, Question 2: a small LSTM trained from scratch on exactly the Ewe tokens the best n-gram uses, to
test whether n-grams beat neural models on low-resource data. It reuses the n-gram's splits, tokenizer and
scoring from src.section_b_ngram so the two numbers compare. It is unrelated to the Section C LLM.

    python -m src.section_b_lstm.run_baseline --dataset 2 --config small   -> results/section_b_lstm/
"""
