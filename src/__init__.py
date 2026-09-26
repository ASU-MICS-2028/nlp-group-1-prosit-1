"""
ICS554 Natural Language Processing, Prosit 1. Three separate models, one package each:

    section_b_ngram   Ewe n-gram language models: counts plus smoothing, no neural network   (Section B)
    section_b_lstm    LSTM trained from scratch on the same Ewe tokens, as a baseline        (Section B, Question 2)
    section_c_llm     pretrained English distilgpt2 adapted to agricultural Q&A with LoRA    (Section C)

Run everything from the repository root with python -m, for example python -m src.section_b_ngram.run_sweep.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the repository root
