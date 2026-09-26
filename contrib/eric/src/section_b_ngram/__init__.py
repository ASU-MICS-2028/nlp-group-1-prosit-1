"""
Section B: statistical n-gram language models for Ewe (Èʋegbe). The models are counts plus smoothing
(MLE, Laplace, interpolation, Kneser-Ney); there is no neural network here.

    python -m src.section_b_ngram.build_datasets          raw Ewe files -> data/processed/
    python -m src.section_b_ngram.run_sweep --dataset all  5 tokenizers x N = 1..6 -> results/section_b_ngram/
"""
