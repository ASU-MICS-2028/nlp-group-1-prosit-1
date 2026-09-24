"""
Section C: a pretrained English language model (distilgpt2) adapted to agricultural question answering
with LoRA. It shares no code, data or weights with Section B.

    python -m src.section_c_llm.prepare_data        KisanVaani -> data/processed/domain_english/
    python -m src.section_c_llm.train_lora          adapters -> models/section_c_llm/, scores -> results/section_c_llm/
    python -m src.section_c_llm.benchmark_decoding  -> results/section_c_llm/decoding_benchmark.json
"""
