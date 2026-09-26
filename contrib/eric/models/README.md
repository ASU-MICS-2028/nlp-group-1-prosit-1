# Trained Models (Section C only)

Only the Section C model has saved weights. The Section B n-gram models are rebuilt from counts in
minutes, and the Section B LSTM baseline's weights were not kept (its scores are in `results/section_b_lstm/`).

`section_c_llm/` holds two LoRA adapters for `distilgpt2`, adapted to agricultural question answering.
Each folder holds only the adapter: `adapter_config.json` and `adapter_model.safetensors` (147,456
trained parameters, about 0.6 MB). The 82M-parameter base model is not stored here; it is downloaded
from Hugging Face when an adapter is loaded.

| Folder | Training objective | Full Q&A perplexity | Answer-only perplexity | WikiText-2 perplexity |
|---|---|---:|---:|---:|
| `section_c_llm/standard/` | loss on every token (standard causal LM) | **28.13** | 30.00 | 78.44 |
| `section_c_llm/masked/` | loss on answer tokens only (question labels set to -100) | 51.39 | **29.09** | 77.24 |
| *(base `distilgpt2`, for reference)* | none | 56.08 | 37.77 | 73.19 |

Scores are token-weighted perplexities on 222 held-out questions (none of which occur in training) and
200 WikiText-2 paragraphs, copied from `results/section_c_llm/lora_results.json`.

## How they were made

```bash
python -m src.section_c_llm.prepare_data   # KisanVaani, one row per distinct question, 80/10/10 split (seed 42)
python -m src.section_c_llm.train_lora     # trains both adapters, scores all three models, writes this folder
```

Both adapters: rank $r=8$, $\alpha=32$, dropout 0.05, target `c_attn` (GPT-2's fused query/key/value
projection, stored as a `Conv1D`, hence `fan_in_fan_out=True`); 3 epochs over the first 500 training
pairs; AdamW, learning rate $5 \times 10^{-4}$ with linear decay; batch size 8; 96-token sequences;
seed 42. Rerunning the script overwrites these folders.

## Loading an adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("distilgpt2"), "models/section_c_llm/standard")
```

Use the versions in `requirements.txt` (peft 0.10.0, transformers 4.38.2).

## Caveats

- **Not for advice.** The answers are fluent but often wrong; see the samples in
  `results/section_c_llm/lora_results.json`.
- **Small, single-seed experiment.** 500 training pairs and one seed: the 30.00 vs 29.09 difference in
  answer perplexity is within noise.
- **Licences.** Base model `distilgpt2` and training data `KisanVaani/agriculture-qa-english-only` are both
  Apache-2.0.

## Checksums (SHA-256 of `adapter_model.safetensors`)

| Folder | SHA-256 |
|---|---|
| `section_c_llm/standard/` | `c16656ee4e158613f001d6111417974fe2da51a7f1b772dd999a5f4cdd33f856` |
| `section_c_llm/masked/` | `264b0e6a9cbd48fb7e91a9bcdb0c07f1bd8a5e243a8aae0d28c782d3aec7fb16` |
