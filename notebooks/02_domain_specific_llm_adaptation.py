# %% [markdown]
"""
# 02_domain_specific_llm_adaptation
"""

# %% [markdown]
"""
# 02. Domain-Specific English LM Adaptation (Agro-Extension Corpus)

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Project**: Prosit 1 (Ankora AI Research Lab)  
**Domain Corpus**: Agro-Extension Text Array (Pest Mitigation, Crop Rotation, Soil Fertilization)  
**Methodology**: Parameter-Efficient Fine-Tuning (PEFT / LoRA) targeting attention projections (`q_proj`, `v_proj`).

---
### Pipeline Overview
1. **Domain Corpus Preparation**: Specialized agricultural technical field guides.
2. **Base Foundation Model Evaluation**: Measure zero-shot domain perplexity baseline.
3. **LoRA Adapter Configuration**: Inject low-rank matrices ($r=8, \alpha=16$) into attention projections to reduce compute and prevent catastrophic forgetting.
4. **Fine-Tuning Execution**: Train causal LM on domain text.
5. **Post-Adaptation Benchmark**: Quantify perplexity reduction and audit domain prompt completions.
"""

# %%
import sys
from pathlib import Path
import torch

# Ensure repo root is on python path
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain_adaptation import load_model_and_tokenizer, prepare_dataset, evaluate_perplexity
from src.viz import plot_perplexity_comparison


# %% [markdown]
"""
## 1. Agro-Extension Domain Corpus
Field guides on tropical crop diseases, soil fertility, and integrated pest management in West Africa.
"""

# %%
agro_extension_corpus = [
    "Fall armyworm (Spodoptera frugiperda) infests maize during early vegetative whorl stages causing severe ragged feeding holes and frass.",
    "Cassava mosaic virus disease is transmitted primarily by the whitefly vector (Bemisia tabaci), resulting in chlorotic leaf mosaic and severe tuber stunting.",
    "Cocoa swollen shoot virus (CSSVD) causes characteristic red vein banding, stem swellings, and severe dieback in Theobroma cacao.",
    "Integrated pest management protocols advise intercropping maize with Desmodium legumes and trap plants such as Napier grass.",
    "Soil nitrogen deficiency in tropical tomato cultivation triggers progressive chlorosis in older lower foliage advancing toward young growth.",
    "Drip irrigation optimizes moisture delivery directly to root zones, substantially reducing foliar fungal blight spores.",
    "Post-harvest storage loss in grain stores is mitigated through hermetic bags preventing maize weevil (Sitophilus zeamais) infestation.",
    "Phosphorus fertilization accelerates seedling root establishment in dry savanna agro-ecological zones with acidic soils."
]

train_split = agro_extension_corpus[:6]
test_split = agro_extension_corpus[6:]
print(f"Domain Training samples: {len(train_split)}, Test samples: {len(test_split)}")


# %% [markdown]
"""
## 2. Load Base Model & Baseline Perplexity
"""

# %%
BASE_MODEL_NAME = "distilgpt2"  # Lightweight base model for rapid local/colab execution

print(f"Loading base foundation model: {BASE_MODEL_NAME}...")
base_model, tokenizer = load_model_and_tokenizer(BASE_MODEL_NAME, use_lora=False)

test_dataset = prepare_dataset(test_split, tokenizer, max_length=128)
baseline_ppl = evaluate_perplexity(base_model, tokenizer, test_dataset)
print(f"Base Model Zero-Shot Perplexity on Agro-Extension Test Set: {baseline_ppl:.2f}")


# %% [markdown]
"""
## 3. Configure & Train Low-Rank Adaptation (LoRA)
We attach LoRA matrices with $r=8, \alpha=16$ to the attention projections (`q_proj`, `v_proj`).
"""

# %%
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

adapted_model, _ = load_model_and_tokenizer(BASE_MODEL_NAME, use_lora=True, lora_r=8)
train_dataset = prepare_dataset(train_split, tokenizer, max_length=128)

training_args = TrainingArguments(
    output_dir=str(REPO_ROOT / "models" / "domain_adapted_checkpoint"),
    num_train_epochs=5,
    per_device_train_batch_size=2,
    learning_rate=5e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=adapted_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Starting LoRA Domain Adaptation Training...")
trainer.train()
print("Training completed successfully.")


# %% [markdown]
"""
## 4. Post-Adaptation Perplexity & Generation Comparison
"""

# %%
post_ppl = evaluate_perplexity(adapted_model, tokenizer, test_dataset)
print(f"LoRA Adapted Model Perplexity on Agro-Extension Test Set: {post_ppl:.2f}")
print(f"Perplexity Improvement: {baseline_ppl - post_ppl:.2f} points")

plot_perplexity_comparison(
    ["Base Model (Zero-Shot)", "Adapted Model (LoRA)"],
    [baseline_ppl, post_ppl],
    title="Agro-Extension Domain Adaptation Perplexity Benchmark",
    save_path=REPO_ROOT / "figures" / "domain_adaptation_perplexity.png",
)


# %% [markdown]
"""
### Qualitative Comparison on Agricultural Domain Prompts
"""

# %%
prompt = "Cassava mosaic virus disease is transmitted primarily by"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    out_base = base_model.generate(input_ids, max_new_tokens=25, do_sample=True, temperature=0.7)
    out_adapt = adapted_model.generate(input_ids, max_new_tokens=25, do_sample=True, temperature=0.7)

print("--- Prompt ---")
print(prompt)
print("\n--- Base Model Completion ---")
print(tokenizer.decode(out_base[0], skip_special_tokens=True))
print("\n--- Adapted Model Completion ---")
print(tokenizer.decode(out_adapt[0], skip_special_tokens=True))

