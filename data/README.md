# Data Directory

This directory stores datasets used in Prosit 1:
1. **Low-Resource African Language Dataset (`data/raw/low_resource/`)**:
   - Chosen language (e.g. Akan/Twi, Ewe, Ga, Yoruba, Hausa, or Masakhane text datasets).
   - Training, validation, and test splits for the n-gram statistical language model.
2. **Domain-Specific English Dataset (`data/raw/domain_english/`)**:
   - Domain of choice (e.g. Healthcare / Clinical text, Agriculture / AgTech, Legal, or Finance).
   - Paired or raw corpora used to adapt / fine-tune the English language model.

## Subdirectories
- `raw/`: Unprocessed text corpora or raw downloads.
- `processed/`: Cleaned, tokenized, and split files ready for training and evaluation.

> **Note**: Large corpora and raw datasets are gitignored. Do not commit large files or licensed datasets to GitHub.
