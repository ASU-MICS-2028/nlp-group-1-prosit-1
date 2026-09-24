# Data Directory

Raw and processed data are gitignored (size and licences). This file records where each raw file comes
from, what is in it, and the one command that rebuilds the processed splits from it.

## Section B, n-gram and LSTM: Ewe (`data/raw/low_resource/`)

Both Section B models (the n-gram and the LSTM baseline) use these splits.

| File | What it actually contains | Column used | Rows |
|---|---|---|---|
| `dataset_1_csv/EWE_ENGLISH.csv` | English/Ewe sentence pairs. A large share is Jehovah's Witnesses publications and Bible verses (after cleaning, 10.6% of its sentences mention Yehowa and 6.8% carry chapter:verse references), not folklore. Some English cells do not match their Ewe row, and 15 rows are binary garbage (dropped by the cleaner). Original source not recorded. | `EWE` | 28,614 |
| `dataset_2_json/eweenglishsentence(3).json` | PHPMyAdmin export of table `eweenglishsentence` (database `ewedictionarydb`). 477 rows are Glosbe dictionary example sentences (glosbe.com); 123 come from peterlin.pl/ewe, including personal introductions that name real people with birth dates and family details. | `ee_sentence` | 600 |
| `dataset_3_speech/selected transcribed audios.xlsx` | Transcriptions of spoken Ewe image descriptions: University of Ghana, project Waxal, locale `ee_gh`, 2023, 539 speakers. The sheet also holds speaker ID, gender, age and device (not used). | `Transcription` | 19,152 |
| `dataset_4_parquet/ewe_corpus.parquet` | 4,408,322 English/Ewe sentence pairs with an alignment score (columns `similarity`, `English`, `Ewe`), sorted by score. We use the first 200,000 rows (the highest-scored pairs). In a random sample of 15 kept sentences, at least 10 are Bible verses or Jehovah's Witnesses text. Downloaded from Hugging Face as `train-00000-of-00001.parquet`; the repository name was not recorded (open item). | `Ewe` | 4,408,322 |

Rebuild every split (per-dataset folders plus `unified`) with:

```bash
python -m src.section_b_ngram.build_datasets
```

Cleaning is in `src/section_b_ngram/data_pipeline.py`: drop binary garbage, strip HTML and URLs,
Unicode NFC, remove zero-width characters, collapse whitespace, keep lines with at least 2 words and 1
letter, deduplicate on the lowercased text (within and across sources, first occurrence wins), shuffle with
seed 42, split 80/10/10.

## Section C, fine-tuned LLM: English agriculture (`data/raw/domain_english/`)

`agriculture_qa.parquet` is `KisanVaani/agriculture-qa-english-only` from Hugging Face (Apache-2.0):
22,615 rows but only 2,212 distinct questions. `python -m src.section_c_llm.prepare_data` downloads it if
missing, keeps one row per question before shuffling (so no test question is also a training question), and
writes `data/processed/domain_english/{train,val,test}.jsonl` plus `stats.json`.

## Subdirectories
- `raw/`: unprocessed downloads, as above.
- `processed/`: cleaned splits written by the two commands above. Never edit these by hand.
