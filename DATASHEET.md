# Datasheet for the Prosit 1 corpora

Following Gebru et al., *Datasheets for Datasets* (2021). Provenance, composition, processing and limits of every
corpus this project uses. The counts come from `results/stage0_audit.md`, `results/multisource.json` and the
Section C scripts.

## 1. Motivation

- **Ewe (Èʋegbe)**: n-gram and neural language models for a low-resource language, for Ankora's speech-recognition
  scenario (Section B).
- **English domain text**: adapting pretrained English models to health and agriculture (Section C).
- **Who made it?** We collected no new text. We cleaned, deduplicated, filtered and split corpora made by others.

## 2. Composition

### Ewe

| Corpus | Where it comes from | What it is | Used as | Sentences after cleaning |
|---|---|---|---|---:|
| Web sentence pairs | https://huggingface.co/datasets/ghananlpcommunity/english-ewe-sentence-pairs-4m (file `train-00000-of-00001.parquet`) | 4,408,322 English/Ewe pairs from automatic sentence mining, with an alignment score. 77.4% of rows are duplicates; 70.3% of the unique Ewe sentences contain no Ewe letter (spam, other languages). | Main corpus | 295,198 (train 265,882 / dev 14,705 / test 14,611) |
| Bible/JW sentence pairs | EWE-English Bilingual Pairs (Gbedevi & Tchaye-Kondi, Kaggle 2020, DOI 10.34740/KAGGLE/DSV/1462736): https://www.kaggle.com/datasets/tchaye59/eweenglish-bilingual-pairs (file `EWE_ENGLISH.csv`, columns `EWE`, `ENGLISH`) | 28,614 English/Ewe pairs crawled from jw.org (Jehovah's Witnesses publications and Bible verses), an Ewe course PDF, the Universal Declaration of Human Rights and peterlin.pl. | Extra source | 18,156 new (7,721 were already in the main corpus) |
| Dictionary examples | A PHPMyAdmin export of a database `ewedictionarydb` (file `eweenglishsentence(3).json`; ⟦where the export was obtained not recorded⟧). Each row records its original web page: 477 from the Glosbe Ewe–English dictionary (https://glosbe.com/ee/en/) and 123 from https://www.peterlin.pl/ewe/ (short texts such as personal introductions and weekday names). | 600 example sentences with English translations. | Extra source | 367 new |
| Spoken Ewe | UGSpeechData, University of Ghana (Waxal project, 2023, locale `ee_gh`): https://www.scidb.cn/en/detail?dataSetId=bbd6baee3acf43bbbc4fe25e21077c8a (file `selected transcribed audios.xlsx`). Described in Wiafe et al., *Data in Brief* 2025, https://doi.org/10.1016/j.dib.2025.111880. The Waxal data is also released at https://huggingface.co/datasets/google/WaxalNLP (CC-BY-4.0). | Transcriptions of spoken descriptions of images by 539 speakers. The only conversational Ewe we have. | Extra source | 19,150 new |

**Religious skew.** Sentences naming Yehowa or carrying a chapter:verse reference: web 12.8%, Bible/JW CSV 22.0%,
dictionary 15.5%, speech 0%. A broader keyword list (Mawu, Yesu, Kristo, Biblia, Israel, …) marks 35.0% of the web
training split. The web-trained model is a model of written, largely religious Ewe: on spoken Ewe its bits per
character rise from 1.45 to 2.12, and 11% of spoken words are unknown to it.

**Language purity.** Our Ewe-letter filter keeps a sentence only if it has an Ewe-specific letter. GlotLID (a
language identifier that knows Ewe, Ga, Fon, Twi and Adangme) calls 92.4% of the kept web sentences Ewe; the rest are
mostly Ga, Aja, Nzema and Adangme, which share Ewe's letters. No Ewe speaker has audited a sample.

### English (Section C)

| Corpus | Where it comes from | Licence | Used |
|---|---|---|---|
| Health | PubMedQA, unlabeled part: https://huggingface.co/datasets/qiaojin/PubMedQA (`pqa_unlabeled`) | MIT | 8,000 train / 300 valid / 500 test abstracts, split by document hash |
| Agriculture | https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only: 22,615 rows but only 2,212 distinct questions | Apache-2.0 | Qwen experiment: split by document; 9.6% of test questions also occur in training with another answer, so results were re-scored on the 104 leak-free test documents (unchanged). distilgpt2 experiment: one row per question before splitting (1,769 / 221 / 222). |
| General English | WikiText-2 test paragraphs: https://huggingface.co/datasets/Salesforce/wikitext (`wikitext-2-raw-v1`) | CC BY-SA | Test only, to measure forgetting |
| High-resource English | WikiText-103 training text: https://huggingface.co/datasets/Salesforce/wikitext (`wikitext-103-raw-v1`), 3.33M sentences after deduplication | CC BY-SA | Section B's high-resource comparison |

### Models we started from

| Model | Where it comes from | Used for |
|---|---|---|
| Qwen2.5-0.5B (base) | https://huggingface.co/Qwen/Qwen2.5-0.5B (downloaded from the ModelScope mirror, https://modelscope.cn/models/Qwen/Qwen2.5-0.5B, and checked against HuggingFace's SHA-256) | Section C, main experiment |
| SmolLM2-135M (base) | https://huggingface.co/HuggingFaceTB/SmolLM2-135M (same mirror and check) | Section C, second model |
| distilgpt2 | https://huggingface.co/distilbert/distilgpt2 | Section C, CPU experiment |
| GlotLID | https://huggingface.co/cis-lmu/glotlid | Grading the Ewe filter only |

## 3. Personal information

- The dictionary export includes personal introductions that name real people with birth dates and family details
  (public web pages).
- The speech spreadsheet holds speaker ID, gender, age and recording device. We use only the transcription column.
- No Ewe data is committed to the repository. The English Section C splits are (their sources are MIT, Apache-2.0 and CC BY-SA).

## 4. Processing

All Ewe text: Unicode NFC; whitespace collapsed; look-alike letters mapped (Icelandic Ð/ð → Ewe Ɖ/ɖ, Greek ε → ɛ);
for the extra sources also broken binary rows dropped, HTML tags, URLs and zero-width characters removed, and lines
kept only with at least two words and one letter. Exact duplicates removed within and across sources. Split by a hash
of a near-duplicate key (lower-cased, digits and punctuation removed), so a verse with and without its number cannot
land in both train and test. Code: `stage0_audit.py`, `stage0_sources.py`.

## 5. Uses and limits

- **Intended use**: coursework experiments on language modelling. Not for producing advice: the adapted English
  models write fluent, often wrong answers.
- **Licences**: the web corpus's licence is not stated on its page, and the Kaggle pairs give no standard licence
  ("Other"; the authors say the sources carry no copyright restrictions), the dictionary export's licence is
  unknown, and the speech data is CC-BY-4.0 on HuggingFace. Do not redistribute the Ewe data or models trained on it.
- **Register**: largely religious written text; spoken Ewe is under-represented even with the Waxal transcriptions.
