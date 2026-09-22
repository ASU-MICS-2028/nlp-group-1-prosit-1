# Datasheet for Language Modeling Corpora

Based on Gebru et al., *Datasheets for Datasets* (2021). Documents provenance, composition and processing
for Prosit 1. Every count below comes from `data/processed/*/stats.json`, written by
`scripts/build_ewe_datasets.py` and `src/prepare_domain_data.py`; see `data/README.md` for the raw files.

---

## 1. Motivation
- **For what purpose was the dataset assembled?**
  1. **Ewe (Èʋegbe)**: statistical n-gram language models for Ankora's speech recognition scenario under data scarcity, comparing 5 tokenizers and n-gram orders $N=1\dots 6$.
  2. **English agriculture Q&A**: adapting distilgpt2 to an agricultural advisory domain with LoRA.
- **Who created it?** We (Ashesi MICS 2028, Group 1) did not collect new text. We combined existing corpora made by others (below), then cleaned, deduplicated and split them.

---

## 2. Composition

### Ewe corpora

| Source | What it contains | Sentences kept | Train / Val / Test | Train words |
|---|---|---:|---|---:|
| Dataset 1 (`EWE_ENGLISH.csv`) | English/Ewe sentence pairs. A large share is Jehovah's Witnesses publications and Bible verses: 10.6% of kept sentences mention Yehowa, 6.8% carry chapter:verse references, and 5 of 6 randomly sampled rows were of this kind. Original source not recorded. | 25,837 | 20,669 / 2,583 / 2,585 | 518,219 |
| Dataset 2 (`eweenglishsentence(3).json`) | Database export: 477 Glosbe dictionary example sentences plus 123 sentences from peterlin.pl (personal introductions, weekday names, short stories). | 526 | 420 / 52 / 54 | 9,642 |
| Dataset 3 (`selected transcribed audios.xlsx`) | Transcribed spoken descriptions of images: University of Ghana, Waxal project, 2023, 539 speakers, locale `ee_gh`. | 19,150 | 15,320 / 1,915 / 1,915 | 508,630 |
| Dataset 4 (`ewe_corpus.parquet`) | First 200,000 rows of a 4,408,322-row English/Ewe sentence-pair file sorted by alignment score. At least 10 of 15 randomly sampled kept sentences are Bible verses or Jehovah's Witnesses text. Hugging Face repository not recorded. | 80,183 | 64,146 / 8,018 / 8,019 | 866,411 |
| **Unified** (all four, deduplicated across sources) | 5.1% of sentences mention Yehowa, 6.2% carry chapter:verse references. | **123,511** | **98,808 / 12,351 / 12,352** | **1,874,130** |

The unified corpus keeps 25,837 sentences from Dataset 1, 413 from Dataset 2, 19,150 from Dataset 3 and
78,111 from Dataset 4; the rest were duplicates of sentences already taken from an earlier source.

### English agriculture corpus
`KisanVaani/agriculture-qa-english-only` (Hugging Face): 22,615 rows but only 2,212 distinct questions.
After keeping one row per question: 1,769 train / 221 validation / 222 test pairs. The LoRA runs train on
the first 500 training pairs (a CPU time budget) and are evaluated on all 222 test pairs.

- **Does the data contain personal information?** Yes, in two places. Dataset 2 includes personal
  introductions that name real people and give birth dates and family details (public web pages).
  The raw Dataset 3 spreadsheet holds speaker IDs, gender and age; we use only the transcription column.
  None of the data is committed to the repository.

---

## 3. Collection Process
- Datasets 1, 2 and 4 were downloaded as files by the team; where each came from is recorded in
  `data/README.md` (two origins are not recorded and are listed as open items).
- Dataset 3 is an export of transcriptions from the University of Ghana Waxal speech project.
- The agriculture corpus is downloaded from Hugging Face by `src/prepare_domain_data.py`.

---

## 4. Preprocessing & Cleaning
All in `src/data_pipeline.py` (Ewe) and `src/prepare_domain_data.py` (English):
1. **Corrupted rows dropped**: lines with control bytes or escaped binary (15 such rows in Dataset 1).
2. **Markup removed**: HTML/XML tags and URLs.
3. **Unicode NFC normalization**, so a base letter and its tone mark are stored the same way everywhere.
   (Nasalized ɔ̃ and ɛ̃ have no single precomposed code point, so the tokenizer regex must also accept
   combining marks, `\u0300-\u036f`.)
4. **Lookalike letters mapped**: capital eth Ð and ð to Ɖ and ɖ, Greek ε to ɛ. Ð looks identical to Ɖ but
   lowercases to ð, so "Ðe" and "Ɖe" had been counted as different words (5,026 lines of the previous unified training split were affected).
5. **Zero-width characters removed, whitespace collapsed**; lines need at least 2 words and 1 letter.
6. **Deduplication** on the lowercased sentence, within and across sources (first occurrence wins);
   English Q&A deduplicated on the normalized question.
7. **Split** 80/10/10 after a seed-42 shuffle. Deduplication happens before the split, so a test sentence
   (or test question) never also appears in training.
8. **Closed vocabulary** from the training split only: tokens seen once in training become `<unk>`, so
   `<unk>` has a real probability for the unknown words it stands for at test time.

---

## 5. Uses & Limitations
- **Intended use**: coursework experiments on n-gram language modeling for Ewe and LoRA domain adaptation.
- **Religious skew**: a large part of the Ewe text is Bible and Jehovah's Witnesses material, so the models
  will favour that register over everyday or technical speech. Only Dataset 3 is conversational speech.
- **Licensing is unverified** for every source; do not redistribute the data or models trained on it.
- **The English corpus is small once deduplicated** (2,212 questions), and the adapted model's answers are
  fluent but often factually wrong (see `reports/domain_adaptation_results.json`); they are not advice.
