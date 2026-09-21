# Datasheet for Language Modeling Corpora

Based on Gebru et al., *Datasheets for Datasets* (2021). Documenting data provenance, composition, and collection for Prosit 1.

---

## 1. Motivation
- **For what purpose was the dataset created?**  
  To train and benchmark language models for:
  1. **Ewe (Èʋegbe)** within an automatic speech recognition (ASR) decoding pipeline for Ankora AI research lab under severe data scarcity, evaluating the empirical scaling laws and breaking points across $N=1\dots 6$ and 5 tokenization strategies.
  2. **Agro-Extension domain specialization** in English via parameter-efficient fine-tuning (PEFT / LoRA) targeting crop disease and pest management advisories.
- **Who created the dataset?**  
  Curated by Ashesi University MICS 2028 Group 1 students (Eric Elikplim Sunu et al.) by ingesting, normalizing, deduplicating, and fusing 4 distinct Ewe text repositories, alongside West African agricultural extension advisory documentation.

---

## 2. Composition
- **What do the instances represent?**  
  - *Ewe Corpus Composition (4 Curated Sources)*:
    1. **Cultural Literature & Folklore (`EWE_ENGLISH.csv`)**: 26,594 unique sentences (517,444 words) of idiomatic prose, folklore, and proverbs.
    2. **Personal Biographies & Dates (`eweenglishsentence(3).json`)**: 526 unique sentences (9,642 words) representing micro-data conditions.
    3. **Spoken Oral Audio Transcriptions (`selected transcribed audios.xlsx`)**: 19,151 unique sentences (508,655 words) from the University of Ghana Waxal speech project, capturing spontaneous spoken syntax.
    4. **Large-Scale Web & Aligned Corpus (`ewe_corpus.parquet`)**: 80,385 unique sentences (867,455 words) of deduplicated contemporary web and scripture text.
  - *Grand Unified Mega-Corpus*:
    - **124,396 clean unique sentences (~2,349,941 total tokens)**.
    - Partitioned via stratified 80/10/10 split: 99,516 Train (1,881,823 words), 12,439 Val (234,312 words), 12,441 Test (233,806 words).
    - 2,260 cross-domain duplicates removed.
  - *Agro-Extension Corpus*: Technical advisory paragraphs covering plant pathology, crop management, and integrated pest management (fall armyworm, cassava mosaic virus, cocoa swollen shoot).
- **Does the dataset contain sensitive or identifiable personal information?**  
  No. Sourced entirely from open public domain archives, academic research speech repositories (Waxal), and open-access agricultural bulletins.

---

## 3. Collection Process
- **How was the data acquired?**  
  - Ewe text collected from regional academic speech repositories (University of Ghana), literary parallel corpora, community translation projects, and web crawls.
  - Agricultural extension text compiled from CSIR Ghana agricultural bulletins and plant protection guides.

---

## 4. Preprocessing & Cleaning (The 5-Stage Pipeline)
- **What cleaning or normalization was done?**  
  1. **Unicode NFC Normalization**: Binds combining tone diacritics to vowels, preventing glyph fragmentation.
  2. **Specialized Ewe Character Preservation**: Preserves `ɖ`, `ƒ`, `ɣ`, `ŋ`, `ɔ`, `ɛ`, `ʋ` and all acute/grave/circumflex accents.
  3. **Tone-Aware Tokenization**: Remedied Python's `str.isalnum()` combining diacritic bug via regex `^[\w\u0300-\u036f]+$`.
  4. **Quality Filtering**: Strips HTML/XML tags, web URLs, pure numerical/punctuation lines, and fragments with fewer than 2 words.
  5. **Normalized Hash Deduplication**: Eliminated 2,260 cross-source duplicates using lowercase normalized hashing.
  6. **Closed Vocabulary Protocol**: Out-of-vocabulary tokens strictly induced from training sets and mapped to `<unk>` to eliminate test set data leakage.

---

## 5. Uses & Limitations
- **Primary Use**: Statistical N-gram language modeling, Weighted Finite-State Transducer (WFST) compilation for Kaldi/ASR decoders, and parameter-efficient domain adaptation.
- **Limitations**: Classical statistical N-gram models are constrained to a fixed context window ($N \le 6$) and cannot perform deep semantic reasoning.
