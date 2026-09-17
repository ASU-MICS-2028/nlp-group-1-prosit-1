# Datasheet for Language Modeling Corpora

Based on Gebru et al., *Datasheets for Datasets* (2021). Documenting data provenance, composition, and collection for Prosit 1.

---

## 1. Motivation
- **For what purpose was the dataset created?**  
  To train and benchmark language models for (1) **Ewe (Èʋegbe)** within an automatic speech recognition (ASR) decoding pipeline, and (2) Agro-Extension domain specialization in English, fulfilling internship requirements from Ankora AI research lab.
- **Who created the dataset?**  
  Curated by Ashesi University MICS 2028 Group 1 students using open Ewe corpora (incorporating conversational utterances, local news, and cultural texts) and agricultural technical extension documentation from CSIR Ghana and agricultural advisory bulletins.

---

## 2. Composition
- **What do the instances represent?**  
  - *Ewe Language Corpus*: Conversational, cultural, and contemporary news utterances in Ewe (Èʋegbe), deliberately excluding purely religious texts (e.g., the Bible) to prevent distribution skew toward archaic liturgical grammar.
  - *Agro-Extension Corpus*: Technical advisory paragraphs covering plant pathology, crop management, and integrated pest management (fall armyworm, cassava mosaic virus, cocoa swollen shoot).
- **How many instances are there in total?**  
  - Ewe Corpus: Cleaned text lines in `data/raw/low_resource/ewe.txt` (or streamed from open Ewe datasets).
  - Agro-Extension Corpus: Curated technical paragraphs (~75,000 tokens).
- **Does the dataset contain sensitive or identifiable personal information?**  
  No. Sourced entirely from open public domain archives, academic repositories, and agricultural bulletins.

---

## 3. Collection Process
- **How was the data acquired?**  
  - Ewe text gathered from open language repositories (e.g., Menyo-20k, local broadcast transcripts, and cultural texts).
  - Agricultural extension text gathered from open-access agricultural bulletins and pest management guides.

---

## 4. Preprocessing & Cleaning
- **What cleaning or normalization was done?**  
  - Preserved native Ewe orthographic glyphs (open vowels `ɛ`, `ɔ`, bilabial fricatives `ƒ`, `ʋ`, retroflex `ɖ`, velar fricative `ɣ`, velar nasal `ŋ`) and tone diacritics under Unicode NFC normalization.
  - Sentence boundaries bounded by start token `<s>` and end token `</s>`.
  - Closed vocabulary induced strictly from the training split, with low-frequency tokens mapped to `<unk>` to prevent test leakage.

---

## 5. Uses
- **Primary Use**: Statistical n-gram language modeling and parameter-efficient domain adaptation.
- **Limitations**: Not suitable for automated clinical diagnoses or commercial chemical pesticide prescriptions without licensed agronomist review.
