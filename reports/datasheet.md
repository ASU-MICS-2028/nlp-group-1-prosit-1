# Datasheet for Language Modeling Corpora

Based on Gebru et al., *Datasheets for Datasets* (2021). Documenting data provenance, composition, and collection for Prosit 1.

---

## 1. Motivation
- **For what purpose was the dataset created?**  
  To train and benchmark language models for (1) Akan/Twi within an automatic speech recognition (ASR) decoding pipeline, and (2) Agro-Extension domain specialization in English, fulfilling internship requirements from Ankora AI research lab.
- **Who created the dataset?**  
  Curated by Ashesi University MICS 2028 Group 1 students using the verified open-source `ghana-nlp/abena-twi-corpus` and agricultural technical extension documentation from CSIR Ghana and agricultural advisory bulletins.

---

## 2. Composition
- **What do the instances represent?**  
  - *Twi Language Corpus*: Multilingual conversational, cultural, and news utterances in Akan/Twi, deliberately excluding religious texts (e.g., the Bible) to prevent distribution skew toward archaic grammar.
  - *Agro-Extension Corpus*: Technical advisory paragraphs covering plant pathology, crop management, and integrated pest management (fall armyworm, cassava mosaic virus, cocoa swollen shoot).
- **How many instances are there in total?**  
  - Twi Corpus: Streamed from `ghana-nlp/abena-twi-corpus` with `SCALE_FACTOR = 0.05` (~2,500 sampled lines for Colab memory stability).
  - Agro-Extension Corpus: Curated technical paragraphs (~75,000 tokens).
- **Does the dataset contain sensitive or identifiable personal information?**  
  No. Sourced entirely from open academic repositories and public agricultural bulletins.

---

## 3. Collection Process
- **How was the data acquired?**  
  - Twi data streamed directly from Hugging Face Hub (`ghana-nlp/abena-twi-corpus`).
  - Agricultural extension text gathered from open-access agricultural extension bulletins and pest management guides.

---

## 4. Preprocessing & Cleaning
- **What cleaning or normalization was done?**  
  - Preserved native Akan orthographic glyphs (open-e `ɛ`, open-o `ɔ`) under Unicode NFC normalization.
  - Sentence boundaries bounded by start token `<s>` and end token `</s>`.
  - Closed vocabulary induced strictly from the training split, with low-frequency tokens mapped to `<unk>` to prevent test leakage.

---

## 5. Uses
- **Primary Use**: Statistical n-gram language modeling and parameter-efficient domain adaptation.
- **Limitations**: Not suitable for automated clinical diagnoses or commercial chemical pesticide prescriptions without licensed agronomist review.
