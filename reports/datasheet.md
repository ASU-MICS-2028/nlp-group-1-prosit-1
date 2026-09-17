# Datasheet for Language Modeling Corpora

Based on Gebru et al., *Datasheets for Datasets* (2021). Documenting data provenance, composition, and collection for Prosit 1.

---

## 1. Motivation
- **For what purpose was the dataset created?**  
  To train and evaluate language models for (1) a low-resource African language within an ASR decoding pipeline, and (2) a specialized domain in English (Agriculture / Healthcare), under the internship brief from Ankora AI research lab.
- **Who created the dataset?**  
  Curated by Ashesi University MICS 2028 Group 1 students from public repositories, research publications, and open cultural archives.

---

## 2. Composition
- **What do the instances that comprise the dataset represent?**  
  Natural language sentences and paragraphs.
  - *Low-Resource Corpus*: Sentences in [Language, e.g. Akan/Twi or Yoruba/Ewe], spanning conversational utterances, news summaries, and cultural proverbs.
  - *Domain English Corpus*: Technical advisory paragraphs covering plant pathology, crop management, and agricultural extension services in West Africa.
- **How many instances are there in total?**  
  - Low-resource: [N] cleaned sentences (~50,000 tokens).
  - Domain English: [M] technical paragraphs (~75,000 tokens).
- **Does the dataset contain sensitive or identifiable personal information?**  
  No. All text is derived from public domain publications, agricultural bulletins, or open speech transcripts. No private personal data or confidential records are included.

---

## 3. Collection Process
- **How was the data acquired?**  
  Text was harvested from open-access sources (Masakhane GitHub, CSIR Ghana research bulletins, and public news broadcasts).
- **Who was involved and how were they compensated?**  
  Open-source datasets created by volunteer academic and language communities.

---

## 4. Preprocessing & Cleaning
- **What cleaning or normalization was done?**  
  - Removed markup, raw HTML tags, and corrupted encoding artifacts.
  - Strictly preserved native tone diacritics and non-ASCII orthographic glyphs (such as ɛ, ɔ, ŋ, gb, kp) using standard Unicode NFC normalization.
  - Replaced low-frequency vocabulary items with `<unk>` strictly using training partition counts to prevent test set data leakage.

---

## 5. Uses
- **Has the dataset been used for previous tasks?**  
  Low-resource corpora have been used in machine translation (Masakhane) and basic ASR acoustic alignment.
- **Are there tasks for which the dataset should not be used?**  
  The corpus should not be used as authoritative medical or legal advice without expert human verification.
