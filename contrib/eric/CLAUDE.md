# AI Assistant Context & Guidelines — NLP Prosit 1

## Project Context
Course: ICS554 Natural Language Processing (Ashesi University, MSc/MPhil in Intelligent Computing Systems).
Client Scenario: Ankora (AI research lab in Ghana developing specialized speech recognition systems).
Tasks:
1. Low-resource African language model using statistical N-grams (MLE, Laplace, Linear Interpolation, Kneser-Ney).
2. Domain-adapted English language model using PEFT/LoRA on open causal LM (e.g., DistilGPT2, TinyLlama).
3. Section A, B, and C technical report writing and 10-minute presentation slides.

Grading Weight:
- Group Presentation: 20%
- Technical Report: 45% (Section A: 15%, Section B: 20%, Section C: 20%)
- Viva Quiz: 35% (Individual oral examination)

## Absolute Constraints
- **Explainability over magic**: We are students preparing for an individual viva quiz (35%). Prefer clear, step-by-step mathematical reasoning and explainable code over complex abstractions we cannot defend before faculty.
- **Strict Data Hygiene**: No leakage between train, validation, and test sets. Vocabularies must be formed exclusively on training splits.
- **Reproducibility**: Explicit random seed (`RANDOM_SEED = 42`) and relative paths only (`Path.cwd()` or `Path(__file__).parent`).
- **Clean Output**: Notebook outputs should be cleared before committing.

## Working Style
- Write concise justifications next to modeling choices.
- Emphasize evaluation via Perplexity (PPL), loss curves, and generation coherence.
- Remind users to record all assistant sessions at the top of `WORKLOG.md`.
