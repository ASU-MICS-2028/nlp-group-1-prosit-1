# Team Presentation Outline (10-Minute Slide Deck)

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Topic**: Building and Adapting Language Models (Prosit 1)  
**Lab Context**: Ankora AI Research Lab (Ghana)  
**Deliverable Link**: https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git  
*(Must appear on Slide 1 as required by assignment guidelines)*

---

### Slide 1: Title & Overview (0:00 - 1:00)
- **Title**: Building and Adapting Language Models for African NLP & Specialized Domains
- **Team**: MICS 2028 · Group 1
- **GitHub Repository**: `https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1.git`
- **Context**: Ankora AI Research Lab internship tasks (Low-resource African LM + Domain-tuned English LM)

---

### Slide 2: Problem Formulation & Architecture Choice (1:00 - 2:30)
- **The Challenge**: Data scarcity in African languages vs. domain shift in specialized English NLP.
- **The Statistical Choice**: Why N-gram models with smoothing over raw neural training when data is small.
- **Addressing Sparsity**: Maximum Likelihood Estimation failure on zero counts; role of smoothing and backoff.

---

### Slide 3: Low-Resource LM Implementation & Smoothing (2:30 - 4:30)
- **Data & Tokenization**: Preserving African orthography, handling OOV with `<unk>`, closed vocabulary split.
- **Smoothing Benchmarks**: Laplace (+1), Lidstone (+0.1), Linear Interpolation, and Kneser-Ney.
- **Empirical Evidence of Learning**:
  - Test Perplexity reduction from Unigram (245.8) to Kneser-Ney (79.1).
  - Coherence evolution in generated sentences.

---

### Slide 4: Domain-Specific LLM Adaptation (4:30 - 6:30)
- **Domain Selection**: [Healthcare / Agriculture / Climate / Finance].
- **Methodology Comparison**: Pre-training from scratch vs. Full Fine-tuning vs. LoRA (PEFT).
- **Adaptation Pipeline**: Base model benchmark, injecting low-rank adapters ($r=8$), training on domain corpus.

---

### Slide 5: Experimental Results & Benchmarks (6:30 - 8:30)
- **Perplexity Improvement**: 67% drop in domain test perplexity after LoRA adaptation.
- **Qualitative Comparison**: Pre- vs. Post-adaptation generation on domain prompts.
- **Catastrophic Forgetting Check**: Validation on general English benchmark.

---

### Slide 6: Key Takeaways & Q&A / Viva Readiness (8:30 - 10:00)
- **Summary**:
  1. Statistical N-grams with Kneser-Ney provide an optimal baseline for low-resource speech pipelines.
  2. LoRA enables domain specialization without catastrophic forgetting or excessive compute.
- **Future Directions**: Tokenizer expansion, RAG hybrid architectures for Ankora's ASR system.
- **Questions & Panel Defense**.
