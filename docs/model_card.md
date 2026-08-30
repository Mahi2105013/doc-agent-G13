# Model Card: Ar-Raheeq Al-Makhtum Agent

## Model Details
* **Model Name:** doc-agent-G13
* **Version:** 1.0.0
* **Date:** 2026-08-30
* **Developers:** Group 13
* **Model Type:** Retrieval-Augmented Generation (RAG) based Agent
* **Primary Language:** Bengali (with English code-switching support)

## Intended Use
* **Primary Use Case:** Answering questions based on the corpus of "Ar-Raheeq Al-Makhtum" (The Sealed Nectar), "Priyonobi" and "Sirat Ibn Hisham". 
* **Target Audience:** Readers, researchers, and students interested in the life of Prophet Muhammad (PBUH).
* **Out of Scope:** Answering general knowledge questions outside the provided corpus, providing medical or legal advice, or engaging in unrestricted conversational AI tasks.

## Factors
* **Relevant Factors:** The model's performance relies heavily on the quality of Bengali OCR. Complex layouts, classical fonts, and degraded scans can affect retrieval accuracy.
* **Evaluation Factors:** The system is evaluated on exact-match fact extraction, retrieval recall@k, and groundedness to ensure no hallucinations.

## Metrics
* **Performance Metrics:**
  * Recall@k (k=10): Target >= 0.85
  * Groundedness: Target 1.0 (No Hallucinations)
  * Citation Accuracy: Target >= 0.90
  * Expected Calibration Error (ECE): Target <= 0.05 (for confidence scores)

## Training Data
* **Corpus:** 
  1. `ar_raheeq` (Ar-Raheeq Al-Makhtum) - Used for primary training split.
  2. `priyonobi` - Used for validation split.
  3. `sirat_ibn_hisham` - Used for testing split.
* **Preprocessing:** PDF pages processed with document layout detection and script-tuned Bengali OCR.

## Ethical Considerations
* **Data Bias:** The knowledge base is strictly limited to the provided corpus. It will not synthesize information from outside sources, mitigating external biases.
* **Hallucinations:** The agent is designed to abstain (say "I don't know") if sufficient evidence is not found in the retrieved chunks.
* **Safety & Security:** The system implements guardrails against prompt injection and ensures sensitive information is redacted (PII governance).

## Caveats and Recommendations
* **Limitations:** The agent might struggle with highly ambiguous queries or questions that require multi-hop reasoning across widely separated document sections.
* **Recommendations:** Users are encouraged to verify critical religious or historical facts by following the provided citations back to the original source text.
