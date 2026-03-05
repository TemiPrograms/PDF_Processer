# 📄 PDF Processor – Gemini RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) system that enables semantic search and cited responses over PDF documents. Uses Google Gemini, FAISS, and Sentence Transformers.

---

## Problem

Mortgage industry professionals regularly work with lengthy PDFs such as loan applications, mortgage notes, title reports, and more. Extracting specific insights from these documents is time-consuming and inefficient.

Traditional keyword search:
- Fails to understand semantic meaning  
- Misses context-based answers  
- Cannot synthesize information  

Large Language Models alone:
- Hallucinate without grounding  
- Lack document traceability  
- Cannot cite specific pages


---

## Solution

PDF Processor is a Retrieval-Augmented Generation (RAG) application that:

- Extracts structured and scanned PDF text (OCR fallback)
- Chunks document content into searchable segments
- Generates semantic embeddings using Sentence Transformers
- Indexes embeddings locally with FAISS (CPU)
- Retrieves top-k relevant chunks for each query
- Generates context-constrained answers using Google Gemini
- Provides explicit page-level citations
- Offers an interactive Gradio web interface

The result is a grounded, citation-aware document assistant.

---

## Requirements

### System Requirements
- **Python:** 3.10 or higher  

### Python Libraries
- `pdfplumber` – PDF text extraction  
- `pytesseract` – OCR fallback for scanned PDFs  
- `faiss-cpu` – vector similarity search  
- `numpy` – numerical operations for embeddings  
- `sentence-transformers` – text embeddings  
- `gradio` – web interface  
- `nest_asyncio` – asynchronous LLM support  
- `llama-index` – Gemini API wrapper and chat message handling  

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```
---

## Architecture Overview

1. User uploads PDF  
2. Text extraction via `pdfplumber`  
3. OCR fallback via `pytesseract` if needed  
4. Text chunking  
5. Sentence embedding generation  
6. FAISS vector indexing  
7. User query embedding  
8. Top-k similarity retrieval  
9. Gemini LLM response constrained to retrieved context  
10. Answer returned with page citations  

---

## Key Engineering Decisions

### 1. Retrieval-Augmented Generation (RAG)

Instead of prompting Gemini directly with the entire document, this system:
- Retrieves only the most relevant chunks
- Grounds responses in document context
- Reduces hallucination risk
- Enables explicit citation

---

### 2. OCR Fallback Strategy

Many PDFs contain scanned images rather than selectable text.  
If `pdfplumber` fails to extract text, the system:

- Converts page to image
- Applies `pytesseract` OCR
- Continues processing seamlessly

This ensures robustness across document types.

---

### 3. Batched Embedding Generation

Embeddings are generated in batches to:
- Improve performance
- Reduce memory spikes
- Enable scalability for larger documents

---

### 4. Local FAISS Index (CPU)

Uses `IndexFlatL2` for:
- Lightweight local deployment
- No external vector database dependency
- Deterministic similarity search

---

### 5. Asynchronous LLM Calls

Gemini responses use `await llm.achat()` to:
- Avoid UI blocking
- Improve responsiveness
- Enable future streaming support

---

## Limitations

- FAISS index is in-memory (not persistent across sessions)
- Chunking uses fixed character window (no semantic splitting)
- Single-document support per session

---

## License

This project is open-source under the MIT License. Feel free to fork, adapt, and expand!

---

## References

1. [What is Computer Vision? – Viso.ai](https://viso.ai/computer-vision/what-is-computer-vision/)  
   Background and inspiration for OCR and document image processing.

2. [Top 12 Machine Learning Use Cases – TechTarget](https://www.techtarget.com/searchenterpriseai/feature/Top-12-machine-learning-use-cases-and-business-applications)  
   Motivation and context for ML-powered document analysis.

3. [When Does Adding Fancy RAG Features Work? – Towards Data Science](https://towardsdatascience.com/when-does-adding-fancy-rag-features-work/)  
   Insights on retrieval-augmented generation design decisions.

4. [facebookresearch/faiss](https://github.com/facebookresearch/faiss)  
   
## Author

Temi Olugbade  
Focus: Applied AI Systems, Secure LLM Architectures, and Intelligent Retrieval Systems
