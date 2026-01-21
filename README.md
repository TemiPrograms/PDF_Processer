# 📄 PDF Processor – Gemini RAG Chatbot

An AI-powered application that allows users to inquire information from upload PDF documents.

## Features
- PDF text extraction with OCR fallback
- Sentence-transformer embeddings
- FAISS vector similarity search
- Gemini-based answers with page citations
- Interactive Gradio UI

## Setup

1. Clone the repository:
git clone [https://github.com/TemiPrograms/pdf-processor](https://github.com/TemiPrograms/PDF_Processer)
cd pdf-processor

2. Install dependencies:
pip install -r requirements.txt

3. Set environment variables:

Create a .env file in the project root with:
GOOGLE_API_KEY=your_api_key_here

Or export it directly in the terminal:
export GOOGLE_API_KEY="your_api_key_here"

4. Run the application:


Notes:
- This project uses FAISS (CPU) for vector similarity search.
- Users must provide their own Google Gemini API key.


