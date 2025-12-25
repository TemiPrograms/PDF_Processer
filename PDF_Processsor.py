import os
import pdfplumber
import pytesseract
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage
import nest_asyncio

nest_asyncio.apply()

# API Key 
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY not set. Please set it as an environment variable."
    )

llm = GoogleGenAI(model="gemini-2.5-flash")


# Initialize Gemini model
llm = GoogleGenAI(model="gemini-2.5-flash")

# embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chat_history = []  # Chat history
pdf_state = {"chunks": None, "index": None}  # PDF + FAISS index

# PDF processing
def extract_text(pdf_path, chunk_size=300):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                img = page.to_image(resolution=300)
                text = pytesseract.image_to_string(img.original)
            if not text:
                continue  # skip empty pages

            for j in range(0, len(text), chunk_size):
                chunk = text[j:j+chunk_size].strip()
                if chunk:
                    chunks.append({
                        "text": chunk,
                        "page": i + 1,
                        "source": os.path.basename(pdf_path)
                    })
    return chunks

# FAISS index with batching
def build_index(chunks, batch_size=50):
    texts = [c["text"] for c in chunks if c["text"].strip() != ""]
    chunks = [c for c in chunks if c["text"].strip() != ""]

    dim = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)

    # batch embedding
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        embeddings = embedding_model.encode(batch_texts, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        index.add(embeddings)

    return index, chunks

# Retrieve top-k chunks
def retrieve(query, index, chunks, k=3):
    query_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_emb, k)
    return [chunks[i] for i in I[0]]

# Generate answer using Gemini + retrieved context
async def gemini_answer(query, retrieved_chunks):
    context = "\n".join(
        f"Page {c['page']} from {c['source']}:\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""
You are a document assistant.
Answer ONLY using the context below.
Cite page numbers explicitly.

Context:
{context}

Question:
{query}
"""
    chat_history.append(ChatMessage(role="user", content=prompt))
    try:
        response = await llm.achat(chat_history)
        answer = response.message.content
        chat_history.append(ChatMessage(role="assistant", content=answer))
        return answer
    except Exception as e:
        return f"Error: {e}"

# Wrapper for Gradio (async → sync)
async def respond_async(user_input, chat_history_ui):
    if pdf_state["chunks"] is None:
        return chat_history_ui + [(user_input, "⚠️ Please upload a PDF first.")], ""
    retrieved = retrieve(user_input, pdf_state["index"], pdf_state["chunks"])
    reply = await gemini_answer(user_input, retrieved)
    chat_history_ui = chat_history_ui + [(user_input, reply)]
    return chat_history_ui, ""

# PDF upload handler
def upload_pdf(pdf_file):
    chunks = extract_text(pdf_file.name)
    if not chunks:
        return "❌ No text found in PDF."
    index, chunks = build_index(chunks)
    pdf_state["chunks"] = chunks
    pdf_state["index"] = index
    return f"✅ Processed {os.path.basename(pdf_file.name)} ({len(chunks)} chunks)"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## ⚡ Gemini RAG PDF Chatbot (Stable Async Version)")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process PDF")

    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="Ask a question")
    submit_btn = gr.Button("Send")

    # Connect PDF upload
    upload_btn.click(upload_pdf, inputs=pdf_input, outputs=status)

    # Connect chat (async wrapper)
    submit_btn.click(respond_async, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(respond_async, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

demo.launch()
