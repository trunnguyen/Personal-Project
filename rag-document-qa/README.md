#  RAG Document Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about them. The system retrieves relevant text chunks via vector similarity search and generates answers using a large language model.

---

##  Features

- Upload and process PDF documents
- Automatic text chunking with overlap
- Vector similarity search with FAISS
- Retrieval-Augmented Generation (RAG)
- Retrieval accuracy metrics printed to terminal
- Interactive Gradio web interface
- Source-aware responses with snippet citations

---

##  Architecture

1. **Document Upload** — PDF parsed with PyPDFLoader
2. **Text Chunking** — Recursive character splitting (1000-char chunks, 200-char overlap)
3. **Embedding Generation** — `all-MiniLM-L6-v2` via HuggingFace
4. **Vector Storage** — FAISS in-memory index
5. **Similarity Retrieval** — Top-3 chunks by cosine similarity
6. **LLM Response Generation** — Groq (Llama 3.1 8B)

---

##  Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| UI | Gradio |
| Language | Python 3.11 |

---

## Project Structure

```
rag-document-qa/
│
├── modules/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── app.py
├── .env
├── DEMO.png
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/trunnguyen/Personal-Project.git
cd rag-document-qa
```

Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
```

Get a free API key at [console.groq.com](https://console.groq.com).

---

##  Run the Application

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

##  Retrieval Metrics

After each query, retrieval accuracy is printed to the terminal:

```
── Retrieval Accuracy ───────────────────────
  Docs retrieved       : 3
  Avg similarity score : 0.9682
  Query terms in chunks: 100.00%
─────────────────────────────────────────────
```
![Metrics](DEMO/retrieval_metircs.png)

---

##  Demo

![Demo](DEMO/dashboard.png)

---

##  Example Workflow

1. Upload a PDF document
2. Ask a question about the document
3. The system retrieves the top 3 most relevant chunks
4. The LLM generates a grounded answer with source snippets

---

##  Future Improvements

- Multi-document support
- Streaming responses
- Vector database persistence (ChromaDB / Pinecone)
- Hybrid search (BM25 + embeddings)
- Conversation memory

---
