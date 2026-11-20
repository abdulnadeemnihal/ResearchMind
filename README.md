# ResearchMind

ResearchMind is a local Retrieval-Augmented Generation (RAG) application for
asking questions about PDF documents. It uses Ollama for embeddings and chat,
FastAPI for the backend, and ChromaDB for vector storage.

## Features

- **PDF upload and indexing:** Upload PDFs and split their text into searchable,
  overlapping chunks.
- **Clean text chunking:** Remove repeated headers and footers, preserve
  sentence boundaries, and avoid splitting words mid-way.
- **Document management:** List, switch between, and delete uploaded documents
  from the web interface.
- **Document-scoped Q/A:** Search and answer questions only within the selected
  PDF document.
- **Suggested questions:** Generate up to eight PDF-specific questions with
  source pages using the local LLM.
- **Grounded responses:** Stream answers with source filenames and page
  numbers, and respond with "could not find" when retrieval is not relevant.
- **Browser-based interface:** Upload documents, view generated questions,
  and ask questions at `http://localhost:8000/`.
- **Retrieval evaluation:** Run `backend/eval.py` to measure on-topic versus
  off-topic retrieval and tune `MAX_DISTANCE`.
- **Local-first operation:** Keep document data and inference on the local
  machine through Ollama and ChromaDB.

## Requirements

- Python 3.10 or newer
- Ollama
- Ollama models: `qwen3.5:4b` and `embeddinggemma`

## Run

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/` in a browser.
