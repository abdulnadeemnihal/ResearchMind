"""ResearchMind AI - minimal local RAG backend (FastAPI + Ollama + ChromaDB).

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import io
import uuid

import chromadb
import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader

# ---- config (move to config.py later) ----
LLM_MODEL = "qwen3.5:4b"
EMBED_MODEL = "embeddinggemma"
CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 150
TOP_K = 4
NUM_CTX = 4096         # keep small for 4GB VRAM
DB_PATH = "./chroma_db"

app = FastAPI(title="ResearchMind AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your future Next.js dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})


def chunk_text(text: str):
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(text), step):
        piece = text[start:start + CHUNK_SIZE].strip()
        if len(piece) > 50:
            yield piece


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


@app.get("/health")
def health():
    return {"status": "ok", "chunks_stored": collection.count()}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    reader = PdfReader(io.BytesIO(await file.read()))
    doc_id = uuid.uuid4().hex[:8]
    ids, texts, metas = [], [], []

    for page_num, page in enumerate(reader.pages, start=1):
        for i, piece in enumerate(chunk_text(page.extract_text() or "")):
            ids.append(f"{doc_id}-p{page_num}-c{i}")
            texts.append(piece)
            metas.append({"doc_id": doc_id, "filename": file.filename, "page": page_num})

    if not texts:
        raise HTTPException(422, "No extractable text (scanned PDF? it needs OCR)")

    for i in range(0, len(texts), 32):  # embed in small batches
        batch = slice(i, i + 32)
        collection.add(
            ids=ids[batch],
            documents=texts[batch],
            metadatas=metas[batch],
            embeddings=embed(texts[batch]),
        )
    return {"doc_id": doc_id, "filename": file.filename, "pages": len(reader.pages), "chunks": len(texts)}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    collection.delete(where={"doc_id": doc_id})
    return {"deleted": doc_id}


class Question(BaseModel):
    question: str


@app.post("/search")
def search(q: Question):
    """Retrieval only - use this to judge retrieval quality separately from the LLM."""
    if collection.count() == 0:
        raise HTTPException(400, "Upload a document first")
    res = collection.query(query_embeddings=embed([q.question]), n_results=TOP_K)
    return [
        {"text": d, "filename": m["filename"], "page": m["page"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


@app.post("/ask")
def ask(q: Question):
    if collection.count() == 0:
        raise HTTPException(400, "Upload a document first")

    res = collection.query(query_embeddings=embed([q.question]), n_results=TOP_K)
    context = "\n\n".join(
        f"[{m['filename']}, p.{m['page']}]\n{d}"
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    )
    sources = sorted({f"{m['filename']} p.{m['page']}" for m in res["metadatas"][0]})

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer ONLY from the context below. "
                "Cite sources like [filename, p.N]. If the context does not contain "
                "the answer, say you could not find it in the documents."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q.question}"},
    ]

    def stream():
        for part in ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            think=False,  # skip Qwen's reasoning phase: faster for RAG
            options={"num_ctx": NUM_CTX, "temperature": 0.2},
        ):
            yield part["message"]["content"]
        yield "\n\nSources: " + ", ".join(sources)

    return StreamingResponse(stream(), media_type="text/plain")