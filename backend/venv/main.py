"""ResearchMind AI - minimal local RAG backend (FastAPI + Ollama + ChromaDB).

v2: cleaner chunking (no repeated headers, splits on sentences, no mid-word cuts).

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import io
import re
import uuid
from collections import Counter

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
CHUNK_SIZE = 900       # target max characters per chunk
CHUNK_OVERLAP = 150    # characters of the previous chunk repeated at the start
MIN_CHUNK = 80         # drop tiny fragments
TOP_K = 4
NUM_CTX = 3072         # smaller context = less VRAM on the RTX 3050
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


# ---------------------------------------------------------------- chunking
def _norm_line(line: str) -> str:
    """Normalise a line so 'Page 3' and 'Page 4' count as the same header."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_boilerplate(pages: list[str]) -> set[str]:
    """Lines that repeat on many pages (running headers/footers, page numbers)."""
    if len(pages) < 3:
        return set()
    counts = Counter()
    for text in pages:
        # headers/footers are short; never treat long body lines as boilerplate
        counts.update({_norm_line(l) for l in text.splitlines() if l.strip() and len(l.strip()) <= 120})
    limit = max(3, int(len(pages) * 0.4))
    return {line for line, n in counts.items() if n >= limit}


def clean_page(text: str, boilerplate: set[str]) -> str:
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if _norm_line(line) in boilerplate:
            continue
        if re.fullmatch(r"\s*(page\s*)?\d+(\s*of\s*\d+)?\s*", line, flags=re.I):
            continue
        lines.append(line.strip())
    return "\n".join(lines)


def split_units(text: str) -> list[str]:
    """Break text into sentence-sized units. Bullets stay as their own units."""
    text = re.sub(r"\n(?=[•●▪\-•]\s)", "\n\n", text)       # bullets start a unit
    units = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s*\n\s*", " ", block).strip()          # un-wrap lines
        if not block:
            continue
        units.extend(s for s in re.split(r"(?<=[.!?;:])\s+", block) if s)
    return units


def hard_split(unit: str, size: int) -> list[str]:
    """Split a very long unit on word boundaries (never mid-word)."""
    words, parts, cur = unit.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            parts.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk_text(text: str) -> list[str]:
    units = []
    for u in split_units(text):
        units.extend(hard_split(u, CHUNK_SIZE) if len(u) > CHUNK_SIZE else [u])

    chunks, cur = [], []
    cur_len = 0
    for u in units:
        if cur and cur_len + len(u) + 1 > CHUNK_SIZE:
            chunks.append(" ".join(cur))
            # carry over trailing units as overlap
            overlap, size = [], 0
            for prev in reversed(cur):
                if size + len(prev) > CHUNK_OVERLAP:
                    break
                overlap.insert(0, prev)
                size += len(prev) + 1
            cur, cur_len = overlap, size
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append(" ".join(cur))

    return [c for c in chunks if len(c) >= MIN_CHUNK]


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


# ------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "chunks_stored": collection.count()}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    reader = PdfReader(io.BytesIO(await file.read()))
    raw_pages = [(p.extract_text() or "") for p in reader.pages]
    boilerplate = find_boilerplate(raw_pages)

    doc_id = uuid.uuid4().hex[:8]
    ids, texts, metas = [], [], []

    for page_num, raw in enumerate(raw_pages, start=1):
        for i, piece in enumerate(chunk_text(clean_page(raw, boilerplate))):
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
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunks": len(texts),
        "removed_repeated_lines": sorted(boilerplate)[:5],
    }


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
