"""ResearchMind AI - minimal local RAG backend (FastAPI + Ollama + ChromaDB).

v2: cleaner chunking (no repeated headers, splits on sentences, no mid-word cuts).

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import io
import json
import re
import uuid
from collections import Counter
from pathlib import Path

import chromadb
import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent

# ---- config (move to config.py later) ----
LLM_MODEL = "qwen3.5:4b"
EMBED_MODEL = "embeddinggemma"
CHUNK_SIZE = 900       # target max characters per chunk
CHUNK_OVERLAP = 150    # characters of the previous chunk repeated at the start
MIN_CHUNK = 80         # drop tiny fragments
TOP_K = 4
MAX_DISTANCE = 0.65    # if even the best chunk is farther than this, say "not found"
NUM_CTX = 3072         # smaller context = less VRAM on the RTX 3050
DB_PATH = str(BASE_DIR / "chroma_db")   # always next to main.py, whatever folder you start uvicorn from

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


@app.get("/documents")
def list_documents():
    """One entry per uploaded PDF, for the UI's document list."""
    metas = collection.get(include=["metadatas"])["metadatas"]
    docs: dict[str, dict] = {}
    for m in metas:
        d = docs.setdefault(
            m["doc_id"], {"doc_id": m["doc_id"], "filename": m["filename"], "chunks": 0, "last_page": 0}
        )
        d["chunks"] += 1
        d["last_page"] = max(d["last_page"], m["page"])
    return sorted(docs.values(), key=lambda d: d["filename"].lower())


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
    doc_id: str | None = None


class SuggestionRequest(BaseModel):
    doc_id: str | None = None


@app.post("/suggested-questions")
def suggested_questions(req: SuggestionRequest):
    if collection.count() == 0:
        raise HTTPException(400, "Upload a document first")

    kwargs = {"include": ["documents", "metadatas"]}
    if req.doc_id:
        kwargs["where"] = {"doc_id": req.doc_id}
    stored = collection.get(**kwargs)

    chunks = []
    for text, meta in zip(stored["documents"], stored["metadatas"]):
        chunks.append((meta["page"], f"[{meta['filename']}, p.{meta['page']}]\n{text}"))
    chunks.sort(key=lambda item: item[0])

    if len(chunks) > 8:
        indexes = {round(i * (len(chunks) - 1) / 7) for i in range(8)}
        sampled_chunks = [chunks[index] for index in sorted(indexes)]
    else:
        sampled_chunks = chunks
    context = "\n\n".join(text for _, text in sampled_chunks)
    if not context:
        raise HTTPException(404, "Document has no indexed text")

    prompt = (
        "Generate 8 useful questions that a reader could answer using ONLY the document "
        "context below. Prefer specific, substantive questions such as hashing, B+ trees, "
        "definitions, comparisons, examples, advantages, disadvantages, and algorithms. "
        "Do not include generic questions or questions requiring outside knowledge. "
        "Return a JSON object with exactly one 'questions' array. Each item needs "
        "'question' and 'pages' fields. "
        "Pages must be source page numbers from the chunk labels.\n\n"
        f"Context:\n{context}"
    )
    suggestions = None
    for attempt in range(2):
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            think=False,
            format="json",
            options={"num_ctx": NUM_CTX, "temperature": 0.1 if attempt else 0.2},
        )
        content = response["message"]["content"].strip()
        try:
            parsed = json.loads(content)
            suggestions = parsed.get("questions") if isinstance(parsed, dict) else parsed
            if isinstance(suggestions, list):
                break
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        suggestions = None

    if suggestions is None:
        raise HTTPException(502, "The model did not return valid question JSON")

    clean = []
    for item in suggestions:
        question = item.get("question")
        pages = item.get("pages", [])
        if isinstance(question, str) and question.strip():
            clean.append({
                "question": question.strip(),
                "pages": sorted({
                    int(page) for page in pages
                    if isinstance(page, (int, float)) and float(page).is_integer()
                }),
            })
        if len(clean) == 8:
            break
    if not clean:
        raise HTTPException(502, "The model did not generate any questions")
    return {"questions": clean}


@app.post("/search")
def search(q: Question):
    """Retrieval only - use this to judge retrieval quality separately from the LLM."""
    if collection.count() == 0:
        raise HTTPException(400, "Upload a document first")
    kwargs = {"query_embeddings": embed([q.question]), "n_results": TOP_K}
    if q.doc_id:
        kwargs["where"] = {"doc_id": q.doc_id}
    res = collection.query(**kwargs)
    return [
        {"text": d, "filename": m["filename"], "page": m["page"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


@app.post("/ask")
def ask(q: Question):
    if collection.count() == 0:
        raise HTTPException(400, "Upload a document first")

    kwargs = {"query_embeddings": embed([q.question]), "n_results": TOP_K}
    if q.doc_id:
        kwargs["where"] = {"doc_id": q.doc_id}
    res = collection.query(**kwargs)
    if min(res["distances"][0]) > MAX_DISTANCE:
        return StreamingResponse(
            iter(["I could not find this in the uploaded documents."]),
            media_type="text/plain",
        )
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


# ----------------------------------------------------------------- web UI
# static/index.html is served at http://localhost:8000/  (same origin: no CORS issues,
# and other devices on your Wi-Fi can use it if you start uvicorn with --host 0.0.0.0)
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/app/")
