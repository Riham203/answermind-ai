from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "indian_law")
DATA_DIR = Path(os.getenv("LEGAL_DATA_DIR", "data"))
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "8"))
MAX_RETRIES = int(os.getenv("INGEST_MAX_RETRIES", "3"))
# nomic-embed-text ~8192 tokens; cap chars to avoid runner crashes (EOF/500 on child servers)
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "8000"))


def clean_text_for_embedding(text: str, max_chars: int) -> str:
    """Strip NUL/C0 controls PDFs often introduce; cap length so Ollama's embed runner does not 500/EOF."""
    if not text:
        return " "
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if ch in "\n\t" or o >= 32:
            out.append(ch)
    cleaned = "".join(out).replace("\r\n", "\n").replace("\r", "\n")
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    cleaned = cleaned.strip()
    return cleaned if cleaned else " "


def sanitize_documents(docs: list[Document], max_chars: int) -> list[Document]:
    return [
        Document(
            page_content=clean_text_for_embedding(d.page_content, max_chars),
            metadata=dict(d.metadata),
        )
        for d in docs
    ]


def describe_chunk(doc: Document) -> str:
    src = doc.metadata.get("source", "?")
    page = doc.metadata.get("page", "?")
    n = len(doc.page_content or "")
    preview = (doc.page_content or "")[:120].replace("\n", " ")
    return f"source={src!r} page={page} len={n} preview={preview!r}"


def check_ollama_reachable(base_url: str, embed_model: str) -> None:
    """Fail fast with a clear message if Ollama is not running or not reachable."""
    tags_url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=8) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot connect to Ollama at {base_url}.\n"
            "  1. Install and start Ollama: https://ollama.com/download\n"
            "  2. Then run: ollama pull "
            + embed_model
            + "\n"
            f"  3. Optional: set OLLAMA_BASE_URL if Ollama runs elsewhere.\n"
            f"Original error: {exc}"
        ) from exc
    # Model name appears in tags JSON when pulled
    if embed_model and embed_model not in body:
        print(
            f"Warning: model '{embed_model}' may not be installed. Run:\n"
            f"  ollama pull {embed_model}\n"
            "Continuing anyway — embedding calls will fail if the model is missing.\n"
        )


def infer_source_title(filename: str) -> str:
    name = filename.lower()
    if "constitution" in name:
        return "Constitution of India"
    if "ipc" in name or "indian penal code" in name:
        return "Indian Penal Code, 1860 (IPC)"
    if "bns" in name or "bhartiya nyaya sanhita" in name or "bharatiya nyaya sanhita" in name:
        return "Bharatiya Nyaya Sanhita, 2023 (BNS)"
    return filename


def load_pdf_documents(data_dir: Path):
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{data_dir}'. Add legal PDFs to that folder first."
        )

    all_docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        source_title = infer_source_title(pdf_path.name)
        for doc in docs:
            doc.metadata["source"] = str(pdf_path)
            doc.metadata["source_title"] = source_title
            # Optional placeholder so app always has a section-like field.
            doc.metadata.setdefault("section", f"Page {doc.metadata.get('page', '?')}")
        all_docs.extend(docs)

    return all_docs


def main():
    print("Starting legal document ingestion...")
    print(f"Data directory: {DATA_DIR.resolve()}")
    print(f"Chroma persist directory: {Path(CHROMA_DIR).resolve()}")
    print(f"Collection: {CHROMA_COLLECTION}")
    print(f"Ollama embedding model: {OLLAMA_EMBED_MODEL} @ {OLLAMA_BASE_URL}")

    check_ollama_reachable(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL)

    raw_docs = load_pdf_documents(DATA_DIR)
    print(f"Loaded {len(raw_docs)} pages from PDF files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    chunks = sanitize_documents(chunks, EMBED_MAX_CHARS)
    print(f"Created {len(chunks)} chunks (sanitized for embedding, max {EMBED_MAX_CHARS} chars).")

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = chunks[start:end]
        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector_store.add_documents(batch)
                success = True
                break
            except Exception as exc:
                print(
                    f"Batch {start + 1}-{end} failed on attempt {attempt}/{MAX_RETRIES}: {exc}"
                )
                time.sleep(1.5 * attempt)

        if not success:
            print(f"Falling back to per-chunk ingestion for batch {start + 1}-{end}...")
            for i, doc in enumerate(batch, start=start + 1):
                doc_success = False
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        vector_store.add_documents([doc])
                        doc_success = True
                        break
                    except Exception as exc:
                        print(
                            f"Chunk {i}/{total} failed on attempt {attempt}/{MAX_RETRIES}: {exc}"
                        )
                        time.sleep(1.0 * attempt)
                if not doc_success:
                    print(f"Skipping chunk {i}/{total} after repeated failures.")
                    print(f"  {describe_chunk(doc)}")
                    # Last resort: very short text (Ollama embed child sometimes 500/EOF on bad inputs)
                    tiny = Document(
                        page_content=clean_text_for_embedding(doc.page_content, 512),
                        metadata=dict(doc.metadata),
                    )
                    try:
                        vector_store.add_documents([tiny])
                        print(f"  Re-ingested chunk {i} with 512-char cap.")
                        doc_success = True
                    except Exception as exc2:
                        print(f"  Fallback also failed: {exc2}")

        print(f"Ingested chunks up to {end}/{total}...")

    print("Ingestion complete. ChromaDB updated successfully.")


if __name__ == "__main__":
    main()
