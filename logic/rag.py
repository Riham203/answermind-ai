from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Generator, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "indian_law")
MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "45"))


def small_talk_reply(query: str) -> str | None:
    text = query.strip().lower()
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }
    thanks = {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
    }

    if text in greetings:
        return (
            "**Hello.** How can I help you today?\n\n"
            "You can ask about:\n\n"
            "- **Constitution of India**\n"
            "- **IPC** (Indian Penal Code) / **BNS** (Bharatiya Nyaya Sanhita)\n\n"
            "**Disclaimer:** This is informational only and not formal legal advice."
        )

    if text in thanks:
        return (
            "You are welcome.\n\n"
            "- If you need more detail, ask for **section-wise** context.\n"
            "- I can compare **IPC vs BNS** when both matter.\n\n"
            "**Disclaimer:** This is informational only and not formal legal advice."
        )

    return None


@dataclass
class Citation:
    title: str
    section: str
    excerpt: str


def retrieve_relevant_sections(query: str, top_k: int = 3) -> List[Citation]:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    docs = vector_store.similarity_search(query=query, k=top_k)

    citations: List[Citation] = []
    for doc in docs:
        metadata = doc.metadata or {}
        citations.append(
            Citation(
                title=str(metadata.get("source_title", metadata.get("source", "Unknown Source"))),
                section=str(metadata.get("section", metadata.get("article", "N/A"))),
                excerpt=doc.page_content.strip(),
            )
        )
    return citations


def build_legal_answer(query: str, citations: List[Citation]) -> str:
    context = "\n\n".join(
        [
            f"Source: {c.title}\nSection: {c.section}\nExtract: {c.excerpt}"
            for c in citations
        ]
    )

    system_prompt = (
        "You are AnswerMind AI, a senior Indian advocate answering in a concise, Gemini-style layout.\n\n"
        "Evidence and accuracy:\n"
        "- Prefer the supplied legal context; cite **Act/Code** and **Section/Article** when you rely on it.\n"
        "- Dual-law awareness: distinguish **IPC** (historical) vs **BNS** (current). If both matter, explain the transition briefly.\n\n"
        "Output format (Markdown):\n"
        "- Lead with a short direct answer (1–3 sentences).\n"
        "- Use optional `##` / `###` headings to separate major sections (Gemini-style scanability).\n"
        "- Use **bold** for key legal terms, statutes, sections, and articles.\n"
        "- For practical guidance, use **numbered lists** (1., 2., 3.) as step-by-step actions.\n"
        "- Use **bullet lists** (- item) for requirements, exceptions, or secondary facts.\n"
        "- Keep paragraphs tight; avoid filler.\n\n"
        "Close every reply with a final line exactly:\n"
        "**Disclaimer:** Informational assistance only; not formal legal advice. Consult a qualified advocate for case-specific guidance."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "User Query:\n{query}\n\nRetrieved Legal Context:\n{context}\n\n"
                "Respond using the required Markdown structure and citations.",
            ),
        ]
    )

    model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    chain = prompt | model
    result = chain.invoke({"query": query, "context": context})
    return result.content


def generate_legal_answer(query: str, top_k: int = 3) -> tuple[str, List[Citation]]:
    quick_reply = small_talk_reply(query)
    if quick_reply is not None:
        return quick_reply, []

    try:
        citations = retrieve_relevant_sections(query=query, top_k=top_k)
        if not citations:
            return (
                "I could not find relevant legal passages in the vector database. "
                "Please ingest your legal documents into ChromaDB and try again.",
                [],
            )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(build_legal_answer, query, citations)
            try:
                answer = future.result(timeout=MODEL_TIMEOUT_SECONDS)
            except FutureTimeoutError:
                return (
                    "The model is taking longer than expected to respond. Please try a shorter query, "
                    "or retry in a moment.",
                    citations,
                )
        return answer, citations
    except Exception as exc:
        message = (
            "Unable to complete RAG lookup. Please ensure Ollama is running on localhost:11434 "
            f"with chat model '{OLLAMA_MODEL}' and embedding model '{OLLAMA_EMBED_MODEL}', "
            f"and Chroma data exists in '{CHROMA_DIR}'. "
            f"Technical detail: {exc}"
        )
        return message, []


def stream_text_by_word(text: str) -> Generator[str, None, None]:
    for word in text.split():
        yield word + " "
