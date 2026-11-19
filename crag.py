"""Corrective RAG (CRAG) in one file.

A clean re-implementation of the CampusX corrective-rag notebooks using the
current LangChain / LangGraph / Pinecone / Tavily APIs.

Usage:
    uv run crag.py ingest [--force] [--max-pages N] [--namespace NS]
    uv run crag.py ask "your question"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

# langchain-community is being sunset (no standalone PDF-loader package exists
# yet), so PyPDFLoader still comes from it. Silence its import-time notice.
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")
from langchain_community.document_loaders import PyPDFLoader

# Gemini 3.x Flash-Lite ignores temperature and warns on every call; we still
# pass temperature=0 for models that honour it, so hide the repeated notice.
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field

# =============================================================================
# 1. CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DOCS_DIR = BASE_DIR / "documents"

# Pinecone (serverless)
INDEX_NAME = "crag-docs"
EMBED_DIM = 768  # nomic-embed-text output size
METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books")

# Embeddings (local, via Ollama)
EMBED_MODEL = "nomic-embed-text:latest"

# LLM. Override with GEMINI_MODEL / GEMINI_FALLBACK_MODEL in .env if needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")

# Chunking / retrieval
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

# Retrieval-evaluator thresholds (CampusX stage 3)
UPPER_TH = 0.7  # any doc above this  -> CORRECT
LOWER_TH = 0.3  # all docs below this -> INCORRECT; otherwise AMBIGUOUS


def require_env(name: str) -> str:
    """Return an env var or exit with a clear message (never prints the value)."""
    value = os.getenv(name, "").strip()
    if not value:
        sys.exit(f"[config] {name} is missing. Add it to .env (see .env.example).")
    return value


@lru_cache(maxsize=4)
def _chat_model(model: str) -> ChatGoogleGenerativeAI:
    """One Gemini chat model. The key is passed explicitly because the library
    defaults to GOOGLE_API_KEY, while this project uses GEMINI_API_KEY.
    Note: Gemini 3.x Flash-Lite uses fixed sampling and ignores temperature;
    temperature=0 is kept for models that do honour it."""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=require_env("GEMINI_API_KEY"),
        temperature=0,
        max_retries=2,
        timeout=60,
    )


def get_llm():
    """Flash-Lite, falling back to Flash if the primary call errors."""
    return _chat_model(GEMINI_MODEL).with_fallbacks(
        [_chat_model(GEMINI_FALLBACK_MODEL)]
    )


def structured_llm(schema: type):
    """`llm.with_structured_output(schema)` with the same Flash fallback.
    (Structured output must be bound to each model before adding fallbacks.)"""
    return _chat_model(GEMINI_MODEL).with_structured_output(schema).with_fallbacks(
        [_chat_model(GEMINI_FALLBACK_MODEL).with_structured_output(schema)]
    )


# =============================================================================
# 2. EMBEDDINGS: nomic-embed-text with task prefixes
# =============================================================================


class NomicEmbeddings(Embeddings):
    """Wraps OllamaEmbeddings and adds the task prefixes nomic-embed-text
    was trained with: "search_document: " for chunks, "search_query: " for
    queries. Every code path goes through this class, so the prefixes are
    always consistent between ingest and retrieval."""

    def __init__(self, model: str = EMBED_MODEL) -> None:
        self._inner = OllamaEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"search_query: {text}")


# =============================================================================
# 3. PINECONE: index creation + vector store
# =============================================================================
# Modern Pinecone SDK (v7+): `Pinecone(api_key=...)` client object.
# The old `pinecone-client` package and `pinecone.init()` are deprecated.


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    return Pinecone(api_key=require_env("PINECONE_API_KEY"))


def ensure_index(timeout_s: int = 120):
    """Create the serverless index if it does not exist, wait until ready,
    and return an Index handle."""
    pc = get_pinecone()
    if not pc.has_index(INDEX_NAME):
        print(f"[pinecone] creating index '{INDEX_NAME}' ({EMBED_DIM}d, {METRIC})")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    deadline = time.time() + timeout_s
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        if time.time() > deadline:
            sys.exit(f"[pinecone] index '{INDEX_NAME}' not ready after {timeout_s}s.")
        time.sleep(2)
    return pc.Index(INDEX_NAME)


def get_vectorstore(namespace: str = DEFAULT_NAMESPACE) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=ensure_index(), embedding=NomicEmbeddings(), namespace=namespace
    )


def namespace_count(index, namespace: str) -> int:
    """Number of vectors already stored in a namespace (0 if absent)."""
    ns = index.describe_index_stats().namespaces.get(namespace)
    return ns.vector_count if ns else 0


# =============================================================================
# 4. INGEST: load PDFs -> chunk -> embed -> upsert
# =============================================================================


def load_pdfs(max_pages: int | None = None) -> list[Document]:
    """Load every PDF in documents/. Keep only clean `source` and `page`
    metadata (PyPDFLoader adds many fields, some of which Pinecone rejects)."""
    pages: list[Document] = []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        for i, page in enumerate(PyPDFLoader(str(pdf)).lazy_load()):
            if max_pages is not None and i >= max_pages:
                break
            if not page.page_content.strip():
                continue
            page.metadata = {
                "source": pdf.name,
                "page": int(page.metadata.get("page", i)) + 1,  # 1-based
            }
            pages.append(page)
        print(f"[ingest] loaded {pdf.name}")
    return pages


def chunk_id(doc: Document, n: int) -> str:
    """Deterministic ID, so re-ingesting overwrites rather than duplicates."""
    key = f"{doc.metadata['source']}|{doc.metadata['page']}|{n}|{doc.page_content}"
    return hashlib.sha1(key.encode()).hexdigest()


def ingest(
    force: bool = False,
    max_pages: int | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    batch_size: int = 128,
) -> int:
    """Load, chunk, embed and upsert the PDFs. Skips the work if the namespace
    already has vectors, unless `force` is set. Returns the chunk count."""
    index = ensure_index()
    existing = namespace_count(index, namespace)
    if existing and not force:
        print(f"[ingest] namespace '{namespace}' already has {existing} vectors; "
              "skipping (use --force to rebuild).")
        return existing
    if existing and force:
        print(f"[ingest] --force: clearing namespace '{namespace}'")
        index.delete(delete_all=True, namespace=namespace)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(load_pdfs(max_pages))
    ids = [chunk_id(c, n) for n, c in enumerate(chunks)]
    print(f"[ingest] {len(chunks)} chunks -> namespace '{namespace}'")

    store = PineconeVectorStore(
        index=index, embedding=NomicEmbeddings(), namespace=namespace
    )
    for start in range(0, len(chunks), batch_size):
        store.add_documents(
            chunks[start : start + batch_size], ids=ids[start : start + batch_size]
        )
        print(f"[ingest] upserted {min(start + batch_size, len(chunks))}/{len(chunks)}")
    return len(chunks)


# =============================================================================
# 5. STATE
# =============================================================================


class CRAGState(TypedDict, total=False):
    question: str
    docs: list[Document]  # retrieved from Pinecone
    doc_scores: list[dict]  # per-chunk evaluator output (for the UI / CLI)
    good_docs: list[Document]  # chunks that passed the evaluator
    verdict: str  # CORRECT | INCORRECT | AMBIGUOUS
    reason: str  # why that verdict
    answer: str


# =============================================================================
# 6. NODES
# =============================================================================


def doc_label(doc: Document) -> str:
    """Human-readable citation for a Document: 'file.pdf p.12' or a URL."""
    md = doc.metadata
    if md.get("url"):
        return md["url"]
    page = md.get("page")
    page = int(page) if isinstance(page, (int, float)) else "?"  # Pinecone returns floats
    return f"{md.get('source', '?')} p.{page}"


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{doc_label(d)}]\n{d.page_content}" for d in docs)


def retrieve(state: CRAGState) -> CRAGState:
    """Stage 1: top-k similarity search in Pinecone."""
    docs = get_vectorstore().similarity_search(state["question"], k=TOP_K)
    return {"docs": docs}


# --- Stage 3: retrieval evaluator --------------------------------------------


class DocEvalScore(BaseModel):
    """Evaluator output for one retrieved chunk."""

    score: float = Field(description="Relevance of the chunk to the question, from 0.0 to 1.0")
    reason: str = Field(description="One short sentence justifying the score")


DOC_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict retrieval evaluator for a RAG system. Given a question "
     "and ONE retrieved chunk, rate how useful the chunk is for answering the "
     "question with a score from 0.0 to 1.0.\n"
     "1.0 = the chunk directly answers the question or contains the key facts.\n"
     "0.5 = on the right topic but only partially helpful.\n"
     "0.0 = unrelated to the question.\n"
     "Judge only what is in the chunk; do not use outside knowledge."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])


def eval_each_doc(state: CRAGState) -> CRAGState:
    """Score every retrieved chunk, then decide the verdict:
    CORRECT (any > UPPER_TH), INCORRECT (all < LOWER_TH), else AMBIGUOUS."""
    docs = state.get("docs", [])
    evaluator = DOC_EVAL_PROMPT | structured_llm(DocEvalScore)
    results: list[DocEvalScore] = evaluator.batch(
        [{"question": state["question"], "chunk": d.page_content} for d in docs],
        config={"max_concurrency": 4},
    )
    scores = [min(max(r.score, 0.0), 1.0) for r in results]
    doc_scores = [
        {"source": doc_label(d), "score": s, "reason": r.reason}
        for d, s, r in zip(docs, scores, results)
    ]

    if any(s > UPPER_TH for s in scores):
        verdict = "CORRECT"
        good = [d for d, s in zip(docs, scores) if s > UPPER_TH]
        reason = f"{len(good)} chunk(s) scored above {UPPER_TH}."
    elif all(s < LOWER_TH for s in scores):  # also true when nothing was retrieved
        verdict, good = "INCORRECT", []
        reason = f"All chunks scored below {LOWER_TH}."
    else:
        # As in CampusX stage 6: keep the partially relevant chunks (> LOWER_TH).
        verdict = "AMBIGUOUS"
        good = [d for d, s in zip(docs, scores) if s > LOWER_TH]
        reason = f"No chunk above {UPPER_TH}, but {len(good)} above {LOWER_TH}."
    return {"doc_scores": doc_scores, "good_docs": good, "verdict": verdict, "reason": reason}


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer ONLY from the context below. "
     "If the context is insufficient, say so plainly instead of guessing. "
     "Cite sources inline using the bracketed labels, e.g. [ida.pdf p.12]."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def generate(state: CRAGState) -> CRAGState:
    """Answer from the evaluator-approved chunks only."""
    context = format_docs(state.get("good_docs", []))
    msg = (ANSWER_PROMPT | get_llm()).invoke(
        {"question": state["question"], "context": context or "(no context)"}
    )
    return {"answer": msg.text}


# =============================================================================
# 7. GRAPH
# =============================================================================


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("eval_each_doc", eval_each_doc)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_edge("eval_each_doc", "generate")
    g.add_edge("generate", END)
    return g.compile()


def run_crag(question: str) -> dict:
    """Run the graph and return what the CLI / UI need."""
    state = build_graph().invoke({"question": question})
    return {
        "answer": state.get("answer", ""),
        "verdict": state.get("verdict", ""),
        "reason": state.get("reason", ""),
        "doc_scores": state.get("doc_scores", []),
        "sources": sorted({doc_label(d) for d in state.get("good_docs", [])}),
    }


def print_result(result: dict) -> None:
    """Pretty-print a run_crag() result for the terminal."""
    print(f"\n=== VERDICT: {result['verdict']} === {result['reason']}")
    for ds in result["doc_scores"]:
        print(f"  {ds['score']:.2f}  {ds['source']:<22} {ds['reason']}")
    print("\n=== ANSWER ===\n" + result["answer"])
    print("\nSources: " + (", ".join(result["sources"]) or "(none)"))


# =============================================================================
# 8. CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrective RAG (CRAG)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Load, chunk, embed and upsert the PDFs")
    p_ing.add_argument("--force", action="store_true", help="Rebuild the namespace")
    p_ing.add_argument("--max-pages", type=int, default=None,
                       help="Only the first N pages of each PDF (smoke tests)")
    p_ing.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest(force=args.force, max_pages=args.max_pages, namespace=args.namespace)
    elif args.cmd == "ask":
        print_result(run_crag(args.question))


if __name__ == "__main__":
    main()
