from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import re
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

# langchain-community is being sunset (no standalone PDF-loader package exists
# yet), so PyPDFLoader still comes from it; silence its import-time notice.
# Gemini 3.x Flash-Lite ignores temperature and warns on every call; we still
# pass temperature=0 for models that honour it, so hide the repeated notice.
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# TavilySearch (langchain-tavily) replaces the deprecated
# langchain_community.tools.TavilySearchResults used in the CampusX notebooks.
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field

log = logging.getLogger("crag")

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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM. Override with GEMINI_MODEL / GEMINI_FALLBACK_MODEL in .env if needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")
GEMINI_MAX_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "6"))

# Chunking / retrieval
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

# Retrieval-evaluator thresholds (CampusX stage 3)
UPPER_TH = 0.7  # any doc above this  -> CORRECT
LOWER_TH = 0.3  # all docs below this -> INCORRECT; otherwise AMBIGUOUS

# Refinement: how many sentence strips the LLM judges per call
STRIP_BATCH = 20
MIN_STRIP_CHARS = 20

# Web search
WEB_MAX_RESULTS = 5


class CRAGError(RuntimeError):
    """A problem the user can fix (missing key, Ollama down, ...).
    The CLI and the UI show its message without a traceback."""


def setup_logging(level: int = logging.INFO) -> None:
    """Short '[crag] ...' log lines; keep chatty HTTP libraries quiet."""
    logging.basicConfig(level=level, format="[%(name)s] %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "pinecone", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def require_env(name: str) -> str:
    """Return an env var or raise a clear error (never logs the value)."""
    value = os.getenv(name, "").strip()
    if not value:
        raise CRAGError(f"{name} is missing. Add it to .env (see .env.example).")
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
        max_retries=1,  # our own backoff below handles longer outages
        timeout=120,
    )


def get_llm():
    """Flash-Lite, falling back to Flash if the primary call errors."""
    return _chat_model(GEMINI_MODEL).with_fallbacks([_chat_model(GEMINI_FALLBACK_MODEL)])


def structured_llm(schema: type):
    """`llm.with_structured_output(schema)` with the same Flash fallback.
    (Structured output must be bound to each model before adding fallbacks.)"""
    return _chat_model(GEMINI_MODEL).with_structured_output(schema).with_fallbacks(
        [_chat_model(GEMINI_FALLBACK_MODEL).with_structured_output(schema)]
    )


# --- Retry with exponential backoff for rate limits / overload ---------------

_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "503", "UNAVAILABLE",
              "504", "DEADLINE_EXCEEDED", "timed out", "Timeout", "EmptyLLMOutput",
              "RemoteProtocolError", "ReadError", "ConnectError", "disconnected")
_BAD_KEY = ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "401")


class EmptyLLMOutput(Exception):
    """Structured output came back empty (treated as retryable)."""


def _is_retryable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(marker in text for marker in _RETRYABLE)


def _short(exc: BaseException) -> str:
    """First line of an API error, without the JSON payload."""
    return str(exc).split("{")[0].strip()[:100]


def llm_batch(chain, inputs: list[dict], max_concurrency: int = 4) -> list:
    """Run `chain` over many inputs with retry + exponential backoff.
    Only the failed items are retried, so successful calls are never repeated.
    Non-retryable errors (bad key, bad request) are raised immediately."""
    results: list = [None] * len(inputs)
    pending = list(range(len(inputs)))
    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        outs = chain.batch(
            [inputs[i] for i in pending],
            config={"max_concurrency": max_concurrency},
            return_exceptions=True,
        )
        failed: list[int] = []
        last_error: BaseException | None = None
        for i, out in zip(pending, outs):
            if out is None:
                out = EmptyLLMOutput("model returned no structured output")
            if isinstance(out, Exception):
                if any(m in str(out) for m in _BAD_KEY):
                    raise CRAGError("Gemini rejected GEMINI_API_KEY; check the key in .env.") from out
                if not _is_retryable(out):
                    raise out
                failed.append(i)
                last_error = out
            else:
                results[i] = out
        if not failed:
            return results
        if attempt == GEMINI_MAX_ATTEMPTS:
            raise CRAGError(
                f"Gemini still failing after {GEMINI_MAX_ATTEMPTS} attempts "
                f"({_short(last_error)}). The API may be overloaded; try again later."
            ) from last_error
        wait = min(60.0, 2.0**attempt) + random.uniform(0, 1)
        log.warning("gemini: %d call(s) failed (%s); retry %d/%d in %.0fs",
                    len(failed), _short(last_error), attempt, GEMINI_MAX_ATTEMPTS - 1, wait)
        time.sleep(wait)
        pending = failed
    return results


def llm_invoke(chain, inputs: dict):
    """Single call with the same retry/backoff as llm_batch."""
    return llm_batch(chain, [inputs], max_concurrency=1)[0]


# =============================================================================
# 2. EMBEDDINGS: nomic-embed-text with task prefixes
# =============================================================================


class NomicEmbeddings(Embeddings):
    """Wraps OllamaEmbeddings and adds the task prefixes nomic-embed-text
    was trained with: "search_document: " for chunks, "search_query: " for
    queries. Every code path goes through this class, so the prefixes are
    always consistent between ingest and retrieval."""

    def __init__(self, model: str = EMBED_MODEL) -> None:
        self._inner = OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)

    @staticmethod
    def _friendly(exc: Exception) -> CRAGError | None:
        """Translate common Ollama failures into actionable messages."""
        text = str(exc).lower()
        if "not found" in text and "model" in text:
            return CRAGError(f"Ollama model '{EMBED_MODEL}' is not available. "
                             f"Run: ollama pull {EMBED_MODEL}")
        if isinstance(exc, ConnectionError) or "connect" in text or "refused" in text:
            return CRAGError(f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
                             "Start it (open the Ollama app or run `ollama serve`).")
        return None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._inner.embed_documents([f"search_document: {t}" for t in texts])
        except Exception as e:
            raise (self._friendly(e) or e) from e

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._inner.embed_query(f"search_query: {text}")
        except Exception as e:
            raise (self._friendly(e) or e) from e


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
    try:
        exists = pc.has_index(INDEX_NAME)
    except Exception as e:
        if "401" in str(e) or "unauthorized" in str(e).lower():
            raise CRAGError("Pinecone rejected PINECONE_API_KEY; check the key in .env.") from e
        raise CRAGError(f"Could not reach Pinecone: {_short(e)}") from e
    if not exists:
        log.info("pinecone: creating index '%s' (%dd, %s)", INDEX_NAME, EMBED_DIM, METRIC)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    deadline = time.time() + timeout_s
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        if time.time() > deadline:
            raise CRAGError(f"Pinecone index '{INDEX_NAME}' is not ready after {timeout_s}s; "
                            "try again in a minute.")
        time.sleep(2)
    return pc.Index(INDEX_NAME)


@lru_cache(maxsize=4)
def get_vectorstore(namespace: str = DEFAULT_NAMESPACE) -> PineconeVectorStore:
    """Read-side vector store (cached: one index check per process)."""
    return PineconeVectorStore(index=ensure_index(), embedding=NomicEmbeddings(),
                               namespace=namespace)


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
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise CRAGError(f"No PDFs found in {DOCS_DIR}.")
    pages: list[Document] = []
    for pdf in pdfs:
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
        log.info("ingest: loaded %s", pdf.name)
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
        log.info("ingest: namespace '%s' already has %d vectors; skipping "
                 "(use --force to rebuild).", namespace, existing)
        return existing
    if existing and force:
        log.info("ingest: --force, clearing namespace '%s'", namespace)
        index.delete(delete_all=True, namespace=namespace)

    NomicEmbeddings().embed_query("ping")  # fail fast if Ollama is down
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(load_pdfs(max_pages))
    ids = [chunk_id(c, n) for n, c in enumerate(chunks)]
    log.info("ingest: %d chunks -> namespace '%s'", len(chunks), namespace)

    store = PineconeVectorStore(index=index, embedding=NomicEmbeddings(), namespace=namespace)
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        store.add_documents(chunks[start:end], ids=ids[start:end])
        log.info("ingest: upserted %d/%d", min(end, len(chunks)), len(chunks))
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
    web_query: str  # query sent to Tavily
    web_docs: list[Document]  # Tavily results as Documents (metadata: url)
    strips: list[dict]  # decomposed sentences: {"text", "source"}
    kept_strips: list[dict]  # strips the filter kept
    refined_context: str  # recomposed context passed to generate
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


# --- Stage 1: retrieve --------------------------------------------------------


def retrieve(state: CRAGState) -> CRAGState:
    """Top-k similarity search in Pinecone."""
    docs = get_vectorstore().similarity_search(state["question"], k=TOP_K)
    log.info("retrieve: %d chunks", len(docs))
    return {"docs": docs}


# --- Stage 3: retrieval evaluator --------------------------------------------


class DocEvalScore(BaseModel):
    """Evaluator output for one retrieved chunk."""

    score: float = Field(description="Relevance of the chunk to the question, from 0.0 to 1.0")
    reason: str = Field(description="One short sentence justifying the score")


DOC_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict retrieval evaluator for a RAG system. Given a question "
        "and ONE retrieved chunk, rate how useful the chunk is for answering the "
        "question with a score from 0.0 to 1.0.\n"
        "1.0 = the chunk directly answers the question or contains the key facts.\n"
        "0.5 = on the right topic but only partially helpful.\n"
        "0.0 = unrelated to the question.\n"
        "Judge only what is in the chunk; do not use outside knowledge."
    )),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])


def eval_each_doc(state: CRAGState) -> CRAGState:
    """Score every retrieved chunk, then decide the verdict:
    CORRECT (any > UPPER_TH), INCORRECT (all < LOWER_TH), else AMBIGUOUS."""
    docs = state.get("docs", [])
    evaluator = DOC_EVAL_PROMPT | structured_llm(DocEvalScore)
    results: list[DocEvalScore] = llm_batch(
        evaluator, [{"question": state["question"], "chunk": d.page_content} for d in docs]
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
    log.info("eval: scores=%s -> %s", [round(s, 2) for s in scores], verdict)
    return {"doc_scores": doc_scores, "good_docs": good, "verdict": verdict, "reason": reason}


# --- Stage 2: retrieval refinement (decompose -> filter -> recompose) ---------


class StripDecision(BaseModel):
    index: int = Field(description="The number shown in brackets before the strip")
    keep: bool = Field(description="True if the strip helps answer the question")


class StripFilter(BaseModel):
    """Keep/drop decision for each strip in one batch."""

    decisions: list[StripDecision]


STRIP_FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You filter context for a RAG system. For EACH numbered sentence strip, "
        "decide keep=true only if it contains information that helps answer the "
        "question (facts, definitions, steps, examples). Drop boilerplate, "
        "headers, references and off-topic text. Return one decision per strip."
    )),
    ("human", "Question: {question}\n\nStrips:\n{strips}"),
])


def decompose(docs: list[Document]) -> list[dict]:
    """Split documents into sentence-level strips, each tagged with its source."""
    strips: list[dict] = []
    for d in docs:
        text = re.sub(r"\s+", " ", d.page_content).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if len(sentence) >= MIN_STRIP_CHARS:
                strips.append({"text": sentence, "source": doc_label(d)})
    return strips


def filter_strips(question: str, strips: list[dict]) -> list[dict]:
    """Ask the LLM which strips to keep, STRIP_BATCH strips per call.
    Strips the model does not mention are dropped (conservative filter)."""
    if not strips:
        return []
    batches = [strips[i : i + STRIP_BATCH] for i in range(0, len(strips), STRIP_BATCH)]
    inputs = [
        {"question": question,
         "strips": "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(batch))}
        for batch in batches
    ]
    results: list[StripFilter] = llm_batch(STRIP_FILTER_PROMPT | structured_llm(StripFilter), inputs)
    kept: list[dict] = []
    for batch, result in zip(batches, results):
        keep_idx = {d.index for d in result.decisions if d.keep}
        kept.extend(s for i, s in enumerate(batch) if i in keep_idx)
    return kept


def recompose(strips: list[dict]) -> str:
    """Join kept strips back into text, grouping consecutive strips that share
    a source under one citation label."""
    parts: list[str] = []
    for s in strips:
        if parts and parts[-1].startswith(f"[{s['source']}]"):
            parts[-1] += " " + s["text"]
        else:
            parts.append(f"[{s['source']}] {s['text']}")
    return "\n\n".join(parts)


def refine(state: CRAGState) -> CRAGState:
    """Decompose -> filter -> recompose, separately for internal and web docs,
    then merge the two contexts (CampusX stage 6):
      CORRECT   -> internal good_docs only
      INCORRECT -> web_docs only
      AMBIGUOUS -> internal good_docs + web_docs, as two labelled sections
    Both sources are filtered in one llm_batch to save round trips."""
    internal = state.get("good_docs", []) if state.get("verdict") != "INCORRECT" else []
    web = state.get("web_docs", [])
    internal_strips, web_strips = decompose(internal), decompose(web)
    kept = filter_strips(state["question"], internal_strips + web_strips)

    web_sources = {doc_label(d) for d in web}
    kept_internal = [s for s in kept if s["source"] not in web_sources]
    kept_web = [s for s in kept if s["source"] in web_sources]
    sections = []
    if kept_internal:
        sections.append("### Internal knowledge (PDFs)\n" + recompose(kept_internal))
    if kept_web:
        sections.append("### Web knowledge\n" + recompose(kept_web))
    log.info("refine: kept %d/%d strips (internal %d, web %d)", len(kept),
             len(internal_strips) + len(web_strips), len(kept_internal), len(kept_web))
    return {
        "strips": internal_strips + web_strips,
        "kept_strips": kept,
        "refined_context": "\n\n".join(sections),
    }


# --- Stage 4: web search fallback ---------------------------------------------


@lru_cache(maxsize=1)
def get_tavily() -> TavilySearch:
    require_env("TAVILY_API_KEY")  # TavilySearch reads it from the environment
    return TavilySearch(max_results=WEB_MAX_RESULTS)


def web_search(state: CRAGState) -> CRAGState:
    """Search the web with Tavily and wrap each hit as a Document with its URL.
    A failed search degrades gracefully to 'no web results'."""
    query = state.get("web_query") or state["question"]
    try:
        response = get_tavily().invoke({"query": query})
    except CRAGError:
        raise
    except Exception as e:
        log.warning("web_search: Tavily failed (%s); continuing without web results", _short(e))
        response = {}
    if isinstance(response, dict) and response.get("error"):
        log.warning("web_search: Tavily error: %s", response["error"])
    results = response.get("results", []) if isinstance(response, dict) else []
    web_docs = [
        Document(
            page_content=f"{r.get('title', '')}\n{r.get('content', '')}".strip(),
            metadata={"url": r["url"], "title": r.get("title", "")},
        )
        for r in results
        if r.get("url")
    ]
    log.info("web_search: %d results for %r", len(web_docs), query)
    return {"web_query": query, "web_docs": web_docs}


# --- Stage 5: query rewriting for web search ----------------------------------


class WebQuery(BaseModel):
    """A search-engine-friendly rewrite of the user's question."""

    query: str = Field(description="Short keyword-style web search query (max ~10 words)")


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Rewrite the user's question into a short, keyword-style web search query. "
        "Keep names, versions, dates and technical terms; drop filler words. "
        "Add a recency hint such as 'latest' only if the question asks about "
        "recent events. Return only the query."
    )),
    ("human", "{question}"),
])


def rewrite_query(state: CRAGState) -> CRAGState:
    """Turn the question into a concise web search query."""
    result: WebQuery = llm_invoke(
        REWRITE_PROMPT | structured_llm(WebQuery), {"question": state["question"]}
    )
    query = result.query.strip() or state["question"]
    log.info("rewrite_query: %r", query)
    return {"web_query": query}


# --- Stage 6: routing (CORRECT / INCORRECT / AMBIGUOUS) + generation ----------


def route_after_eval(state: CRAGState) -> str:
    """Three-way CRAG routing after the evaluator:
    CORRECT -> refine internal docs; INCORRECT / AMBIGUOUS -> also search the web."""
    return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Answer ONLY from the context below; do not "
        "use outside knowledge. The context may contain an 'Internal knowledge "
        "(PDFs)' section and a 'Web knowledge' section: combine them, and if they "
        "disagree, say so. If the context is insufficient, say so plainly instead "
        "of guessing. Cite sources inline using the bracketed labels, e.g. "
        "[ida.pdf p.12] or [https://...], and end with a short 'Sources:' list."
    )),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def generate(state: CRAGState) -> CRAGState:
    """Answer from the refined context only."""
    context = state.get("refined_context", "").strip()
    msg = llm_invoke(
        ANSWER_PROMPT | get_llm(),
        {"question": state["question"], "context": context or "(no relevant context found)"},
    )
    return {"answer": msg.text}


# =============================================================================
# 7. GRAPH
# =============================================================================


@lru_cache(maxsize=1)
def build_graph():
    """retrieve -> eval_each_doc -> [CORRECT] refine -> generate
                                  -> [INCORRECT|AMBIGUOUS] rewrite_query -> web_search
                                     -> refine -> generate"""
    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("eval_each_doc", eval_each_doc)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_search", web_search)
    g.add_node("refine", refine)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges(
        "eval_each_doc", route_after_eval, {"rewrite_query": "rewrite_query", "refine": "refine"}
    )
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)
    return g.compile()


def run_crag(question: str, graph=None) -> dict:
    """Run the graph and return what the CLI / UI need.
    `graph` lets the Streamlit app pass in its cached compiled graph."""
    state = (graph or build_graph()).invoke({"question": question})
    return {
        "answer": state.get("answer", ""),
        "verdict": state.get("verdict", ""),
        "reason": state.get("reason", ""),
        "doc_scores": state.get("doc_scores", []),
        "web_query": state.get("web_query", ""),
        "strips_total": len(state.get("strips", [])),
        "strips_kept": len(state.get("kept_strips", [])),
        # Sources = where the kept strips came from, i.e. what the answer used.
        "sources": list(dict.fromkeys(s["source"] for s in state.get("kept_strips", []))),
    }


def print_result(result: dict) -> None:
    """Pretty-print a run_crag() result for the terminal."""
    print(f"\n=== VERDICT: {result['verdict']} === {result['reason']}")
    for ds in result["doc_scores"]:
        print(f"  {ds['score']:.2f}  {ds['source']:<22} {ds['reason']}")
    if result["web_query"]:
        print(f"\nWeb search query: {result['web_query']}")
    print(f"\nRefinement: kept {result['strips_kept']}/{result['strips_total']} sentence strips")
    print("\n=== ANSWER ===\n" + result["answer"])
    print("\nSources: " + (", ".join(result["sources"]) or "(none)"))


# --- Small evaluation: one question per verdict -------------------------------

EVAL_QUESTIONS: list[tuple[str, str]] = [
    # Clearly answerable from the books
    ("CORRECT", "What is a pandas DataFrame?"),
    # Unrelated current event: nothing in the PDFs
    ("INCORRECT", "Who won the 2026 FIFA World Cup final?"),
    # Borderline: the Python book mentions versions, but not the current one
    ("AMBIGUOUS", "What is the latest stable version of Python and when was it released?"),
]


def run_eval() -> bool:
    """Run EVAL_QUESTIONS through the graph and print expected vs actual
    verdicts. Returns True if every verdict matched."""
    rows = []
    for expected, question in EVAL_QUESTIONS:
        print(f"\n>>> [{expected}] {question}")
        result = run_crag(question)
        ok = result["verdict"] == expected
        rows.append((ok, expected, result["verdict"], question))
        print(f"    verdict: {result['verdict']} ({'match' if ok else 'MISMATCH'})"
              f"  scores: {[round(d['score'], 2) for d in result['doc_scores']]}")
        if result["web_query"]:
            print(f"    web query: {result['web_query']}")
        print(f"    answer: {result['answer'][:200].replace(chr(10), ' ')}...")

    print("\n=== EVAL SUMMARY ===")
    for ok, expected, actual, question in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  expected {expected:<9} got {actual:<9} {question}")
    passed = sum(ok for ok, *_ in rows)
    print(f"  {passed}/{len(rows)} verdicts matched")
    return passed == len(rows)


# =============================================================================
# 8. CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrective RAG (CRAG)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Load, chunk, embed and upsert the PDFs")
    p_ing.add_argument("--force", action="store_true", help="Rebuild the namespace")
    p_ing.add_argument("--max-pages", type=int, default=None,
                       help="Only the first N pages of each PDF (smoke tests)")
    p_ing.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")

    sub.add_parser("eval", help="Run 3 sample questions (CORRECT/INCORRECT/AMBIGUOUS)")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        if args.cmd == "ingest":
            ingest(force=args.force, max_pages=args.max_pages, namespace=args.namespace)
        elif args.cmd == "ask":
            print_result(run_crag(args.question))
        elif args.cmd == "eval":
            sys.exit(0 if run_eval() else 1)
    except CRAGError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
