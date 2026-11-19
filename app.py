"""Minimal Streamlit UI for the CRAG pipeline in crag.py.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import crag

crag.setup_logging()

VERDICT_COLORS = {"CORRECT": "green", "AMBIGUOUS": "orange", "INCORRECT": "red"}

st.set_page_config(page_title="Corrective RAG", page_icon="📚")


@st.cache_resource(show_spinner="Connecting to Pinecone, Ollama and Gemini...")
def load_graph():
    """Build the compiled LangGraph once per server process and warm up the
    clients it uses (Pinecone vector store, Tavily)."""
    crag.get_vectorstore()
    crag.get_tavily()
    return crag.build_graph()


def show_trace(result: dict) -> None:
    """The 'How CRAG answered this' expander."""
    with st.expander("How CRAG answered this"):
        verdict = result["verdict"] or "UNKNOWN"
        st.badge(verdict, color=VERDICT_COLORS.get(verdict, "gray"))
        st.caption(result["reason"])

        st.markdown("**Retrieved chunk scores**")
        st.dataframe(
            [
                {"score": round(d["score"], 2), "chunk": d["source"], "reason": d["reason"]}
                for d in result["doc_scores"]
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            f"**Refinement:** kept {result['strips_kept']} of "
            f"{result['strips_total']} sentence strips"
        )
        if result["web_query"]:
            st.markdown(f"**Rewritten web query:** `{result['web_query']}`")
        st.markdown("**Sources**")
        for src in result["sources"] or ["(none)"]:
            st.markdown(f"- {src}")


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Corrective RAG")
    st.caption(
        f"LLM: `{crag.GEMINI_MODEL}`  \nEmbeddings: `{crag.EMBED_MODEL}`  \n"
        f"Index: `{crag.INDEX_NAME}` / `{crag.DEFAULT_NAMESPACE}`"
    )
    if st.button("Ingest PDFs", width="stretch"):
        with st.spinner("Loading, chunking and embedding PDFs..."):
            try:
                n = crag.ingest()  # idempotent: skips if the namespace is filled
                st.success(f"Namespace ready: {n} chunks.")
            except crag.CRAGError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Ingest failed: {e}")
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []

# --- Chat --------------------------------------------------------------------
st.title("📚 Corrective RAG")
st.caption("Ask about the PDFs in documents/. CRAG falls back to the web when they can't answer.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("result"):
            show_trace(msg["result"])

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving, grading and refining..."):
                result = crag.run_crag(question, graph=load_graph())
            st.markdown(result["answer"])
            show_trace(result)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "result": result}
            )
        except crag.CRAGError as e:  # missing key, Ollama down, Gemini overloaded...
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
