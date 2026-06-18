
from json import encoder

import streamlit as st
import time
from dataclasses import dataclass
from typing import Optional

from cross_encoder import load_resources, hybrid_rag


# CONFIG

st.set_page_config(
    page_title="Assistant IA BCT",
    page_icon="🏦",
    layout="wide"
)


# STYLE

st.markdown("""
<style>

.source-card {
    background-color: rgba(120,120,120,0.08);
    border-left: 4px solid #1f6f54;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
}

.source-meta {
    font-size: 0.85em;
    opacity: 0.75;
    margin-bottom: 8px;
}

.timing-pill {
    display: inline-block;
    background-color: rgba(31,111,84,0.15);
    border-radius: 12px;
    padding: 4px 10px;
    margin-right: 6px;
    margin-bottom: 4px;
}

</style>
""", unsafe_allow_html=True)



# DATACLASS


@dataclass
class RetrievedChunk:
    text: str
    document_number: str
    document_title: str
    article: Optional[str]
    rerank_score: float
    rrf_score: float



# CHARGEMENT UNIQUE

@st.cache_resource(show_spinner="Chargement du moteur RAG...")
def get_resources():

    model, reranker, collection, bm25, children = load_resources()

    return {
        "model": model,
        "reranker": reranker,
        "collection": collection,
        "bm25": bm25,
        "children": children
    }


resources = get_resources()


# PIPELINE

def run_pipeline(query, llm_provider, resources):

    provider = "groq"

    if "Gemini" in llm_provider:
        provider = "gemini"

    result = hybrid_rag(
        query=query,
        model=resources["model"],
        reranker=resources["reranker"],
        collection=resources["collection"],
        bm25=resources["bm25"],
        children=resources["children"],
        provider=provider
    )

    chunks = []

    for source in result["sources"]:

        chunks.append(
            RetrievedChunk(
                text=source["extrait"],
                document_number=source["number"],
                document_title=source["type"],
                article=source["article"],
                rerank_score=source["rerank_score"],
                rrf_score=source["rrf_score"]
            )
        )

    return {
        "answer": result["answer"],
        "chunks": chunks,
        "timings": result.get("timings", {}),
        "raw_sources": result["sources"]
    }



# SIDEBAR
 

with st.sidebar:

    st.title("🏦 Assistant BCT")

    st.caption(
        "Assistant IA de compréhension de la réglementation des changes"
    )

    st.divider()

    llm_provider = st.selectbox(
        "LLM",
        [
            "Groq · Llama 3.3 70B",
            "Gemini Flash"
        ]
    )

    show_sources = st.toggle(
        "Afficher les sources",
        value=True
    )

    show_timings = st.toggle(
        "Afficher les temps",
        value=True
    )

    debug = st.toggle(
        "Mode Debug",
        value=False
    )

    st.divider()

    st.metric(
        "Chunks indexés",
        resources["collection"].count()
    )

    st.divider()

    if st.button(
        " Réinitialiser",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Pipeline : BM25 + BGE-M3 → RRF → "
        "CrossEncoder → LLM"
    )


# SESSION

if "messages" not in st.session_state:
    st.session_state.messages = []


# HEADER


st.title("🏦 Assistant IA - Réglementation des Changes")

st.caption(
    "Banque Centrale de Tunisie"
)

st.divider()



# HISTORIQUE


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant":

            if show_timings and msg.get("timings"):

                pills = "".join(
                    f'<span class="timing-pill">{k}: {v:.2f}s</span>'
                    for k, v in msg["timings"].items()
                )

                st.markdown(
                    pills,
                    unsafe_allow_html=True
                )

            if show_sources and msg.get("chunks"):

                with st.expander(
                    f" {len(msg['chunks'])} source(s)"
                ):

                    for chunk in msg["chunks"]:

                        st.markdown(
                            f"""
                            <div class="source-card">

                            <div class="source-meta">
                             {chunk.document_title}
                            · n°{chunk.document_number}
                            · {chunk.article}
                            · rerank={chunk.rerank_score:.3f}
                            · rrf={chunk.rrf_score:.5f}
                            </div>

                            {chunk.text}

                            </div>
                            """,
                            unsafe_allow_html=True
                        )



# QUESTION


question = st.chat_input(
    "Posez votre question..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        placeholder.markdown(
            " Recherche dans la réglementation..."
        )

        result = run_pipeline(
            query=question,
            llm_provider=llm_provider,
            resources=resources
        )

        placeholder.markdown(
            result["answer"]
        )

        if show_timings and result["timings"]:

            pills = "".join(
                f'<span class="timing-pill">{k}: {v:.2f}s</span>'
                for k, v in result["timings"].items()
            )

            st.markdown(
                pills,
                unsafe_allow_html=True
            )

        if show_sources and result["chunks"]:

            with st.expander(
                f" {len(result['chunks'])} source(s)"
            ):

                for chunk in result["chunks"]:

                    st.markdown(
                        f"""
                        <div class="source-card">

                        <div class="source-meta">
                         {chunk.document_title}
                        · n°{chunk.document_number}
                        · {chunk.article}
                        · rerank={chunk.rerank_score:.3f}
                        · rrf={chunk.rrf_score:.5f}
                        </div>

                        {chunk.text}

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        if debug:

            st.subheader("Debug")

            st.json(
                result["raw_sources"]
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "chunks": result["chunks"],
            "timings": result["timings"]
        }
    )


