import streamlit as st
import time
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from cross_encoder import load_resources, hybrid_rag

# CONFIG
st.set_page_config(
    page_title="Assistant Réglementaire · BCT",
    page_icon="🏛️",
    layout="wide"
)



st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --navy: #0B2545;
    --navy-light: #15396B;
    --gold: #9C7A35;
    --gold-light: #C9A646;
    --paper: #F7F5F0;
    --paper-dark: #EFEBE2;
    --ink: #1C1C1C;
    --line: #DAD4C4;
    --muted: #6B6457;
}

/* Fond général */
.stApp {
    background-color: var(--paper);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

/* Masquer le bandeau Streamlit par défaut pour un rendu plus institutionnel */
header[data-testid="stHeader"] {
    background-color: transparent;
}

/* ---------------------------------------------------------------------- */
/* MASTHEAD                                                                */
/* ---------------------------------------------------------------------- */

.bct-masthead {
    background-color: var(--navy);
    border-bottom: 3px solid var(--gold);
    border-radius: 6px;
    padding: 28px 32px 22px 32px;
    margin-bottom: 4px;
}

.bct-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold-light);
    margin-bottom: 6px;
}

.bct-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.15;
}

.bct-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: #C9CFDA;
    margin-top: 8px;
}

.bct-pillars {
    display: inline-flex;
    gap: 4px;
    margin-left: 2px;
    vertical-align: middle;
}
.bct-pillars span {
    width: 3px;
    height: 16px;
    background-color: var(--gold);
    display: inline-block;
}

/* ---------------------------------------------------------------------- */
/* SIDEBAR                                                                 */
/* ---------------------------------------------------------------------- */

[data-testid="stSidebar"] {
    background-color: var(--navy);
}

[data-testid="stSidebar"] * {
    color: #E8E9ED;
}

.bct-sidebar-brand {
    padding: 4px 0 14px 0;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 14px;
}

.bct-sidebar-brand .bct-pillars span {
    height: 14px;
}

.bct-sidebar-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 1.15rem;
    color: #FFFFFF;
}

.bct-sidebar-caption {
    font-size: 0.78rem;
    color: #AEB6C4;
    margin-top: 4px;
}

.bct-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gold-light);
    margin: 16px 0 6px 0;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}

/* Widgets dans la sidebar */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: var(--navy-light);
    border-color: rgba(255,255,255,0.25);
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--gold-light);
    font-family: 'IBM Plex Mono', monospace;
}

[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #AEB6C4;
}

[data-testid="stSidebar"] div.stButton > button {
    background-color: transparent;
    border: 1px solid var(--gold);
    color: var(--gold-light);
    font-weight: 500;
}
[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: var(--gold);
    color: var(--navy);
    border-color: var(--gold);
}

/* ---------------------------------------------------------------------- */
/* MESSAGES DE CHAT                                                       */
/* ---------------------------------------------------------------------- */

[data-testid="stChatMessage"] {
    background-color: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 4px 6px;
}

/* ---------------------------------------------------------------------- */
/* PASTILLES DE TEMPS DE TRAITEMENT                                       */
/* ---------------------------------------------------------------------- */

.timing-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    background-color: var(--paper-dark);
    border: 1px solid var(--line);
    color: var(--muted);
    border-radius: 4px;
    padding: 3px 9px;
    margin-right: 6px;
    margin-bottom: 4px;
}
.timing-pill b {
    color: var(--navy);
}

/* ---------------------------------------------------------------------- */
/* FICHES SOURCES (citations réglementaires)                              */
/* ---------------------------------------------------------------------- */

.source-card {
    background-color: #FFFFFF;
    border: 1px solid var(--line);
    border-left: 4px solid var(--gold);
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.source-ref {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 1.0rem;
    color: var(--navy);
}

.source-article {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 2px;
    margin-bottom: 10px;
}

.source-text {
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--ink);
}

.source-scores {
    margin-top: 10px;
    display: flex;
    gap: 8px;
}

.source-scores span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    background-color: var(--paper-dark);
    border-radius: 4px;
    padding: 2px 7px;
}

/* ---------------------------------------------------------------------- */
/* PIED DE PAGE                                                           */
/* ---------------------------------------------------------------------- */

.bct-footer {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    font-size: 0.76rem;
    color: var(--muted);
    font-style: italic;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATACLASS (inchangée)
# ----------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    document_number: str
    document_title: str
    article: Optional[str]
    rerank_score: float
    rrf_score: float


# ----------------------------------------------------------------------------
# CHARGEMENT UNIQUE (inchangé)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# PIPELINE (inchangé)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# AFFICHAGE (helpers purement présentationnels — aucune logique métier)
# ----------------------------------------------------------------------------

def render_timings(timings: dict):
    """Affiche les temps de traitement sous forme de pastilles."""

    if not timings:
        return

    pills = "".join(
        f'<span class="timing-pill">{k} <b>{v:.2f}s</b></span>'
        for k, v in timings.items()
    )

    st.markdown(pills, unsafe_allow_html=True)


def render_sources(chunks):
    """Affiche les sources sous forme de fiches de citation réglementaire."""

    if not chunks:
        return

    with st.expander(f"📄 {len(chunks)} source(s) citée(s)"):

        for chunk in chunks:

            article_line = f"Art. {chunk.article}" if chunk.article else "Référence non numérotée"

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-ref">{chunk.document_title} n° {chunk.document_number}</div>
                    <div class="source-article">{article_line}</div>
                    <div class="source-text">{chunk.text}</div>
                    <div class="source-scores">
                        <span>pertinence (rerank) : {chunk.rerank_score:.3f}</span>
                        <span>fusion RRF : {chunk.rrf_score:.5f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------

with st.sidebar:
    
    st.markdown(
        """
        <div class="bct-sidebar-brand">
            <span class="bct-pillars">
                <span></span><span></span><span></span><span></span>
            </span>
            <div class="bct-sidebar-title">Assistant Réglementaire</div>
            <div class="bct-sidebar-caption">Banque Centrale de Tunisie</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="bct-section-label">Modèle de génération</div>', unsafe_allow_html=True)

    llm_provider = st.selectbox(
        "LLM",
        [
            "Groq · Llama 3.3 70B",
            "Gemini Flash"
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div class="bct-section-label">Affichage</div>', unsafe_allow_html=True)

    show_sources = st.toggle(
        "Afficher les sources",
        value=True
    )

    show_timings = st.toggle(
        "Afficher les temps de traitement",
        value=True
    )

    debug = st.toggle(
        "Mode débogage",
        value=False
    )

    st.markdown('<div class="bct-section-label">Indexation</div>', unsafe_allow_html=True)

    st.metric(
        "Chunks indexés",
        resources["collection"].count()
    )

    st.markdown('<div class="bct-section-label">Session</div>', unsafe_allow_html=True)

    if st.button(
        "Réinitialiser la conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div class="bct-footer" style="border-top-color: rgba(255,255,255,0.15); color:#AEB6C4;">
        Pipeline : BM25 + BGE-M3 → RRF → CrossEncoder → LLM
        </div>
        """,
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------------
# SESSION (inchangée)
# ----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------------------------------------------------
# EN-TÊTE
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="bct-masthead">
        <div class="bct-eyebrow">Banque Centrale de Tunisie</div>
        <div class="bct-title">
            Assistant Réglementaire des Changes
            <span class="bct-pillars"><span></span><span></span><span></span><span></span></span>
        </div>
        <div class="bct-subtitle">
            Recherche et analyse documentaire sur le corpus réglementaire (1975–2024)
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ----------------------------------------------------------------------------
# HISTORIQUE
# ----------------------------------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant":

            if show_timings:
                render_timings(msg.get("timings"))

            if show_sources:
                render_sources(msg.get("chunks"))


# ----------------------------------------------------------------------------
# QUESTION
# ----------------------------------------------------------------------------

question = st.chat_input(
    "Posez votre question sur la réglementation des changes..."
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

        if show_timings:
            render_timings(result["timings"])

        if show_sources:
            render_sources(result["chunks"])

        if debug:

            st.subheader("Débogage")

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


# ----------------------------------------------------------------------------
# PIED DE PAGE
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="bct-footer">
    Les réponses générées par cet assistant sont fournies à titre indicatif et ne se substituent
    pas aux textes officiels publiés par la Banque Centrale de Tunisie et au Journal Officiel de
    la République Tunisienne (JORT).
    </div>
    """,
    unsafe_allow_html=True
)