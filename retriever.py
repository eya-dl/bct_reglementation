"""
retriever.py — Moteur de recherche hybride BCT
===============================================
Encapsule tout le pipeline de récupération :
  1. Chargement des ressources (BGE-M3, CrossEncoder, ChromaDB, BM25)
  2. Recherche sémantique (ChromaDB + BGE-M3)
  3. Recherche lexicale (BM25Okapi + boosts)
  4. Fusion RRF (Reciprocal Rank Fusion)
  5. Reranking cross-encoder (BAAI/bge-reranker-v2-m3)
  6. Formatage du contexte pour le LLM

Réutilise intégralement la logique de moteur_hybride.py.
"""

import os
import re
import pickle
import unicodedata
from collections import defaultdict
from pathlib import Path
from config import CROSS_REF_PATTERNS

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

from config import (
    BGE_PREFIX, PICKLE_PATH, CHROMA_PATH, COLLECTION, MD_FOLDER,
    RRF_K, N_CANDIDATES, N_RERANK, N_RESULTS,
    BOOST_NUMBER_MATCH, BOOST_TITLE_MATCH, BOOST_NUMBER_IN_TEXT,
    EMBED_MODEL, RERANKER_MODEL, RERANKER_MAX_LEN, STOPWORDS_FR,BOOST_PRIOR_DOCUMENT,
)

# Import du chunker existant
from chunk import chunk_all_documents

# ── Regex ─────────────────────────────────────────────────────────────────────
NUM_PATTERN = re.compile(r"\d{1,4}[-/]\d{1,4}|\d+")


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALISATION & TOKENISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize(text))
    return [t for t in tokens if t not in STOPWORDS_FR]


def extract_numbers(text: str) -> set[str]:
    return set(NUM_PATTERN.findall(text.lower()))


def number_variants(num: str) -> set[str]:
    variants = {num}
    if "-" in num or "/" in num:
        sep = "-" if "-" in num else "/"
        parts = num.split(sep)
        variants.add("".join(parts))
        variants.update(parts)
    return variants


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNK MINIMAL (pour les résultats ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleChunk:
    """Représentation légère d'un chunk récupéré depuis ChromaDB."""
    def __init__(self, text: str, meta: dict):
        self.text            = text
        self.chunk_id        = meta.get("chunk_id", "")
        self.document_type   = meta.get("document_type", "")
        self.document_number = meta.get("document_number", "")
        self.document_title  = meta.get("document_title", "")
        self.breadcrumb      = meta.get("breadcrumb", "")
        self.article_title   = meta.get("article_title", "")
        self.level           = meta.get("level", "")
        self.chunk_type      = meta.get("chunk_type", "")
        self.parent_id       = meta.get("parent_id", "")


def _meta_to_chunk(text: str, meta: dict) -> SimpleChunk:
    return SimpleChunk(text, meta)


# ══════════════════════════════════════════════════════════════════════════════
#  BM25 — CONSTRUCTION DU TEXTE ET BOOST
# ══════════════════════════════════════════════════════════════════════════════

def build_bm25_text(chunk) -> str:
    """Concatène les champs pertinents d'un chunk pour l'indexation BM25."""
    parts = [
        chunk.document_type   or "",
        chunk.document_number or "",
        chunk.article_title   or "",
        chunk.breadcrumb      or "",
        chunk.text            or "",
    ]
    extra = []
    if chunk.document_number:
        for num in extract_numbers(chunk.document_number):
            extra.extend(number_variants(num))
    return " ".join(parts) + (" " + " ".join(extra) if extra else "")


def compute_boost(q_tokens: list, q_nums: set, chunk) -> float:
    """Calcule le boost BM25 selon les correspondances de numéros et titres."""
    boost = 0.0
    if q_nums and chunk.document_number:
        c_variants = set()
        for n in extract_numbers(chunk.document_number):
            c_variants.update(number_variants(n))
        q_variants = set()
        for n in q_nums:
            q_variants.update(number_variants(n))
        if q_variants & c_variants:
            boost += BOOST_NUMBER_MATCH
        elif q_nums & extract_numbers(chunk.text or ""):
            boost += BOOST_NUMBER_IN_TEXT
    if chunk.article_title:
        if set(tokenize(chunk.article_title)) & set(q_tokens):
            boost += BOOST_TITLE_MATCH
    return boost


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES RESSOURCES
# ══════════════════════════════════════════════════════════════════════════════

def load_or_build_bm25(children: list):
    """Charge le BM25 depuis le cache pickle ou le reconstruit."""
    from rank_bm25 import BM25Okapi
    if os.path.exists(PICKLE_PATH):
        print("  Chargement BM25 depuis cache...")
        with open(PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["children"]
    print("  Construction BM25...")
    corpus    = [build_bm25_text(c) for c in children]
    tokenized = [tokenize(t) for t in corpus]
    bm25      = BM25Okapi(tokenized, k1=1.5, b=0.5)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "children": children}, f)
    print(f"  BM25 construit sur {len(children)} chunks")
    return bm25, children


def load_resources() -> tuple:
    """
    Charge toutes les ressources nécessaires au retrieval.
    À appeler une seule fois au démarrage.

    Retourne :
        (embed_model, reranker, collection, bm25, children)
    """
    print("  Chargement BGE-M3...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("  Chargement CrossEncoder (reranker)...")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=RERANKER_MAX_LEN)

    print("  Connexion ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)
    print(f"  Collection : {collection.count()} chunks")

    print("  Chargement chunks...")
    _, children = chunk_all_documents(MD_FOLDER)
    bm25, children = load_or_build_bm25(children)

    return embed_model, reranker, collection, bm25, children


#  RECHERCHE SÉMANTIQUE

def semantic_search(
    query:      str,
    model,
    collection,
    n:          int = N_CANDIDATES,
) -> list[dict]:
    """Recherche par embeddings BGE-M3 dans ChromaDB."""
    emb = model.encode(BGE_PREFIX + query, normalize_embeddings=True)
    res = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    results = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({
            "chunk_id": meta.get("chunk_id", ""),
            "score":    round(1 - dist, 4),
            "chunk":    _meta_to_chunk(doc, meta),
        })
    return results


#  RECHERCHE BM25

def bm25_search(
    query:    str,
    bm25,
    children: list,
    n:        int = N_CANDIDATES,
) -> list[dict]:
    """Recherche lexicale BM25 avec boosts sur numéros et titres d'articles."""
    q_tokens = tokenize(query)
    q_nums   = extract_numbers(query)
    scores   = bm25.get_scores(q_tokens)

    results = []
    for idx, chunk in enumerate(children):
        raw   = float(scores[idx])
        boost = compute_boost(q_tokens, q_nums, chunk)
        results.append({
            "chunk_id": chunk.chunk_id,
            "score":    round(raw + boost, 4),
            "raw":      round(raw, 4),
            "boost":    round(boost, 4),
            "chunk":    chunk,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]



#  FUSION RRF

def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    sem_results:  list[dict],
    bm25_results: list[dict],
    n_final:      int = N_RERANK,
) -> list[dict]:
    """
    Fusionne les résultats sémantiques et BM25 via Reciprocal Rank Fusion.
    Retourne les n_final meilleurs candidats pour le reranker.
    """
    rrf_scores: dict[str, float]  = defaultdict(float)
    chunks_map: dict[str, object] = {}

    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        rrf_scores[cid] += rrf_score(rank)
        chunks_map[cid]  = r["chunk"]

    for rank, r in enumerate(sem_results, start=1):
        cid = r["chunk_id"]
        rrf_scores[cid] += rrf_score(rank)
        if cid not in chunks_map:
            chunks_map[cid] = r["chunk"]

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    return [
        {
            "chunk_id":  cid,
            "rrf_score": round(rrf_scores[cid], 6),
            "chunk":     chunks_map[cid],
        }
        for cid in sorted_ids[:n_final]
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  RERANKING CROSS-ENCODER
# ══════════════════════════════════════════════════════════════════════════════

def rerank(
    query:      str,
    candidates: list[dict],
    reranker:   CrossEncoder,
    top_n:      int = N_RESULTS,
) -> list[dict]:
    """
    Re-score les candidats RRF avec le cross-encoder BGE.
    Retourne les top_n chunks triés par rerank_score décroissant.
    """
    if not candidates:
        return []
    pairs  = [(query, r["chunk"].text) for r in candidates]
    scores = reranker.predict(pairs)
    for r, s in zip(candidates, scores):
        r["rerank_score"] = round(float(s), 4)
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATAGE DU CONTEXTE
# ══════════════════════════════════════════════════════════════════════════════

def build_context(ranked: list[dict]) -> str:
    """Formate les chunks retenus en contexte lisible pour le LLM générateur."""
    parts = []
    for i, r in enumerate(ranked, 1):
        c   = r["chunk"]
        ref = " — ".join(filter(None, [
            c.document_type,
            c.document_number,
            c.article_title or c.breadcrumb,
        ]))
        parts.append(f"[Source {i}] {ref}\n{c.text.strip()}")
    return "\n\n" + ("\n\n---\n\n").join(parts) + "\n"




def apply_prior_document_boost(
    ranked:          list[dict],
    prior_doc_refs:  set[tuple[str, str]] | None,
    boost:           float = BOOST_PRIOR_DOCUMENT,
) -> list[dict]:
    """
    Booste le rerank_score des chunks appartenant aux mêmes documents
    (type, numéro) que ceux retenus pour une sous-question précédente
    dans une décomposition composite.

    Ne filtre rien : un chunk hors de prior_doc_refs peut toujours
    remonter en tête s'il a un score nettement supérieur — c'est un
    biais doux, pas une contrainte stricte.
    """
    if not prior_doc_refs:
        return ranked

    for r in ranked:
        c = r["chunk"]
        key = (c.document_type, c.document_number)
        if key in prior_doc_refs:
            r["rerank_score"] = round(r["rerank_score"] + boost, 4)
            r["prior_boost_applied"] = True
    
    
    print(f"   🐛 DEBUG prior_doc_refs = {prior_doc_refs}")

    return sorted(ranked, key=lambda x: x["rerank_score"], reverse=True)
# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE RETRIEVAL COMPLET (utilisé par nodes.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_retrieval(
    query:           str,
    model,
    reranker,
    collection,
    bm25,
    children:        list,
    prior_doc_refs:  set[tuple[str, str]] | None = None,
) -> tuple[list[dict], str]:
    """
    Exécute le pipeline complet :
      semantic_search + bm25_search → RRF → rerank → boost documentaire → build_context

    prior_doc_refs : ensemble de (document_type, document_number) à favoriser,
    utilisé pour les sous-questions dérivées d'une question composite —
    biaise le retrieval vers les mêmes textes que la sous-question précédente.

    Retourne :
        (ranked_chunks, context_str)
    """
    sem_results  = semantic_search(query, model, collection, n=N_CANDIDATES)
    bm25_results = bm25_search(query, bm25, children, n=N_CANDIDATES)
    fused        = reciprocal_rank_fusion(sem_results, bm25_results, n_final=N_RERANK)
    ranked       = rerank(query, fused, reranker, top_n=N_RERANK)  # top_n élargi avant boost
    ranked       = apply_prior_document_boost(ranked, prior_doc_refs)
    ranked       = ranked[:N_RESULTS]  # troncature finale après boost
    context      = build_context(ranked)
    return ranked, context

def format_sources(ranked: list[dict]) -> list[dict]:
    """
    Formate les chunks reranked en liste de sources pour la sortie finale.
    Compatible avec le format de hybrid_rag() de moteur_hybride.py.
    """
    return [
        {
            "rank":         i + 1,
            "chunk_id":     r["chunk_id"],
            "rrf_score":    r.get("rrf_score", 0),
            "rerank_score": r["rerank_score"],
            "type":         r["chunk"].document_type,
            "number":       r["chunk"].document_number,
            "article":      r["chunk"].article_title,
            "breadcrumb":   r["chunk"].breadcrumb,
            "extrait":      r["chunk"].text[:300],
        }
        for i, r in enumerate(ranked)
    ]


def extract_cross_references(chunks: list) -> list[dict]:
    """
    Analyse le texte des chunks retenus pour détecter des références
    explicites à d'autres articles/lois/circulaires.
    """
    references = []
    seen = set()

    for chunk in chunks:
        text = chunk.get("text", "") or chunk.get("document", "")

        for pattern in CROSS_REF_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                # Normalise selon le pattern matché (3 groupes vs 2 vs 1)
                if len(groups) == 3:
                    ref = {"article": groups[0], "type": groups[1].lower(), "number": groups[2]}
                elif len(groups) == 2:
                    ref = {"article": None, "type": groups[0].lower(), "number": groups[1]}
                else:
                    ref = {"article": groups[0], "type": None, "number": None}

                key = (ref["type"], ref["number"], ref["article"])
                if key not in seen:
                    seen.add(key)
                    references.append(ref)

    return references


def fetch_chunks_by_reference(references: list[dict], existing_ids: set, collection, top_k_per_ref: int = 2) -> list:
    """
    Cherche dans ChromaDB les chunks correspondant aux références détectées,
    en excluant ceux déjà présents dans le contexte actuel.
    """
    additional_chunks = []

    for ref in references:
        where_filter = {}
        if ref.get("number"):
            where_filter["number"] = ref["number"]
        if ref.get("article"):
            where_filter["article_ref"] = ref["article"]

        if not where_filter:
            continue

        try:
            results = collection.get(
                where=where_filter,
                limit=top_k_per_ref,
                include=["documents", "metadatas"]
            )
        except Exception as e:
            print(f"   ⚠️ Erreur lookup cross-référence {ref} : {e}")
            continue

        for doc, meta, chunk_id in zip(results.get("documents", []), results.get("metadatas", []), results.get("ids", [])):
            if chunk_id in existing_ids:
                continue
            additional_chunks.append({
                "text": doc,
                "metadata": meta,
                "id": chunk_id,
                "cross_referenced": True,  # marqueur pour l'affichage/debug
            })
            existing_ids.add(chunk_id)

    return additional_chunks