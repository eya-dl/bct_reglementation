# moteur_hybride.py
import re
import unicodedata
import pickle
import os
import chromadb
from sentence_transformers import SentenceTransformer
from chunk import chunk_all_documents
from pathlib import Path
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

BGE_PREFIX   = "Represent this sentence for searching relevant passages: "
PICKLE_PATH  = "bm25_store.pkl"
CHROMA_PATH  = "./chroma_db"
COLLECTION   = "reglementation_bct"
MD_FOLDER    = Path("pdf_chunks")

# RRF
RRF_K        = 60     # constante RRF standard (60 recommandé)
N_RESULTS    = 5      # résultats finaux
N_CANDIDATES = 20     # candidats récupérés par chaque moteur avant fusion

# Boosts BM25
BOOST_NUMBER_MATCH  = 6.0
BOOST_TITLE_MATCH   = 2.5
BOOST_NUMBER_IN_TEXT = 3.0

# ── Stopwords ─────────────────────────────────────────────────────────────────

STOPWORDS_FR = {
    "le","la","les","de","du","des","un","une","et","ou","en","dans",
    "pour","par","sur","avec","sans","ce","cette","ces","cet","qui","que",
    "au","aux","d","l","n","ne","pas","plus","moins","etre","est","sont",
    "a","dont","comme","donc","il","elle","ils","elles","leur","leurs",
    "tout","toute","tous","toutes","afin","ainsi","lors","doit","doivent",
    "peut","peuvent","soit","sa","son","ses","se","s","y","on","ni","etant",
    "ete","fait","faite","selon","entre","vers","chez","depuis","avant",
    "apres","sous","celui","celle","ceux","celles"
}

NUM_PATTERN = re.compile(r"\d{1,4}[-/]\d{1,4}|\d+")

# ── Normalisation ─────────────────────────────────────────────────────────────

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
        sep   = "-" if "-" in num else "/"
        parts = num.split(sep)
        variants.add("".join(parts))
        variants.update(parts)
    return variants

# ── Texte BM25 ────────────────────────────────────────────────────────────────

def build_bm25_text(c) -> str:
    parts = [
        c.document_type  or "",
        c.document_number or "",
        c.article_title  or "",
        c.breadcrumb     or "",
        c.text           or "",
    ]
    extra = []
    if c.document_number:
        for num in extract_numbers(c.document_number):
            extra.extend(number_variants(num))

    return " ".join(parts) + (" " + " ".join(extra) if extra else "")

# ── Boost BM25 ────────────────────────────────────────────────────────────────

def compute_boost(q_tokens: list, q_nums: set, chunk) -> float:
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

# ── Chargement BM25 ───────────────────────────────────────────────────────────

def load_or_build_bm25(children):
    from rank_bm25 import BM25Okapi

    if os.path.exists(PICKLE_PATH):
        print("Chargement BM25 depuis cache...")
        with open(PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["children"]

    print("Construction BM25...")
    corpus    = [build_bm25_text(c) for c in children]
    tokenized = [tokenize(t) for t in corpus]
    bm25      = BM25Okapi(tokenized, k1=1.5, b=0.5)

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "children": children}, f)

    print(f"BM25 construit sur {len(children)} chunks")
    return bm25, children

# ── Chargement modèle + ChromaDB ──────────────────────────────────────────────

def load_resources():
    print("Chargement BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")

    print("Connexion ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)
    print(f"Collection : {collection.count()} chunks")

    print("Chargement chunks...")
    _, children = chunk_all_documents(MD_FOLDER)

    bm25, children = load_or_build_bm25(children)

    return model, collection, bm25, children

# ── RRF (Reciprocal Rank Fusion) ──────────────────────────────────────────────

def rrf_score(rank: int, k: int = RRF_K) -> float:
    """Score RRF : 1 / (k + rank)  — rank commence à 1."""
    return 1.0 / (k + rank)

def reciprocal_rank_fusion(
    sem_results:  list[dict],   # [{"chunk_id": ..., "score": ..., "chunk": ...}, ...]
    bm25_results: list[dict],
    n_final:      int = N_RESULTS,
) -> list[dict]:
    """
    Fusionne deux listes de résultats via RRF.
    Chaque liste est triée par score décroissant.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunks_map: dict[str, object] = {}

    # ── Contribution BM25 ─────────────────────────────────────────────────────
    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        rrf_scores[cid] += rrf_score(rank)
        chunks_map[cid]  = r["chunk"]

    # ── Contribution embedding ────────────────────────────────────────────────
    for rank, r in enumerate(sem_results, start=1):
        cid = r["chunk_id"]
        rrf_scores[cid] += rrf_score(rank)
        if cid not in chunks_map:
            chunks_map[cid] = r["chunk"]

    # ── Tri final ─────────────────────────────────────────────────────────────
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    return [
        {
            "chunk_id":  cid,
            "rrf_score": round(rrf_scores[cid], 6),
            "chunk":     chunks_map[cid],
        }
        for cid in sorted_ids[:n_final]
    ]

# ── Recherche sémantique ──────────────────────────────────────────────────────

def semantic_search(
    query: str,
    model,
    collection,
    n: int = N_CANDIDATES,
) -> list[dict]:

    emb = model.encode(
        BGE_PREFIX + query,
        normalize_embeddings=True
    )

    res = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    results = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        results.append({
            "chunk_id": meta.get("chunk_id", ""),
            "score":    round(1 - dist, 4),
            "chunk":    _meta_to_chunk(doc, meta),
        })

    return results   # déjà trié par score desc

# ── Recherche BM25 ────────────────────────────────────────────────────────────

def bm25_search(
    query: str,
    bm25,
    children: list,
    n: int = N_CANDIDATES,
) -> list[dict]:

    q_tokens = tokenize(query)
    q_nums   = extract_numbers(query)
    scores   = bm25.get_scores(q_tokens)

    results = []
    for idx, chunk in enumerate(children):
        raw   = float(scores[idx])
        boost = compute_boost(q_tokens, q_nums, chunk)
        final = raw + boost
        results.append({
            "chunk_id": chunk.chunk_id,
            "score":    round(final, 4),
            "raw":      round(raw, 4),
            "boost":    round(boost, 4),
            "chunk":    chunk,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]

# ── Objet Chunk minimal depuis métadonnées ChromaDB ───────────────────────────

class SimpleChunk:
    """Wrapper léger pour les résultats ChromaDB."""
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

# ── Recherche hybride principale ──────────────────────────────────────────────

def hybrid_search(
    query:      str,
    model,
    collection,
    bm25,
    children:   list,
    n_final:    int = N_RESULTS,
    n_cand:     int = N_CANDIDATES,
) -> list[dict]:

    sem_results  = semantic_search(query, model, collection, n=n_cand)
    bm25_results = bm25_search(query, bm25, children, n=n_cand)

    return reciprocal_rank_fusion(sem_results, bm25_results, n_final=n_final)

# ── Affichage ─────────────────────────────────────────────────────────────────

def afficher(question: str, results: list[dict]):
    print("\n" + "╔" + "═" * 78 + "╗")
    print(f"║  QUESTION : {question[:64]:<64}║")
    print("╚" + "═" * 78 + "╝")

    for i, r in enumerate(results, 1):
        c     = r["chunk"]
        score = r["rrf_score"]

        print(f"\n┌─ Résultat {i}  rrf={score:.5f} " + "─" * 40)
        print(f"│  Type       : {c.document_type} {c.document_number}")
        print(f"│  Breadcrumb : {c.breadcrumb[:90]}")
        if c.article_title:
            print(f"│  Article    : {c.article_title[:80]}")
        print(f"│  Niveau     : {c.level} ({c.chunk_type})")
        print("│")
        contenu = c.text.strip().replace("\n", " ")
        print(f"│  {contenu[:500]}")
        print("└" + "─" * 78)

# ── Interface interactive ─────────────────────────────────────────────────────

def main():
    model, collection, bm25, children = load_resources()

    print("\n" + "━" * 80)
    print("  MOTEUR HYBRIDE — BGE-M3 + BM25 + RRF")
    print("  Tape 'quit' pour quitter")
    print("━" * 80)

    while True:
        try:
            question = input("\n❓ Question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        results = hybrid_search(question, model, collection, bm25, children)
        afficher(question, results)

        choix = input("\n  Chunk complet ? (numéro ou Entrée) : ").strip()
        if choix.isdigit():
            idx = int(choix) - 1
            if 0 <= idx < len(results):
                print("\n" + "═" * 80)
                print(results[idx]["chunk"].text)
                print("═" * 80)

if __name__ == "__main__":
    main()