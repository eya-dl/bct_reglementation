import re
import os
import unicodedata
import pickle
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from chunk import chunk_all_documents
from pathlib import Path
from collections import defaultdict
import os
from dotenv import load_dotenv
load_dotenv()
print(repr(os.getenv("GROQ_API_KEY")))  
#Configuration 

BGE_PREFIX   = "Represent this sentence for searching relevant passages: "
PICKLE_PATH  = "bm25_store.pkl"
CHROMA_PATH  = "./chroma_db"
COLLECTION   = "reglementation_bct"
MD_FOLDER    = Path("pdf_chunks")

# RRF
RRF_K        = 60
N_CANDIDATES = 20   # candidats par moteur avant fusion
N_RERANK     = 15   # chunks envoyés au cross-encoder après RRF
N_RESULTS    = 5    # résultats finaux retenus

# Boosts BM25
BOOST_NUMBER_MATCH   = 6.0
BOOST_TITLE_MATCH    = 2.5
BOOST_NUMBER_IN_TEXT = 3.0

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")   # "groq" | "gemini"
# Liste des clés Groq disponibles
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
]

# On enlève les clés vides
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL    = "gemini-1.5-flash"

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

# Texte BM25 

def build_bm25_text(c) -> str:
    parts = [
        c.document_type   or "",
        c.document_number or "",
        c.article_title   or "",
        c.breadcrumb      or "",
        c.text            or "",
    ]
    extra = []
    if c.document_number:
        for num in extract_numbers(c.document_number):
            extra.extend(number_variants(num))
    return " ".join(parts) + (" " + " ".join(extra) if extra else "")

# Boost BM25 

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

#Chargement BM25 

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

#  Chargement ressources 

def load_resources():
    print("Chargement BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")

    print("Chargement Cross-Encoder (reranker)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    print("Connexion ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)
    print(f"Collection : {collection.count()} chunks")

    print("Chargement chunks...")
    _, children = chunk_all_documents(MD_FOLDER)
    bm25, children = load_or_build_bm25(children)

    return model, reranker, collection, bm25, children

#  RRF 

def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)

def reciprocal_rank_fusion(
    sem_results:  list[dict],
    bm25_results: list[dict],
    n_final:      int = N_RERANK,      # on garde N_RERANK pour le reranker
) -> list[dict]:
    rrf_scores: dict[str, float] = defaultdict(float)
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

# ── Recherche sémantique ──────────────────────────────────────────────────────

def semantic_search(query: str, model, collection, n: int = N_CANDIDATES) -> list[dict]:
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

# ── Recherche BM25 ────────────────────────────────────────────────────────────

def bm25_search(query: str, bm25, children: list, n: int = N_CANDIDATES) -> list[dict]:
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

# ── Chunk minimal depuis ChromaDB ─────────────────────────────────────────────

class SimpleChunk:
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


#RERANKER (Cross-Encoder)


def rerank(
    query:      str,
    candidates: list[dict],     # sortie de reciprocal_rank_fusion
    reranker:   CrossEncoder,
    top_n:      int = N_RESULTS,
) -> list[dict]:
    """
    Re-score chaque paire (query, chunk.text) avec le cross-encoder BGE.
    Retourne les top_n chunks triés par rerank_score décroissant.
    """
    if not candidates:
        return []

    pairs  = [(query, r["chunk"].text) for r in candidates]
    scores = reranker.predict(pairs)            # ndarray float32

    for r, s in zip(candidates, scores):
        r["rerank_score"] = round(float(s), 4)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]

# ═══════════════════════════════════════════════════════════════════════════════
# NOUVEAU ─ GÉNÉRATION LLM
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Tu es un assistant expert en réglementation des changes de la Banque Centrale de Tunisie (BCT).
Lorsqu'une définition réglementaire est présente dans le contexte :

- reproduis la définition le plus fidèlement possible ;
- n'introduis pas de nouveaux concepts juridiques ;
- ne remplace pas les termes réglementaires par des synonymes ;
- distingue clairement le texte réglementaire de l'explication.
Règles obligatoires :
N’écris jamais "Source", "Source 1", "Source 2", "texte fourni", "extrait", "document fourni" ou toute mention visible des sources dans la réponse finale.
N’écris jamais "je ne trouve pas", "je ne dispose pas de l'information", "les textes fournis ne permettent pas", ou toute formulation qui montre un échec de recherche.
 réponds uniquement à partir du contexte, ne combine pas les règles de catégories différentes
 
Rédige comme une note réglementaire :
   - pas de langage conversationnel,
   - pas d’excuses,
   - pas de phrases du type "selon les sources fournies".
réponds uniquement à partir du contexte
 La réponse doit être autonome et directement présentable à un client final.
Sois précis, structuré et professionnel et court. Réponds en français."""


def build_context(ranked: list[dict]) -> str:
    """Formate les chunks retenus en contexte lisible pour le LLM."""
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


from groq import Groq
from groq import RateLimitError


def _generate_groq(query: str, context: str) -> str:
    """
    Utilise automatiquement une deuxième clé Groq
    si la première atteint la limite de quota.
    """

    last_exception = None

    for api_key in GROQ_API_KEYS:

        try:

            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Contexte réglementaire :\n{context}\n\nQuestion : {query}"
                    },
                ],
                temperature=0.1,
                max_tokens=1024,
            )

            return response.choices[0].message.content

        except RateLimitError as e:

            print(f" Quota atteint pour une clé Groq. Passage à la suivante...")
            last_exception = e
            continue

        except Exception as e:

            print(f"Erreur avec une clé Groq : {e}")
            last_exception = e
            continue

    raise RuntimeError(
        "Toutes les clés Groq sont indisponibles."
    ) from last_exception



def generate(query: str, context: str, provider: str = LLM_PROVIDER) -> str:
    if provider == "groq":
        return _generate_groq(query, context)
    
    else:
        raise ValueError(f"Provider inconnu : '{provider}'. Utilise 'groq' ou 'gemini'.")


# PIPELINE COMPLET : hybrid_search → rerank → LLM


def hybrid_rag(
    query:      str,
    model,
    reranker,
    collection,
    bm25,
    children:   list,
    provider:   str = LLM_PROVIDER,
) -> dict:
    
    # 1. Retrieval dual
    sem_results  = semantic_search(query, model, collection, n=N_CANDIDATES)
    bm25_results = bm25_search(query, bm25, children, n=N_CANDIDATES)

    # 2. Fusion RRF  →  N_RERANK candidats pour le reranker
    fused = reciprocal_rank_fusion(sem_results, bm25_results, n_final=N_RERANK)

    # 3. Reranking cross-encoder  →  N_RESULTS chunks finaux
    ranked = rerank(query, fused, reranker, top_n=N_RESULTS)

    # 4. Génération LLM
    context = build_context(ranked)
    answer  = generate(query, context, provider)

    sources = [
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

    return {"answer": answer, "sources": sources}

# ── Affichage ─────────────────────────────────────────────────────────────────

def afficher_sources(sources: list[dict]):
    print("\n📚 Sources retenues après reranking :")
    for s in sources:
        print(
            f"  [{s['rank']}] rrf={s['rrf_score']:.5f} | rerank={s['rerank_score']:.4f}"
            f" | {s['type']} {s['number']} — {s['article'] or s['breadcrumb'][:60]}"
        )


def afficher_reponse(answer: str):
    print("\n" + "═" * 80)
    print("💬 RÉPONSE")
    print("─" * 80)
    print(answer)
    print("═" * 80)

# ── Interface interactive ─────────────────────────────────────────────────────

def main():
    model, reranker, collection, bm25, children = load_resources()

    print("\n" + "━" * 80)
    print("  MOTEUR HYBRIDE — BGE-M3 + BM25 + RRF + CrossEncoder + LLM")
    print(f"  Provider LLM : {LLM_PROVIDER.upper()}")
    print("  Tape 'quit' pour quitter")
    print("━" * 80)

    while True:
        try:
            question = input("\n Question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        result = hybrid_rag(question, model, reranker, collection, bm25, children)

        afficher_sources(result["sources"])
        afficher_reponse(result["answer"])

        # Optionnel : voir le texte complet d'une source
        choix = input("\n  Chunk complet ? (numéro ou Entrée) : ").strip()
        if choix.isdigit():
            idx = int(choix) - 1
            if 0 <= idx < len(result["sources"]):
                # retrouver le chunk depuis fused (on re-cherche par chunk_id)
                cid    = result["sources"][idx]["chunk_id"]
                # recherche rapide dans children
                target = next((c for c in children if c.chunk_id == cid), None)
                if target:
                    print("\n" + "═" * 80)
                    print(target.text)
                    print("═" * 80)


if __name__ == "__main__":
    main()