"""
config.py — Constantes et paramètres globaux du pipeline agentique BCT
=======================================================================
Seul fichier à modifier pour changer les paramètres du système.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Chemins ───────────────────────────────────────────────────────────────────
PICKLE_PATH  = "bm25_store.pkl"
CHROMA_PATH  = "./chroma_db"
COLLECTION   = "reglementation_bct"
MD_FOLDER    = Path("pdf_chunks")

# ── Retrieval ─────────────────────────────────────────────────────────────────
BGE_PREFIX   = "Represent this sentence for searching relevant passages: "
RRF_K        = 60
N_CANDIDATES = 20   # candidats par moteur avant fusion RRF
N_RERANK     = 15   # chunks envoyés au cross-encoder après RRF
N_RESULTS    = 5    # résultats finaux retenus

# ── Boosts BM25 ───────────────────────────────────────────────────────────────
BOOST_NUMBER_MATCH   = 6.0
BOOST_TITLE_MATCH    = 2.5
BOOST_NUMBER_IN_TEXT = 3.0

# ── Modèles ───────────────────────────────────────────────────────────────────
EMBED_MODEL    = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LEN = 512

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq")   # "groq" | "gemini"

# ── Clés Groq ────────────────────────────────────────────────────────────────
GROQ_API_KEYS = [
    os.getenv(f"GROQ_API_KEY_{i}")
    for i in range(1, 14)
]
# Supprime les clés vides

GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k]

if not GROQ_API_KEYS:
    raise RuntimeError("Aucune clé Groq trouvée.")




print(f" {len(GROQ_API_KEYS)} clé(s) Groq chargée(s).")

GROQ_MODEL      = "llama-3.3-70b-versatile"

# ── Clé Gemini (optionnel) ────────────────────────────────────────────────────
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
]
GEMINI_MODEL    = "gemini-1.5-flash"

# ── Pipeline agentique ────────────────────────────────────────────────────────
MAX_RETRIES  = 2    # max tentatives retrieval si hallucination détectée
MAX_REPLANS  = 2    # max replanifications si réponse non pertinente
CONTEXT_LIMIT = 4000  # chars max du contexte envoyé aux graders

# ── Stopwords français ────────────────────────────────────────────────────────
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

# ── Réponses statiques ────────────────────────────────────────────────────────
HORS_SCOPE_REPONSE = (
    "Cet assistant est dédié à la réglementation des changes de la Banque Centrale "
    "de Tunisie. Il peut répondre aux questions relatives aux transferts, comptes en "
    "devises, investissements étrangers, change manuel, importation/exportation de "
    "moyens de paiement, etc. Votre question semble en dehors de ce périmètre."
)

NOTION_GENERALE_DISCLAIMER = (
    "\n\n*Remarque : cette explication est une notion générale et ne provient pas "
    "d'un texte réglementaire BCT spécifique. Pour toute démarche officielle, "
    "consultez les textes réglementaires applicables.*"
)

SALUTATION_REPONSE = (
    "Bonjour. Je suis l'assistant réglementaire de la Banque Centrale de Tunisie. "
    "Je suis à votre disposition pour répondre à vos questions sur la réglementation "
    "des changes : transferts, comptes en devises, investissements, change manuel, "
    "bureaux de change, emprunts extérieurs, etc. Comment puis-je vous aider ?"
)

# ── Prompt système LLM générateur ─────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant expert en réglementation des changes de la Banque Centrale de Tunisie (BCT).
Lorsqu'une définition réglementaire est présente dans le contexte :

- reproduis la définition le plus fidèlement possible ;
- n'introduis pas de nouveaux concepts juridiques ;
- ne remplace pas les termes réglementaires par des synonymes ;
- distingue clairement le texte réglementaire de l'explication.

Règles obligatoires :
N'écris jamais "Source", "Source 1", "texte fourni", "extrait", "document fourni"
ou toute mention visible des sources dans la réponse finale.
N'écris jamais "je ne trouve pas" ou toute formulation qui montre un échec de recherche.
Réponds uniquement à partir du contexte, ne combine pas les règles de catégories différentes.



Rédige comme une note réglementaire :
- pas de langage conversationnel,
- pas d'excuses,
- pas de phrases du type "selon les sources fournies".

La réponse doit être autonome et directement présentable à un client final.
Sois précis, structuré, professionnel et concis. Réponds en français."""

# Limite de sécurité pour éviter une décomposition excessive (ex: question mal formée)
MAX_SUBQUESTIONS = 5
BOOST_PRIOR_DOCUMENT = 0.15  # boost du rerank_score pour les chunks du même document que la sous-question précédente

BOOST_PRIOR_DOCUMENT = 0.15


import re


CROSS_REF_PATTERNS = [
    # "article 12 de la loi n°76-18" / "article 12 de la circulaire n°2020-01"
    re.compile(r"article\s+(\d+)\s+(?:de\s+la\s+|du\s+)(loi|décret|circulaire)\s*(?:n°|no|numéro)?\s*([\d\-/]+)", re.IGNORECASE),
    # "loi n°76-18" / "circulaire n°2020-01" (référence sans article précis)
    re.compile(r"(loi|décret|circulaire)\s*n°\s*([\d\-/]+)", re.IGNORECASE),
    # "conformément à l'article 12" (référence interne au même texte)
    re.compile(r"(?:conformément à|voir|selon)\s+l['’]article\s+(\d+)", re.IGNORECASE),
]