import re
import numpy as np
import unicodedata
import pickle
import os
from rank_bm25 import BM25Okapi
from chunk import chunk_all_documents
from pathlib import Path

# STOPWORDS (étendus)
STOPWORDS_FR = set([
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "ou", "en", "dans",
    "pour", "par", "sur", "avec", "sans", "ce", "cette", "ces", "cet", "qui", "que",
    "au", "aux", "d", "l", "n", "ne", "pas", "plus", "moins", "etre", "est",
    "sont", "a", "dont", "comme", "donc", "il", "elle", "ils", "elles",
    "leur", "leurs", "tout", "toute", "tous", "toutes", "afin", "ainsi",
    "lors", "doit", "doivent", "peut", "peuvent", "ainsi", "soit", "sa",
    "son", "ses", "se", "s", "y", "on", "ou", "ni", "ou", "etant", "ete",
    "fait", "faite", "selon", "entre", "vers", "chez", "depuis", "avant",
    "apres", "sous", "leurs", "celui", "celle", "ceux", "celles"
])
# NB: normalize() retire les accents -> on ne met que des formes sans accent ici


PICKLE_PATH = "bm25_store.pkl"

# Pondération configurable (ajustable selon tes tests)
BM25_PARAMS = {"k1": 1.5, "b": 0.5}  # b réduit car chunks courts/homogènes

# Boosts (à ajuster empiriquement)
BOOST_NUMBER_MATCH = 6.0      # le numéro de la requête matche le document_number du chunk
BOOST_TITLE_MATCH = 2.5        # un terme de la requête apparaît dans article_title
BOOST_NUMBER_IN_TEXT = 3.0     # le numéro est présent ailleurs dans le texte/breadcrumb


# ─────────────────────────────
# NORMALISATION
# ─────────────────────────────
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


# ─────────────────────────────
# TOKENIZER
# ─────────────────────────────
def tokenize(text):
    text = normalize(text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS_FR]


# ─────────────────────────────
# EXTRACTION NUMÉRIQUE
# ─────────────────────────────
NUM_PATTERN = re.compile(r"\d{1,4}[-/]\d{1,4}|\d+")


def extract_numbers(text):
    """Retourne l'ensemble des séquences numériques (ex: '94-07', '2019', '12')."""
    return set(NUM_PATTERN.findall(text.lower()))


def number_variants(num):
    """
    Génère des variantes normalisées d'une référence numérique
    pour améliorer le matching BM25 pur (sans dépendre de strict_filter).
    Ex: '94-07' -> {'94-07', '9407', '94', '07'}
        '2019'  -> {'2019'}
    """
    variants = {num}
    if "-" in num or "/" in num:
        sep = "-" if "-" in num else "/"
        parts = num.split(sep)
        variants.add("".join(parts))       
        variants.update(parts)              
    return variants


# ─────────────────────────────
# BUILD TEXT BM25
# ─────────────────────────────
def build_text(c):
    """
    Construit le texte indexé par BM25.
    On ajoute des tokens synthétiques pour les numéros de document
    (ex: '94-07' -> aussi '9407', '94', '07') afin que le score BM25 brut
    capte les références numériques même sans strict_filter.
    """
    parts = [
        c.document_type or "",
        c.document_number or "",
        c.article_title or "",
        c.breadcrumb or "",
        c.text or "",
    ]

    extra_tokens = []
    if c.document_number:
        for num in extract_numbers(c.document_number):
            extra_tokens.extend(number_variants(num))

    base_text = " ".join(parts)
    if extra_tokens:
        base_text += " " + " ".join(extra_tokens)

    return base_text


# PICKLE
def save(bm25, children, corpus):
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "children": children, "corpus": corpus}, f)


def load():
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["children"], data["corpus"]


# LOAD / BUILD
md_folder = Path("pdf_chunks")

if os.path.exists(PICKLE_PATH):
    bm25, children, corpus = load()
    print("[OK] BM25 chargé")

else:
    print("[INFO] build BM25...")

    parents, children = chunk_all_documents(md_folder)

    corpus = [build_text(c) for c in children]
    tokenized = [tokenize(t) for t in corpus]

    bm25 = BM25Okapi(tokenized, **BM25_PARAMS)

    save(bm25, children, corpus)
    print("[OK] BM25 sauvegardé")


# BOOST CONDITIONNEL (au lieu d'un bonus aveugle)
def compute_boost(query, q_tokens, q_nums, chunk):
    """
    Calcule un boost qui dépend réellement du contenu de la requête,
    et non plus un bonus fixe appliqué à tout le monde.
    """
    boost = 0.0

    # --- Matching numérique sur document_number ---
    if q_nums and chunk.document_number:
        c_nums = extract_numbers(chunk.document_number)
        # comparer aussi les variantes (94-07 vs 9407 vs 94 / 07)
        c_variants = set()
        for n in c_nums:
            c_variants.update(number_variants(n))

        q_variants = set()
        for n in q_nums:
            q_variants.update(number_variants(n))

        if q_variants & c_variants:
            boost += BOOST_NUMBER_MATCH
        elif q_nums & extract_numbers(chunk.text or ""):
            # le numéro apparaît dans le texte mais pas dans document_number
            boost += BOOST_NUMBER_IN_TEXT

    # --- Matching sur article_title ---
    if chunk.article_title:
        title_tokens = set(tokenize(chunk.article_title))
        if title_tokens & set(q_tokens):
            boost += BOOST_TITLE_MATCH

    return boost


# BM25 SEARCH + RÉPONSE
def bm25_search(query, top_k=5):

    query_tokens = tokenize(query)
    q_nums = extract_numbers(query)

    scores = bm25.get_scores(query_tokens)

    results = []
    for idx, c in enumerate(children):  # fix: plus de children.index(c) -> O(n)
        raw_score = float(scores[idx])
        boost = compute_boost(query, query_tokens, q_nums, c)
        final_score = raw_score + boost

        results.append({
            "score": final_score,
            "raw_score": raw_score,
            "boost": boost,
            "chunk": c,
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


# ─────────────────────────────
#  MODE RÉPONSE (RAG SIMPLE)
# ─────────────────────────────
def answer(query):

    results = bm25_search(query, top_k=3)

    if not results:
        return "Aucun résultat trouvé."

    response = "\n\n".join([
        f" SOURCE {i+1} (score={r['score']:.2f}, raw={r['raw_score']:.2f}, boost={r['boost']:.2f})\n"
        f"Type: {r['chunk'].document_type} | Num: {r['chunk'].document_number}\n"
        f"{r['chunk'].text[:800]}"
        for i, r in enumerate(results)
    ])

    return response


# TEST
if __name__ == "__main__":
    

    while True:
        try:
            # Récupération de l'input utilisateur
            user_query = input(" Entrez votre question: ").strip()
            
            # Gestion d'une entrée vide (on évite de lancer une recherche pour rien)
            if not user_query:
                continue
                
            # Condition de sortie
            if user_query.lower() in ["exit", "quit"]:
                
                break

            print("\n" + "-"*30)
            print(f"QUESTION: {user_query}")
            print("-"*30)

            # Lancement de la recherche et affichage
            print(answer(user_query))
            print("\n" + "="*50 + "\n")

        except KeyboardInterrupt:
            # Gestion propre du Ctrl+C dans le terminal
            print("\n\n Session interrompue (Ctrl+C).")
            break