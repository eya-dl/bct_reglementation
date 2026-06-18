import chromadb
from sentence_transformers import SentenceTransformer

# ── Chargement ────────────────────────────────────────────────────────────────

print("Chargement modèle BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3")

print("Connexion ChromaDB...")
client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("reglementation_bct")

print(f"Collection : {collection.count()} chunks indexés\n")

# ── Préfixe BGE-M3 pour les requêtes ─────────────────────────────────────────

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Seuil de confiance ────────────────────────────────────────────────────────

SCORE_SEUIL = 0.15   # En dessous → résultat probablement non pertinent

# ── Fonction de recherche ─────────────────────────────────────────────────────

def recherche(question: str, n_results: int = 5) -> list[dict]:

    embedding = model.encode(
        BGE_QUERY_PREFIX + question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "score":    round(1 - dist, 4),   # distance cosine → similarité
            "doc":      doc,
            "meta":     meta,
        })

    return hits

# ── Affichage ─────────────────────────────────────────────────────────────────

def afficher_resultats(question: str, hits: list[dict]):

    print("\n" + "╔" + "═" * 78 + "╗")
    print(f"║  QUESTION : {question[:64]:<64}║")
    print("╚" + "═" * 78 + "╝")

    for i, hit in enumerate(hits, 1):
        score = hit["score"]
        meta  = hit["meta"]
        doc   = hit["doc"]

        # Indicateur de confiance
        if score >= 0.40:
            confiance = " FORT   "
        elif score >= SCORE_SEUIL:
            confiance = " MOYEN  "
        else:
            confiance = " FAIBLE "

        print(f"\n┌─ Résultat {i}  {confiance}  score={score:.4f} " + "─" * 30)

        # Métadonnées
        doc_type   = meta.get("document_type",   "—")
        doc_num    = meta.get("document_number",  "—")
        breadcrumb = meta.get("breadcrumb",       "—")
        article    = meta.get("article_title",    "")
        level      = meta.get("level",            "—")
        chunk_type = meta.get("chunk_type",       "—")

        print(f"│  Type       : {doc_type} {doc_num}")
        print(f"│  Breadcrumb : {breadcrumb[:90]}")
        if article:
            print(f"│  Article    : {article}")
        print(f"│  Niveau     : {level} ({chunk_type})")
        print("│")

        # Contenu tronqué à 600 caractères
        contenu = doc.strip().replace("\n", " ")
        if len(contenu) > 600:
            contenu = contenu[:600] + "..."
        print(f"│  {contenu}")
        print("└" + "─" * 78)

    # Résumé des scores
    scores = [h["score"] for h in hits]
    print(f"\n  Scores : min={min(scores):.4f}  max={max(scores):.4f}  "
          f"moy={sum(scores)/len(scores):.4f}")

    faibles = [h for h in hits if h["score"] < SCORE_SEUIL]
    if faibles:
        print(f"   {len(faibles)} résultat(s) sous le seuil ({SCORE_SEUIL})"
              f"  pertinents")

# ── Boucle interactive ────────────────────────────────────────────────────────

def main():
    print("━" * 80)
    print("  TEST RETRIEVAL — BGE-M3 + ChromaDB")
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
            print("Au revoir.")
            break

        hits = recherche(question, n_results=5)
        afficher_resultats(question, hits)

        # Option : afficher le chunk complet
        choix = input("\n  Afficher un chunk complet ? (numéro ou Entrée) : ").strip()
        if choix.isdigit():
            idx = int(choix) - 1
            if 0 <= idx < len(hits):
                print("\n" + "═" * 80)
                print("CHUNK COMPLET")
                print("═" * 80)
                print(hits[idx]["doc"])
                print("═" * 80)

if __name__ == "__main__":
    main()