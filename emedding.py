import argparse
import re
import unicodedata
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from chunk import chunk_all_documents


# ── Configuration ─────────────────────────────────────────────────────────────
MD_FOLDER    = Path("pdf_chunks")
CHROMA_PATH  = "./chroma_db"
COLLECTION   = "reglementation_bct"
EMBED_MODEL  = "BAAI/bge-m3"
MAX_SEQ_LEN  = 8192          # corrige le défaut 128 tokens de BGE-M3
BATCH_SIZE   = 32            # 16 → 32 : x2 vitesse si GPU disponible
BGE_PREFIX   = "Represent this sentence for searching relevant passages: "

# Seuil de déduplication : ratio SequenceMatcher au-dessus duquel
# deux chunks sont considérés quasi-identiques (0.95 = très strict)
DEDUP_THRESHOLD = 0.95


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Normalise un texte pour la comparaison de déduplication."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_chunks(chunks: list) -> tuple[list, int]:
    """
    Supprime les chunks quasi-identiques basés sur le texte normalisé.
    Garde le premier exemplaire de chaque groupe (le mieux structuré
    est généralement celui parsé en premier).

    Retourne : (chunks_uniques, nb_supprimes)
    """
    from difflib import SequenceMatcher

    seen: list[tuple[str, str]] = []   # (chunk_id, texte_normalisé)
    kept: list = []
    removed = 0

    for chunk in chunks:
        norm = normalize_text(chunk.text or "")
        # Clé primaire : même (document_number, article_title) → candidat doublon
        key = (chunk.document_number or "", chunk.article_title or chunk.breadcrumb or "")

        is_dup = False
        for seen_key, seen_norm in seen:
            if key == seen_key and key != ("", ""):
                ratio = SequenceMatcher(None, norm, seen_norm).ratio()
                if ratio >= DEDUP_THRESHOLD:
                    is_dup = True
                    break

        if is_dup:
            removed += 1
        else:
            kept.append(chunk)
            seen.append((key, norm))

    return kept, removed


#  CONSTRUCTION DU TEXTE D'EMBEDDING (VERSION AMÉLIORÉE)

def generate_question_hint(chunk) -> str:
    
    doc_ref = " ".join(filter(None, [
        chunk.document_type,
        chunk.document_number,
    ]))
    subject = chunk.article_title or chunk.breadcrumb or chunk.document_title or ""

    if chunk.article_title:
        # Chunk d'article → question sur les conditions/dispositions de l'article
        return f"Q: Que prévoit {doc_ref} concernant {chunk.article_title} ?"
    elif chunk.breadcrumb:
        # Chunk de section → question sur les dispositions de la section
        # On prend le dernier segment du breadcrumb (le plus précis)
        last_crumb = chunk.breadcrumb.split(">")[-1].strip()
        return f"Q: Quelles sont les dispositions relatives à {last_crumb} selon {doc_ref} ?"
    elif chunk.document_title:
        # Chunk d'en-tête ou de titre → question sur l'objet du texte
        return f"Q: Quel est l'objet de {doc_ref} — {chunk.document_title} ?"
    else:
        return f"Q: Que contient {doc_ref} ?" if doc_ref.strip() else ""


def build_embedding_text(chunk) -> str:
    

    # ── Bloc de référence 
    ref_parts = filter(None, [
        chunk.document_type,
        chunk.document_number,
        f"— {chunk.document_title}" if chunk.document_title else "",
    ])
    ref_line = " ".join(ref_parts).strip()

    
    context_parts = []
    if chunk.breadcrumb:
        context_parts.append(f"Contexte : {chunk.breadcrumb}")
    if chunk.article_title and chunk.article_title not in (chunk.breadcrumb or ""):
        context_parts.append(f"Article : {chunk.article_title}")
    context_block = "\n".join(context_parts)

    # ── AMÉLIORATION 2 : Style Q/A simulé 
    question_hint = generate_question_hint(chunk)
    # La réponse courte = les 200 premiers caractères du chunk (style "A:")
    answer_hint = ""
    if question_hint and chunk.text:
        short_answer = chunk.text[:200].replace("\n", " ").strip()
        answer_hint = f"A: {short_answer}"

    # ── Assemblage final 
    sections = filter(None, [
        ref_line,
        context_block,
        question_hint,
        answer_hint,
        "",                        
        "Contenu :",
        chunk.text or "",
    ])

    return "\n".join(sections).strip()


#  PIPELINE D'INDEXATION

def main(reset: bool = False, dry_run: bool = False):

    # ── 1. Chargement du modèle 
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = MAX_SEQ_LEN          # CRITIQUE : corrige la troncation
    print(f"  max_seq_length = {model.max_seq_length}")

    # ── 2. Génération des chunks 
    _, children = chunk_all_documents(MD_FOLDER)
    print(f"  {len(children)} chunks bruts chargés")

    # ── 3. Déduplication 
    print("\nDéduplication des chunks quasi-identiques...")
    children, nb_removed = deduplicate_chunks(children)
    print(f"  {nb_removed} doublons supprimés → {len(children)} chunks uniques")

    # ── 4. Mode dry-run : affiche 5 exemples et sort 
    if dry_run:
        print("\n── DRY RUN : 5 exemples de textes d'embedding ──\n")
        for i, chunk in enumerate(children[:5]):
            print(f"{'='*70}")
            print(f"[{i+1}] chunk_id = {chunk.chunk_id}")
            print(f"{'─'*70}")
            print(build_embedding_text(chunk))
            print()
        return

    # ── 5. Construction des textes d'embedding 
    print("\nConstruction des textes d'embedding enrichis...")
    texts = [build_embedding_text(c) for c in children]

    # Statistiques sur les longueurs
    lengths = [len(t) for t in texts]
    print(f"  Longueur moyenne : {sum(lengths)//len(lengths)} chars")
    print(f"  Longueur max     : {max(lengths)} chars")
    print(f"  Longueur min     : {min(lengths)} chars")

    # ── 6. Calcul des embeddings 
    print(f"\nCalcul des embeddings (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"  {len(embeddings)} embeddings calculés — shape : {embeddings.shape}")

    # ── 7. Connexion ChromaDB 
    print("\nConnexion à ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        print("  Reset de la collection...")
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Collection '{COLLECTION}' prête ({collection.count()} chunks existants)")

    # ── 8. Vidage si non-reset mais collection non vide 
    if not reset and collection.count() > 0:
        print("  Vidage des chunks existants avant réindexation...")
        try:
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)
                print(f"  {len(existing_ids)} chunks supprimés")
        except Exception as e:
            print(f"  Avertissement lors du vidage : {e}")

    # ── 9. Insertion par lots 
    print(f"\nInsertion de {len(children)} chunks dans ChromaDB...")
    total = len(children)

    for i in range(0, total, BATCH_SIZE):
        batch_chunks = children[i:i + BATCH_SIZE]
        batch_embs   = embeddings[i:i + BATCH_SIZE]

        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=[emb.tolist() for emb in batch_embs],
            documents=[c.text for c in batch_chunks],
            metadatas=[c.to_dict() for c in batch_chunks],
        )

        done = min(i + BATCH_SIZE, total)
        bar  = "█" * (done * 30 // total) + "░" * (30 - done * 30 // total)
        print(f"  [{bar}] {done}/{total}", end="\r")

    print()

    # ── 10. Vérification finale 
    final_count = collection.count()
    print(f"\n✅ Indexation terminée.")
    print(f"   Chunks dans ChromaDB : {final_count}")
    print(f"   Doublons évités      : {nb_removed}")
    print(f"   Modèle               : {EMBED_MODEL} (max_seq={MAX_SEQ_LEN})")
    print(f"   Améliorations actives:")
    print(f"     • Contexte parent injecté (article_title + breadcrumb en préfixe)")
    print(f"     • Style Q/A simulé (question + réponse courte générées)")
    print(f"     • Déduplication pré-indexation (seuil={DEDUP_THRESHOLD})")

    # Exemple de ce qui a été indexé
    print(f"\n── Exemple de texte embeddé (chunk 0) ──")
    print(build_embedding_text(children[0]))


#  POINT D'ENTRÉE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexation BCT améliorée")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprime et recrée la collection ChromaDB avant d'indexer"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche 5 exemples de textes d'embedding sans indexer"
    )
    args = parser.parse_args()
    main(reset=args.reset, dry_run=args.dry_run)
