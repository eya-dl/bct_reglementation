import chromadb
from sentence_transformers import SentenceTransformer
from chunk import chunk_all_documents
from pathlib import Path

# 1. Charger modèle
model = SentenceTransformer("BAAI/bge-m3")
print(model.max_seq_length)
# 2. Générer chunks
md_folder = Path("pdf_chunks")
parents, children = chunk_all_documents(md_folder)

# 3. Fonction embedding 
def build_embedding_text(chunk):
    return f"""
Type document : {chunk.document_type}
Titre document : {chunk.document_title}
Numéro : {chunk.document_number}

Breadcrumb : {chunk.breadcrumb}

Contenu :
{chunk.text}
""".strip()

# 4. Préparer textes
texts = [build_embedding_text(c) for c in children]

# 5. Embeddings
embeddings = model.encode(
    texts,
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("OK embeddings :", len(embeddings))

# Création de la base Chroma

print("Connexion à ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="reglementation_bct",
    metadata={"hnsw:space": "cosine"}  
)

print("Insertion des chunks...")

# Optionnel : vider la collection si elle existe déjà
try:
    collection.delete(
        ids=collection.get()["ids"]
    )
except:
    pass

# Insertion par lots
batch_size = 100

for i in range(0, len(children), batch_size):

    batch_chunks = children[i:i + batch_size]
    batch_embs   = embeddings[i:i + batch_size]

    collection.add(
        ids=[
            c.chunk_id
            for c in batch_chunks
        ],

        embeddings=[
            emb.tolist()
            for emb in batch_embs
        ],

        documents=[
            c.text
            for c in batch_chunks
        ],

        metadatas=[
            c.to_dict()
            for c in batch_chunks
        ]
    )

    print(
        f"{min(i + batch_size, len(children))}/{len(children)} indexés"
    )

print("\nIndexation terminée.")
print("Total dans Chroma :", collection.count())
