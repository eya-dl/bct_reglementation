import chromadb

print("Version :", chromadb.__version__)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

print("Connexion OK")
