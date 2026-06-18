# debug_corpus.py
import re
from pathlib import Path
from collections import Counter

md_folder = Path("pdf_chunks")

# Mots-clés à surveiller
KEYWORDS = [
    "circulaire", "loi", "decret", "avis", "note", "arrete",
    "chapitre", "titre", "section", "article", "paragraphe"
]

print("=" * 60)
print("DIAGNOSTIC CORPUS BCT")
print("=" * 60)

all_lines = []

for md_file in sorted(md_folder.glob("*.md")):
    text  = md_file.read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    all_lines.extend(lines)

print(f"\nTotal lignes non vides : {len(all_lines)}")

# ── 1. Formes exactes des entêtes de documents ────────────────────────────────
print("\n── Lignes contenant un numéro de document (forme exacte) ──")
num_re = re.compile(r"N[°º]?\s*[\d]{2,4}[-–/][\d]{2,4}", re.IGNORECASE)

doc_lines = [l for l in all_lines if num_re.search(l)]
print(f"  {len(doc_lines)} lignes avec numéro détecté\n")
for l in doc_lines[:30]:
    print(f"  {repr(l)}")

# ── 2. Formes des marqueurs structurels ───────────────────────────────────────
print("\n── Lignes commençant par CHAPITRE / TITRE / SECTION ──")
struct_re = re.compile(r"^(CHAPITRE|TITRE|SECTION|PARAGRAPHE)\b", re.IGNORECASE)
struct_lines = [l for l in all_lines if struct_re.match(l)]
for l in struct_lines[:20]:
    print(f"  {repr(l)}")

# ── 3. Formes des sections romaines ──────────────────────────────────────────
print("\n── Lignes commençant par chiffre romain + séparateur ──")
rom_re = re.compile(r"^[IVXLC]+\s*[-–.\s]", re.IGNORECASE)
rom_lines = [l for l in all_lines if rom_re.match(l)]
for l in rom_lines[:20]:
    print(f"  {repr(l)}")

# ── 4. Formes des articles ────────────────────────────────────────────────────
print("\n── Lignes commençant par Article ──")
art_re = re.compile(r"^Article\s+\d+", re.IGNORECASE)
art_lines = [l for l in all_lines if art_re.match(l)]
print(f"  {len(art_lines)} articles détectés")
for l in art_lines[:10]:
    print(f"  {repr(l)}")

# ── 5. Lignes qui devraient être des documents mais ne matchent pas ───────────
print("\n── Lignes avec mot-clé doc mais sans numéro détecté ──")
doc_kw_re  = re.compile(r"^(CIRCULAIRE|LOI|DECRET|AVIS|NOTE|ARRETE)\b", re.IGNORECASE)
has_num_re = re.compile(r"N[°º]?\s*[\d]", re.IGNORECASE)

suspicious = [
    l for l in all_lines
    if doc_kw_re.match(l) and not has_num_re.search(l)
]
print(f"  {len(suspicious)} lignes suspectes")
for l in suspicious[:15]:
    print(f"  {repr(l)}")

# ── 6. Distribution longueurs de lignes des entêtes ──────────────────────────
print("\n── Longueurs des lignes de type document ──")
doc_kw_lines = [l for l in all_lines if doc_kw_re.match(l)]
lengths = Counter(len(l) for l in doc_kw_lines)
for length, count in sorted(lengths.items()):
    bar = "█" * min(count, 40)
    print(f"  {length:3d} chars : {bar} ({count})")

# ── 7. Échantillon des 50 premières lignes d'un fichier ──────────────────────
print("\n── 50 premières lignes du premier fichier ──")
first_file = sorted(md_folder.glob("*.md"))[0]
lines = [l.strip() for l in first_file.read_text(encoding="utf-8").splitlines()]
for i, l in enumerate(lines[:50]):
    if l:
        print(f"  {i:3d}: {repr(l)}")