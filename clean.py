from pathlib import Path
import re


# ── Suppression tableaux vides ────────────────────────────────────────────────

def remove_empty_tables(markdown: str) -> str:
    lines   = markdown.splitlines()
    cleaned = []
    i = 0

    while i < len(lines):
        if lines[i].startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1

            non_empty = sum(
                1
                for line in table_lines
                for cell in line.strip("|").split("|")
                if cell.strip() and cell.strip() != "---"
            )

            if non_empty >= 5:
                cleaned.extend(table_lines)
        else:
            cleaned.append(lines[i])
            i += 1

    return "\n".join(cleaned)


# ── Nettoyages spécifiques BCT ────────────────────────────────────────────────

def remove_page_artifacts(markdown: str) -> str:
    """
    Supprime les artefacts de pagination Docling :
    - Numéros de page isolés : "42", "- 42 -"
    - En-têtes/pieds répétitifs
    - Lignes de pointillés : "............."
    - Tirets de séparation : "----------"
    """
    lines   = markdown.splitlines()
    cleaned = []

    # Numéro de page seul sur une ligne
    page_num_re   = re.compile(r"^\s*-?\s*\d{1,3}\s*-?\s*$")
    # Ligne de pointillés ou tirets
    dots_re       = re.compile(r"^[\.\-\_\*]{4,}\s*$")
    # Références de bas de page : "1 Ainsi modifié par..."
    footnote_re   = re.compile(r"^\d+\s+(?:Ainsi|Tel|Voir|Modifié)", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        if page_num_re.match(stripped):
            continue
        if dots_re.match(stripped):
            continue
        # Garder les footnotes mais les marquer pour le chunker
        if footnote_re.match(stripped):
            cleaned.append(f"[NOTE] {stripped}")
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def fix_broken_lines(markdown: str) -> str:
    """
    Réassemble les lignes cassées par Docling.

    Docling coupe parfois :
        "CIRCULAIRE AUX INTERMEDIAIRES AGREES"
        "N°94-03 DU 1ER FEVRIER 1994"
    en deux lignes. On les réassemble si la ligne suivante
    commence par N° ou est une continuation évidente.
    """
    lines   = markdown.splitlines()
    result  = []
    i = 0

    # Ligne qui commence un entête de document sans numéro
    doc_start_re  = re.compile(
        r"^(?:CIRCULAIRE|LOI|DECRET|NOTE|ARRETE)\b",
        re.IGNORECASE
    )
    # Ligne qui commence par un numéro de document
    num_start_re  = re.compile(
        r"^N[°º]?\s*[\d]{2,4}[-–/][\d]{2,4}",
        re.IGNORECASE
    )
    # Ligne qui commence par OBJET :
    objet_re = re.compile(r"^OBJET\s*:", re.IGNORECASE)

    while i < len(lines):
        line      = lines[i].strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        # Entête document sans numéro + ligne suivante = numéro
        if (doc_start_re.match(line)
                and not re.search(r"N[°º]?\s*[\d]", line, re.IGNORECASE)
                and num_start_re.match(next_line)):
            result.append(line + " " + next_line)
            i += 2
            continue

        # "OBJET :" seul sur une ligne → fusionner avec la suivante
        if objet_re.match(line) and next_line and not next_line.startswith("|"):
            result.append(line + " " + next_line)
            i += 2
            continue

        result.append(lines[i])
        i += 1

    return "\n".join(result)


def normalize_article_headers(markdown: str) -> str:
    """
    Normalise les variantes de titres d'articles pour le parser.

    Docling produit parfois :
        "Article  1 er :"   → "Article 1er :"
        "Article1er"        → "Article 1er"
        "ARTICLE 5"         → "Article 5"
        "Art. 5"            → "Article 5"
    """
    # ────────────────────────────────────────────────────────────────────
    # ✅ FIX BCT-2 (cohérence avec chunk.py) — liste de suffixes ordinaux
    # complète (quater/quinquies/sexto/septies/octies en plus de bis/ter),
    # pour que le nettoyage n'introduise pas d'espacement bizarre sur des
    # formes qu'ARTICLE_RE reconnaît désormais dans chunk.py.
    # Espaces multiples dans "Article  1  er"
    # ────────────────────────────────────────────────────────────────────
    markdown = re.sub(
        r"Article\s{2,}(\d+)\s*(er|bis|ter|quater|quinquies|sexto|septies|octies|nouveau)?",
        lambda m: f"Article {m.group(1)}{(' ' + m.group(2)) if m.group(2) else ''}",
        markdown,
        flags=re.IGNORECASE
    )
    # "ARTICLE" → "Article"
    markdown = re.sub(
        r"^ARTICLE\s+(\d+)",
        lambda m: f"Article {m.group(1)}",
        markdown,
        flags=re.IGNORECASE | re.MULTILINE
    )
    # "Art. 5" → "Article 5"
    markdown = re.sub(
        r"^Art\.\s+(\d+)",
        lambda m: f"Article {m.group(1)}",
        markdown,
        flags=re.IGNORECASE | re.MULTILINE
    )

    return markdown


def remove_duplicate_headers(markdown: str) -> str:
    """
    Supprime les entêtes de document dupliquées.

    Docling répète parfois le titre d'un document plusieurs fois
    quand il apparaît en en-tête de page.
    On garde la première occurrence, on supprime les suivantes.
    """
    lines  = markdown.splitlines()
    seen   = set()
    result = []

    # Pattern d'entête de document
    doc_header_re = re.compile(
        r"^(?:#{1,6}\s+)?(?:CIRCULAIRE|LOI|DECRET|AVIS|NOTE|ARRETE)"
        r".*N[°º]?\s*[\d]{2,4}[-–/][\d]{2,4}",
        re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        if doc_header_re.match(stripped):
            # Normaliser pour comparaison (supprimer espaces multiples)
            key = re.sub(r"\s+", " ", stripped).upper()
            if key in seen:
                continue   # doublon → on saute
            seen.add(key)
        result.append(line)

    return "\n".join(result)


def remove_list_bullets(markdown: str) -> str:
    """
    Convertit les listes à puces Docling en texte plat.

    Docling transforme parfois les alinéas en :
        "- a) du produit en dinars..."
        "- b) des sommes provenant..."
    Ces puces perturbent la détection des patterns structurels.
    On garde le contenu, on supprime juste le "- " initial.
    """
    # "- -CIRCULAIRE..." → on garde pour la détection des listes de références
    # "- a)" / "- b)" → on nettoie le tiret mais on garde la lettre
    markdown = re.sub(
        r"^-\s+(?=[a-z]\))",
        "",
        markdown,
        flags=re.MULTILINE
    )
    return markdown

def fix_article_bullets(markdown: str) -> str:
    """
    Supprime le tiret devant les articles :
    "- Article 2 :" → "Article 2 :"
    "- Article premier :" → "Article premier :"
    """
    markdown = re.sub(
        r"^-\s+(Article\s+\d+)",
        r"\1",
        markdown,
        flags=re.IGNORECASE | re.MULTILINE
    )
    return markdown


def convert_dense_tables(markdown: str) -> str:
    """
    Convertit les tableaux avec cellules très longues en texte structuré.
    Un tableau est "dense" si une cellule dépasse 200 caractères.
    """
    lines   = markdown.splitlines()
    result  = []
    i = 0

    while i < len(lines):
        if not lines[i].startswith("|"):
            result.append(lines[i])
            i += 1
            continue

        # Collecter le tableau
        table_lines = []
        while i < len(lines) and lines[i].startswith("|"):
            table_lines.append(lines[i])
            i += 1

        # Vérifier si une cellule est très longue
        max_cell_len = max(
            len(cell.strip())
            for line in table_lines
            for cell in line.strip("|").split("|")
        )

        if max_cell_len < 200:
            # Tableau normal → garder tel quel
            result.extend(table_lines)
            continue

        # Tableau dense → convertir en texte structuré
        # Extraire en-têtes (première ligne non-séparateur)
        headers = []
        rows    = []

        for line in table_lines:
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Ligne séparateur "---"
            if all(re.match(r"^-+$", c) for c in cells if c):
                continue
            if not headers:
                headers = cells
            else:
                rows.append(cells)

        # Convertir chaque ligne en bloc de texte
        for row in rows:
            result.append("")
            for j, cell in enumerate(row):
                if not cell.strip():
                    continue
                header = headers[j] if j < len(headers) else f"Colonne {j+1}"
                header = header.strip()
                if header and header != cell:
                    result.append(f"**{header}** : {cell}")
                else:
                    result.append(cell)
            result.append("")

    return "\n".join(result)

# ── Pipeline de nettoyage ─────────────────────────────────────────────────────

def clean_markdown(markdown: str) -> str:
    """
    Pipeline complet de nettoyage dans l'ordre optimal.
    """

    # 1. Artefacts Docling basiques
    markdown = re.sub(r"<!--\s*image\s*-->", "", markdown, flags=re.IGNORECASE)

    # 2. Réassembler lignes cassées (avant toute autre opération)
    markdown = fix_broken_lines(markdown)

    # 3. Supprimer doublons d'entêtes
    markdown = remove_duplicate_headers(markdown)

    # 4. Normaliser titres d'articles
    markdown = normalize_article_headers(markdown)

    # 5. Supprimer artefacts de pagination
    markdown = remove_page_artifacts(markdown)

    # 8. Convertir tableaux denses en texte
    markdown = convert_dense_tables(markdown)

    markdown = fix_article_bullets(markdown)
    # 6. Nettoyer les puces parasites
    markdown = remove_list_bullets(markdown)

    # 7. Supprimer tableaux quasi vides
    markdown = remove_empty_tables(markdown)

    # 8. Normaliser espaces (après toutes les autres opérations)
    markdown = re.sub(r"[ \t]+", " ", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    # 9. Nettoyer fins de lignes
    lines = [line.rstrip() for line in markdown.splitlines()]

    return "\n".join(lines).strip()


# ── Stats de nettoyage ────────────────────────────────────────────────────────

def stats(original: str, cleaned: str, filename: str):
    orig_lines    = len(original.splitlines())
    clean_lines   = len(cleaned.splitlines())
    orig_chars    = len(original)
    clean_chars   = len(cleaned)
    reduction_pct = round((1 - clean_chars / orig_chars) * 100, 1) if orig_chars else 0

    print(f"  lignes : {orig_lines:5d} → {clean_lines:5d} "
          f"| chars : {orig_chars:7d} → {clean_chars:7d} "
          f"| réduction : {reduction_pct}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    md_folder = Path("pdf_chunks")

    md_files = list(md_folder.glob("*.md"))
    # Ne pas retraiter les fichiers déjà nettoyés
    md_files = [f for f in md_files if not f.stem.endswith("_clean")]

    print(f"{len(md_files)} fichiers à nettoyer\n")

    for md_file in sorted(md_files):
        print(f"Nettoyage : {md_file.name}")

        original = md_file.read_text(encoding="utf-8")
        cleaned  = clean_markdown(original)

        output_file = md_file.with_stem(md_file.stem + "_clean")
        output_file.write_text(cleaned, encoding="utf-8")

        stats(original, cleaned, md_file.name)
        print(f"  ✓ → {output_file.name}")

    print("\nTerminé.")