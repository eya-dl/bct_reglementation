import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import Counter

# ── Nettoyage Markdown Docling ────────────────────────────────────────────────

MD_PREFIX_RE = re.compile(r"^#{1,6}\s+")

def strip_md(line: str) -> str:
    return MD_PREFIX_RE.sub("", line).strip()


# ── Patterns documents ────────────────────────────────────────────────────────

DOCUMENT_PATTERNS = {
    "loi": re.compile(
        r"^LOI\s+(?:ORGANIQUE\s+)?N[°º]?\s*[\d]{2,4}[-–/][\d]{2,4}",
        re.IGNORECASE
    ),
    "decret": re.compile(
        r"^DECRET\s+(?:GOUVERNEMENTAL\s+|LOI\s+)?N[°º]?\s*[\d]{2,4}[-–/][\d]{2,4}",
        re.IGNORECASE
    ),
    "circulaire": re.compile(
        r"^CIRCULAIRE\s+(?:AUX?\s+[\w\s\.\-]+\s+)?N[°º]?\s*[\d]{2,4}[-–/][\d]{2,4}",
        re.IGNORECASE
    ),
    "avis": re.compile(
        r"^AVIS\s+(?:DE\s+CHANGE|DE\s+CHANGE\s+N[°º]?)",
        re.IGNORECASE
    ),
    "note": re.compile(
        r"^NOTE\s+(?:AUX?\s+[\w\s\.]+\s+)?N[°º]?\s*[\d]{2,4}[-–/][\d]{2,4}",
        re.IGNORECASE
    ),
    "arrete": re.compile(
        r"^(?:ARRETE|ARRETÉ)\s+(?:[\w\s]+\s+)?N[°º]?\s*[\d]{2,4}[-–/][\d]{2,4}",
        re.IGNORECASE
    ),
    "annexe": re.compile(
        r"^ANNEXE\s+(?:[IVXLC]+|[0-9]+)\s+",
        re.IGNORECASE
    ),
}

DOCUMENT_NUMBER_RE = re.compile(
    r"N[°º]?\s*([\d]{2,4}\s*[-–/]\s*[\d]{2,4}(?:\s*[-–/]\s*[\d]{2,4})?)",
    re.IGNORECASE
)

def extract_document_number(title: str) -> str:
    m = DOCUMENT_NUMBER_RE.search(title)
    if not m:
        return ""
    num = m.group(1).strip()
    num = re.sub(r"\s*[-–]\s*", "-", num)
    return num


# ── Patterns structurels ──────────────────────────────────────────────────────

STRUCTURE_PATTERNS = {
    "partie": re.compile(
        r"^(?:PREMIERE|DEUXIEME|TROISIEME|[A-Z]+IEME?)\s+PARTIE\b",
        re.IGNORECASE
    ),
    "titre": re.compile(
        r"^TITRE\s+(?:PREMIER|[IVXLC]+|\d+)\s*(?:[-–:.\s].*)?$",
        re.IGNORECASE
    ),
    "chapitre": re.compile(
        r"^CHAPITRE\s+(?:PREMIER|[IVXLC]+|\d+)\s*(?:[-–:.\s].*)?$",
        re.IGNORECASE
    ),
    "section_num": re.compile(
        r"^SECTION\s+\d+\s*(?:[-–:.].*)?$",
        re.IGNORECASE
    ),
    "paragraphe": re.compile(
        r"^PARAGRAPHE\s+\d+\s*(?:[-–:.].*)?$",
        re.IGNORECASE
    ),
    "section_rom": re.compile(
        r"^(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\s*[-–]\s+[A-ZÀÂÇÈÉÊËÎÏÔÙÛÜ]",
        re.IGNORECASE
    ),
    "sous_section": re.compile(
        r"^[A-Z]\s*[)\-]\s+[A-ZÀÂÇÈÉÊËÎÏÔÙÛÜ]"
    ),
    "point_num": re.compile(
        r"^\d+°\)\s+\S"
    ),
    "titre_libre": re.compile(
        r"^(?:REGLES|DISPOSITIONS|CONDITIONS|MODALITES|OBLIGATIONS|DEFINITIONS)"
        r"\s+(?:DE\s+|D[EU]\s+|GENERALES?|COMMUNES?|DIVERSES?|RELATIVES?\s)",
        re.IGNORECASE
    ),
}

FLUSH_LEVELS = {
    "partie", "titre", "chapitre",
    "section_num", "section_rom", "paragraphe", "titre_libre"
}

PARENT_LEVELS = {"chapitre", "section", "paragraphe"}

MAX_PARENT_TOKENS = 1200

ARTICLE_RE = re.compile(
    r"(Article\s+\d+\s*(?:er|bis|ter|nouveau|\d+)?\s*(?:\(nouveau\))?)",
    re.IGNORECASE
)

# ────────────────────────────────────────────────────────────────────────────
# ✅ FIX 3 — Pattern pour détecter les items de liste (droits d'auteur, etc.)
# ────────────────────────────────────────────────────────────────────────────
LIST_ITEM_RE = re.compile(
    r"^(?:\d+°?\s*[)\.]|\-\s|•\s|[a-z]\)\s)",
    re.MULTILINE
)


# ── Contexte hiérarchique ─────────────────────────────────────────────────────

class HierarchyContext:
    LEVELS = [
        "document", "partie", "titre", "chapitre",
        "section", "paragraphe", "sous_section", "point_num"
    ]

    def __init__(self):
        self._ctx = {l: "" for l in self.LEVELS}

    def set(self, level: str, value: str):
        norm = "section" if level in ("section_num", "section_rom", "titre_libre") else level
        if norm not in self.LEVELS:
            return
        self._ctx[norm] = value
        idx = self.LEVELS.index(norm)
        for l in self.LEVELS[idx + 1:]:
            self._ctx[l] = ""

    def get(self, level: str) -> str:
        return self._ctx.get(level, "")

    def breadcrumb(self) -> str:
        return " > ".join(
            v[:80] for k, v in self._ctx.items()
            if v and k != "document"
        )

    def snapshot(self) -> dict:
        return dict(self._ctx)

    def parent_level(self) -> str:
        for level in ("paragraphe", "section", "chapitre"):
            if self._ctx.get(level):
                return level
        return ""


# ── Chunk ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:        str
    chunk_type:      str
    level:           str
    text:            str
    document_type:   str
    document_title:  str
    document_number: str
    breadcrumb:      str
    ctx_titre:       str = ""
    ctx_chapitre:    str = ""
    ctx_section:     str = ""
    ctx_document:    str = ""
    parent_id:       str = ""
    article_title:   str = ""
    part_num:        int = 0

    @property
    def token_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "chunk_id":        self.chunk_id,
            "chunk_type":      self.chunk_type,
            "level":           self.level,
            "document_type":   self.document_type,
            "document_title":  self.document_title,
            "document_number": self.document_number,
            "breadcrumb":      self.breadcrumb,
            "titre":           self.ctx_titre,
            "chapitre":        self.ctx_chapitre,
            "section":         self.ctx_section,
            "article_title":   self.article_title,
            "parent_id":       self.parent_id,
            "token_count":     self.token_count,
        }

    def __repr__(self):
        return (f"Chunk(id={self.chunk_id}, type={self.chunk_type}, "
                f"level={self.level}, tokens={self.token_count})")


# ── Parser ────────────────────────────────────────────────────────────────────

class BCTParser:

    def __init__(self, doc_name: str, max_child_tokens: int = 500):
        self.doc_name         = doc_name
        self.max_child_tokens = max_child_tokens
        self.ctx              = HierarchyContext()
        self._pid = 0
        self._cid = 0
        self._buffer:  list[str] = []
        self.parents:  list[Chunk] = []
        self.children: list[Chunk] = []
        self._doc_type   = "inconnu"
        self._doc_title  = ""
        self._doc_number = ""

        # ────────────────────────────────────────────────────────────────────
        # ✅ FIX 1 — Tracker l'article courant pour les chunks point_num
        # ────────────────────────────────────────────────────────────────────
        self._current_article: str = ""

    def _new_pid(self) -> str:
        self._pid += 1
        return f"{self.doc_name}__P__{self._pid:04d}"

    def _new_cid(self) -> str:
        self._cid += 1
        return f"{self.doc_name}__C__{self._cid:04d}"

    def _base(self, breadcrumb: str) -> dict:
        snap = self.ctx.snapshot()
        return {
            "document_type":   self._doc_type,
            "document_title":  self._doc_title,
            "document_number": self._doc_number,
            "breadcrumb":      breadcrumb,
            "ctx_titre":       snap.get("titre", ""),
            "ctx_chapitre":    snap.get("chapitre", ""),
            "ctx_section":     snap.get("section", ""),
            "ctx_document":    snap.get("document", ""),
        }

    # ── Découpe articles ──────────────────────────────────────────────────────

    def _split_articles(self, text: str) -> list[tuple[str, str]]:
        matches = list(ARTICLE_RE.finditer(text))
        if not matches:
            return []
        results = []
        intro = text[:matches[0].start()].strip()
        if len(intro.split()) > 5:
            results.append(("__intro__", intro))
        for i, match in enumerate(matches):
            title = match.group().strip()
            start = match.end()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body  = text[start:end].strip()
            if body:
                results.append((title, body))
        return results

    # ── Découpe sections romaines ─────────────────────────────────────────────

    def _split_roman_sections(self, text: str) -> list[tuple[str, str]]:
        pattern = re.compile(
            r"^((?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\s*[-–]\s+[^\n]+)",
            re.MULTILINE | re.IGNORECASE
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return []
        results = []
        intro = text[:matches[0].start()].strip()
        if len(intro.split()) > 5:
            results.append(("__intro__", intro))
        for i, match in enumerate(matches):
            sec_title = match.group().strip()
            start     = match.end()
            end       = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body      = text[start:end].strip()
            if sec_title or body:
                results.append((sec_title, body))
        return results

    # ────────────────────────────────────────────────────────────────────────
    # ✅ FIX 3 — Découpe les gros blocs de liste (catégorie K décret 77-608)
    # ────────────────────────────────────────────────────────────────────────

    def _split_list_block(
        self, header: str, body: str,
        pid: str, breadcrumb: str, base: dict
    ):
        """
        Découpe un bloc de liste (type 'K - OPERATIONS A CARACTERE GENERAL')
        en sous-chunks thématiques quand le corps est trop long.
        Stratégie : regrouper les items consécutifs jusqu'à max_child_tokens.
        """
        lines  = body.splitlines()
        buffer = header  # démarre avec le titre de la section
        part   = 1

        for line in lines:
            candidate = buffer + "\n" + line
            if len(candidate.split()) > self.max_child_tokens and len(buffer.split()) > 5:
                self.children.append(Chunk(
                    chunk_id      = self._new_cid(),
                    chunk_type    = "child",
                    level         = "list_block",
                    text          = buffer,
                    parent_id     = pid,
                    article_title = f"{header} (partie {part})",
                    part_num      = part,
                    **base,
                ))
                part  += 1
                buffer = header + "\n" + line   # reset avec le titre pour garder le contexte
            else:
                buffer = candidate

        if len(buffer.split()) > 5:
            self.children.append(Chunk(
                chunk_id      = self._new_cid(),
                chunk_type    = "child",
                level         = "list_block",
                text          = buffer,
                parent_id     = pid,
                article_title = f"{header} (partie {part})",
                part_num      = part,
                **base,
            ))

    # ── Découpe article long ──────────────────────────────────────────────────

    def _split_long_article(
        self, art_title: str, art_body: str,
        pid: str, breadcrumb: str, base: dict
    ):
        alineas  = [p.strip() for p in art_body.split("\n\n") if p.strip()]
        buffer   = art_title
        part_num = 1
        for alinea in alineas:
            candidate = buffer + "\n" + alinea
            if len(candidate.split()) > self.max_child_tokens:
                if len(buffer.split()) > 5:
                    self.children.append(Chunk(
                        chunk_id      = self._new_cid(),
                        chunk_type    = "child",
                        level         = "article_part",
                        text          = buffer,
                        parent_id     = pid,
                        article_title = art_title,
                        part_num      = part_num,
                        **base,
                    ))
                    part_num += 1
                buffer = art_title + "\n" + alinea  # ✅ FIX : réinsérer le titre pour chaque partie
            else:
                buffer = candidate
        if len(buffer.split()) > 5:
            self.children.append(Chunk(
                chunk_id      = self._new_cid(),
                chunk_type    = "child",
                level         = "article_part",
                text          = buffer,
                parent_id     = pid,
                article_title = art_title,
                part_num      = part_num,
                **base,
            ))

    # ── Flush ─────────────────────────────────────────────────────────────────

    def _flush(self):
        text = "\n".join(self._buffer).strip()
        self._buffer = []
        if not text or len(text.split()) < 8:
            return

        breadcrumb   = self.ctx.breadcrumb()
        base         = self._base(breadcrumb)
        parent_level = self.ctx.parent_level()
        pid          = None

        if parent_level:
            parent_text = text
            if len(text.split()) > MAX_PARENT_TOKENS:
                parent_text = (
                    " ".join(text.split()[:MAX_PARENT_TOKENS])
                    + "\n[...tronqué]"
                )
            pid = self._new_pid()
            self.parents.append(Chunk(
                chunk_id   = pid,
                chunk_type = "parent",
                level      = parent_level,
                text       = parent_text,
                **base,
            ))

        articles = self._split_articles(text)

        if articles:
            for art_title, art_body in articles:
                is_intro  = (art_title == "__intro__")
                full_text = art_body if is_intro else f"{art_title}\n{art_body}"
                level     = "intro" if is_intro else "article"

                # ────────────────────────────────────────────────────────────
                # ✅ FIX 1 — Mettre à jour l'article courant
                # ────────────────────────────────────────────────────────────
                if not is_intro:
                    self._current_article = art_title

                if len(full_text.split()) <= self.max_child_tokens:
                    self.children.append(Chunk(
                        chunk_id      = self._new_cid(),
                        chunk_type    = "child",
                        level         = level,
                        text          = full_text,
                        parent_id     = pid or "",
                        article_title = "" if is_intro else art_title,
                        **base,
                    ))
                else:
                    self._split_long_article(
                        art_title, art_body, pid or "", breadcrumb, base
                    )
        else:
            sections = self._split_roman_sections(text)
            if sections:
                for sec_title, sec_body in sections:
                    is_intro  = (sec_title == "__intro__")
                    full_text = sec_body if is_intro else f"{sec_title}\n{sec_body}"
                    if not full_text.strip() or len(full_text.split()) < 5:
                        continue
                    if len(full_text.split()) <= self.max_child_tokens:
                        self.children.append(Chunk(
                            chunk_id      = self._new_cid(),
                            chunk_type    = "child",
                            level         = "intro" if is_intro else "section_content",
                            text          = full_text,
                            parent_id     = pid or "",
                            article_title = "" if is_intro else sec_title,
                            **base,
                        ))
                    else:
                        # ────────────────────────────────────────────────────
                        # ✅ FIX 3 — Utiliser _split_list_block si liste dense
                        # ────────────────────────────────────────────────────
                        list_items = LIST_ITEM_RE.findall(sec_body)
                        if len(list_items) >= 3:
                            self._split_list_block(
                                sec_title, sec_body, pid or "", breadcrumb, base
                            )
                        else:
                            paras    = [p.strip() for p in sec_body.split("\n\n") if p.strip()]
                            buffer_p = sec_title
                            for para in paras:
                                candidate = buffer_p + "\n" + para
                                if len(candidate.split()) > self.max_child_tokens:
                                    if len(buffer_p.split()) > 5:
                                        self.children.append(Chunk(
                                            chunk_id      = self._new_cid(),
                                            chunk_type    = "child",
                                            level         = "section_content",
                                            text          = buffer_p,
                                            parent_id     = pid or "",
                                            article_title = sec_title,
                                            **base,
                                        ))
                                    buffer_p = sec_title + "\n" + para  # ✅ FIX : réinsérer le titre
                                else:
                                    buffer_p = candidate
                            if len(buffer_p.split()) > 5:
                                self.children.append(Chunk(
                                    chunk_id      = self._new_cid(),
                                    chunk_type    = "child",
                                    level         = "section_content",
                                    text          = buffer_p,
                                    parent_id     = pid or "",
                                    article_title = sec_title,
                                    **base,
                                ))
            else:
                # ────────────────────────────────────────────────────────────
                # ✅ FIX 1 — Chunk paragraph : inclure le titre de l'article
                #            courant pour les chunks point_num (PPR, etc.)
                # ────────────────────────────────────────────────────────────
                ctx_article = self._current_article
                enriched    = (
                    f"[Contexte : {ctx_article}]\n{text}"
                    if ctx_article else text
                )
                # ────────────────────────────────────────────────────────────
                # ✅ FIX 3 — Bloc de liste trop grand → _split_list_block
                # ────────────────────────────────────────────────────────────
                if len(enriched.split()) > self.max_child_tokens:
                    list_items = LIST_ITEM_RE.findall(text)
                    header_line = text.splitlines()[0] if text.splitlines() else ""
                    body_rest   = "\n".join(text.splitlines()[1:])
                    if len(list_items) >= 3 and body_rest:
                        self._split_list_block(
                            header_line, body_rest, pid or "", breadcrumb, base
                        )
                    else:
                        self.children.append(Chunk(
                            chunk_id      = self._new_cid(),
                            chunk_type    = "child",
                            level         = "paragraph",
                            text          = enriched,
                            parent_id     = pid or "",
                            article_title = ctx_article,
                            **base,
                        ))
                else:
                    self.children.append(Chunk(
                        chunk_id      = self._new_cid(),
                        chunk_type    = "child",
                        level         = "paragraph",
                        text          = enriched,
                        parent_id     = pid or "",
                        article_title = ctx_article,
                        **base,
                    ))

    # ── Parse ─────────────────────────────────────────────────────────────────

    def parse(self, text: str) -> tuple[list[Chunk], list[Chunk]]:

        for raw_line in text.splitlines():
            clean = strip_md(raw_line)
            if not clean:
                continue

            doc_type = None
            for dtype, pat in DOCUMENT_PATTERNS.items():
                if pat.match(clean):
                    doc_type = dtype
                    break

            if doc_type:
                if doc_type == "annexe":
                    self._flush()
                    continue
                self._flush()
                # ────────────────────────────────────────────────────────────
                # ✅ FIX 1 — Reset l'article courant à chaque nouveau document
                # ────────────────────────────────────────────────────────────
                self._current_article = ""
                self._doc_type   = doc_type
                self._doc_title  = clean
                self._doc_number = extract_document_number(clean)
                self.ctx.set("document", clean)
                continue

            level = None
            for lvl, pat in STRUCTURE_PATTERNS.items():
                if pat.match(clean):
                    level = lvl
                    break

            if level in FLUSH_LEVELS:
                self._flush()
                # ────────────────────────────────────────────────────────────
                # ✅ FIX 1 — Reset l'article courant à chaque nouvelle section
                # ────────────────────────────────────────────────────────────
                self._current_article = ""
                self.ctx.set(level, clean)
                self._buffer.append(clean)
                continue

            if level in ("sous_section", "point_num"):
                self._flush()
                self.ctx.set(level, clean)
                self._buffer.append(clean)
                continue

            self._buffer.append(clean)

        self._flush()
        return self.parents, self.children


# ── Pipeline ──────────────────────────────────────────────────────────────────

def chunk_all_documents(md_folder: Path) -> tuple[list[Chunk], list[Chunk]]:
    md_files = sorted(
        md_folder.glob("*_clean.md"),
        key=lambda f: int(re.search(r"part_(\d+)_", f.stem).group(1))
    )

    print("── Fichiers à fusionner (ordre des pages) ──")
    for f in md_files:
        print(f"  {f.name}")

    full_text = "\n".join(f.read_text(encoding="utf-8") for f in md_files)

    print(f"\nChunking du corpus complet ({len(md_files)} fichiers fusionnés)")
    parser = BCTParser("corpus_bct")
    all_parents, all_children = parser.parse(full_text)

    print(f"\n{'─'*50}")
    print(f"TOTAL : {len(all_parents)} parents | {len(all_children)} enfants")

    print("\n── Distribution types documents ──")
    for dt, n in Counter(c.document_type for c in all_children).most_common():
        print(f"  {dt:20s} : {n}")

    print("\n── Distribution niveaux enfants ──")
    for lvl, n in Counter(c.level for c in all_children).most_common():
        print(f"  {lvl:20s} : {n}")

    print("\n── Numéros extraits (top 15) ──")
    nums = Counter(
        f"{c.document_type}/{c.document_number}"
        for c in all_children
        if c.document_number
    )
    for key, n in nums.most_common(15):
        print(f"  {key:30s} : {n} chunks")

    # ── Vérification des fixes ────────────────────────────────────────────────
    print("\n── ✅ Vérification Fix 1 (article_title dans chunks paragraph) ──")
    paragraph_with_ctx = [c for c in all_children
                          if c.level == "paragraph" and c.article_title]
    paragraph_without  = [c for c in all_children
                          if c.level == "paragraph" and not c.article_title]
    print(f"  Paragraphs avec article_title : {len(paragraph_with_ctx)}")
    print(f"  Paragraphs sans article_title : {len(paragraph_without)}")

    print("\n── ✅ Vérification Fix 3 (list_block chunks) ──")
    list_blocks = [c for c in all_children if c.level == "list_block"]
    print(f"  Chunks list_block créés : {len(list_blocks)}")
    for c in list_blocks[:3]:
        print(f"    {c.article_title} | {c.token_count} tokens")

    inconnus = [c for c in all_children if c.document_type == "inconnu"]
    if inconnus:
        print(f"\n── Échantillon chunks 'inconnu' ({len(inconnus)} total) ──")
        for c in inconnus[:5]:
            print(f"  breadcrumb : {c.breadcrumb}")
            print(f"  texte[100] : {c.text[:100]}")
            print()

    return all_parents, all_children


if __name__ == "__main__":
    md_folder = Path("pdf_chunks")
    chunk_all_documents(md_folder)