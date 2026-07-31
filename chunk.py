import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import Counter

# ── Nettoyage Markdown Docling ────────────────────────────────────────────────

MD_PREFIX_RE = re.compile(r"^#{1,6}\s+")
# ────────────────────────────────────────────────────────────────────────────
# ✅ FIX BCT — strip_md retire aussi le gras Markdown ("**SECTION II**")
# résiduel quand le pipeline Docling émet des titres en gras plutôt qu'en #.
# Sans ça, les regex structurelles (^SECTION..., ^Article...) échouent car
# la ligne commence par "**".
# ────────────────────────────────────────────────────────────────────────────
MD_BOLD_RE = re.compile(r"^\*\*(.+)\*\*$")


def strip_md(line: str) -> str:
    line = MD_PREFIX_RE.sub("", line).strip()
    m = MD_BOLD_RE.match(line)
    if m:
        line = m.group(1).strip()
    return line


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

# ────────────────────────────────────────────────────────────────────────────
# ✅ FIX BCT-5 — le tiret devient optionnel : les Avis de Change sont
# numérotés "n°3", "n°5" (un seul nombre, sans séparateur), alors que
# l'ancien pattern exigeait deux groupes de chiffres séparés par -/–//.
# ────────────────────────────────────────────────────────────────────────────
DOCUMENT_NUMBER_RE = re.compile(
    r"N[°º]?\s*(\d{1,4}(?:\s*[-–/]\s*\d{1,4}){0,2})",
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
    # ────────────────────────────────────────────────────────────────────
    # ✅ FIX BCT-1 — accepte désormais chiffres arabes ET romains
    # ("SECTION 1" et "SECTION II"), et ne réclame plus de titre sur la
    # même ligne (le titre arrive en général sur la ligne suivante dans
    # ce corpus). Le rattachement du titre est géré au niveau du parse()
    # via un lookahead (_looks_like_title_line), pas dans la regex.
    # ────────────────────────────────────────────────────────────────────
    "section_num": re.compile(
        r"^SECTION\s+(?:\d+|[IVXLC]+)\b\s*(?:[-–:.]\s*.*)?$",
        re.IGNORECASE
    ),
    "paragraphe": re.compile(
        r"^PARAGRAPHE\s+\d+\s*(?:[-–:.].*)?$",
        re.IGNORECASE
    ),
    # NB : section_rom reste dédié aux lignes de la forme "II - TITRE"
    # (roman + tiret + texte sur la même ligne), un format distinct de
    # "SECTION II" pris en charge par section_num ci-dessus.
    "section_rom": re.compile(
        r"^(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\s*[-–]\s+[A-ZÀÂÇÈÉÊËÎÏÔÙÛÜ]",
        re.IGNORECASE
    ),
    "sous_section": re.compile(
        r"^[A-Z]\s*[)\-]\s+[A-ZÀÂÇÈÉÊËÎÏÔÙÛÜ]"
    ),
    # ────────────────────────────────────────────────────────────────────
    # ✅ FIX BCT-3 — la parenthèse fermante devient optionnelle : le
    # corpus contient à la fois "1°) Les..." et "1° Les..." (sans ")").
    # ────────────────────────────────────────────────────────────────────
    "point_num": re.compile(
        r"^\d+°\)?\s+\S"
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

# ────────────────────────────────────────────────────────────────────────────
# ✅ Fix code smell — source unique de vérité pour la hiérarchie parente.
# Avant : PARENT_LEVELS était défini ici mais jamais utilisé, tandis que
# HierarchyContext.parent_level() avait la même liste codée en dur en
# double, avec un risque de divergence silencieuse. Maintenant
# parent_level() consomme directement cette constante (voir plus bas).
# L'ordre est important : priorité paragraphe > section > chapitre.
# ────────────────────────────────────────────────────────────────────────────
PARENT_LEVELS = ("paragraphe", "section", "chapitre")

MAX_PARENT_TOKENS = 1200

# ────────────────────────────────────────────────────────────────────────────
# ✅ FIX BCT-2 — "Article premier" (sans chiffre) est maintenant reconnu,
# et la liste des suffixes ordinaux latins est complète (bis/ter/quater/
# quinquies/sexto/septies/octies), là où l'original s'arrêtait à "ter" et
# retombait sur "\d+" (ce qui de toute façon ne matchait pas "quater" etc.
# car ce ne sont pas des chiffres).
# ────────────────────────────────────────────────────────────────────────────
ARTICLE_RE = re.compile(
    r"(Article\s+(?:premier|\d+)\s*"
    r"(?:bis|ter|quater|quinquies|sexto|septies|octies|nouveau)?\s*"
    r"(?:\(nouveau\))?)",
    re.IGNORECASE
)

# ────────────────────────────────────────────────────────────────────────────
# ✅ FIX 3 (existant) — Pattern pour détecter les items de liste (droits d'auteur, etc.)
# ────────────────────────────────────────────────────────────────────────────
LIST_ITEM_RE = re.compile(
    r"^(?:\d+°?\s*[)\.]|\-\s|•\s|[a-z]\)\s)",
    re.MULTILINE
)

# ────────────────────────────────────────────────────────────────────────────
# ✅ FIX BCT-4 — overlap entre chunks. Nombre de mots de "queue" repris du
# chunk précédent au début du chunk suivant, quand un article/section/liste
# est découpé en plusieurs parties. Évite qu'une phrase, un renvoi ou une
# donnée chiffrée à cheval sur la coupure ne soit perdue sans filet.
# ────────────────────────────────────────────────────────────────────────────
OVERLAP_WORDS = 25
OVERLAP_MARKER = "[...suite]"


def _overlap_tail(text: str, n_words: int = OVERLAP_WORDS) -> str:
    """Retourne les derniers n_words mots d'un texte (pour reprise en tête du chunk suivant)."""
    words = text.split()
    if len(words) <= n_words:
        return text.strip()
    return " ".join(words[-n_words:]).strip()


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
        # ✅ Fix code smell : utilise désormais PARENT_LEVELS (source unique)
        for level in PARENT_LEVELS:
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

        # ✅ FIX 1 (existant) — Tracker l'article courant pour les chunks point_num
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

    # ────────────────────────────────────────────────────────────────────────
    # ✅ FIX BCT-1 — détecte si une ligne ressemble à un titre de continuation
    # (le titre de section/chapitre/titre qui suit sur la ligne d'après),
    # pour pouvoir le rattacher au header ("SECTION II" + ligne suivante
    # "MONTANT DE L'ALLOCATION" -> "SECTION II - MONTANT DE L'ALLOCATION").
    # Heuristique : ligne courte, majoritairement en MAJUSCULES, qui n'est
    # elle-même ni un en-tête de document, ni un autre marqueur structurel,
    # ni un début d'article.
    # ────────────────────────────────────────────────────────────────────────
    def _looks_like_title_line(self, line: str) -> bool:
        if not line or len(line.split()) > 12:
            return False
        for pat in DOCUMENT_PATTERNS.values():
            if pat.match(line):
                return False
        for pat in STRUCTURE_PATTERNS.values():
            if pat.match(line):
                return False
        if ARTICLE_RE.match(line):
            return False
        letters = [c for c in line if c.isalpha()]
        if not letters:
            return False
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        return upper_ratio > 0.7

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
    # ✅ FIX 3 (existant) — Découpe les gros blocs de liste
    # ✅ FIX BCT-4 — chaque partie > 1 reprend en tête un overlap du texte
    #    de fin de la partie précédente, en plus du header.
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
                overlap = _overlap_tail(buffer)
                part  += 1
                buffer = header + "\n" + OVERLAP_MARKER + " " + overlap + "\n" + line
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
    # ✅ FIX BCT-4 — overlap ajouté entre les parties d'un même article.

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
                overlap = _overlap_tail(buffer)
                buffer = art_title + "\n" + OVERLAP_MARKER + " " + overlap + "\n" + alinea
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

                # ✅ FIX 1 (existant) — Mettre à jour l'article courant
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
                        # ✅ FIX 3 (existant) — Utiliser _split_list_block si liste dense
                        list_items = LIST_ITEM_RE.findall(sec_body)
                        if len(list_items) >= 3:
                            self._split_list_block(
                                sec_title, sec_body, pid or "", breadcrumb, base
                            )
                        else:
                            # ✅ FIX BCT-4 — overlap entre parties de paragraphes
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
                                    overlap  = _overlap_tail(buffer_p)
                                    buffer_p = sec_title + "\n" + OVERLAP_MARKER + " " + overlap + "\n" + para
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
                # ✅ FIX 1 (existant) — Chunk paragraph : inclure le titre de l'article
                #    courant pour les chunks point_num (PPR, etc.)
                ctx_article = self._current_article
                enriched    = (
                    f"[Contexte : {ctx_article}]\n{text}"
                    if ctx_article else text
                )
                # ✅ FIX 3 (existant) — Bloc de liste trop grand → _split_list_block
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

        # ✅ FIX BCT-1 — liste indexable (au lieu d'un simple for) pour
        # permettre le lookahead sur la ligne suivante (titre de section
        # rattaché sur la ligne d'après).
        lines = [strip_md(raw) for raw in text.splitlines()]
        lines = [l for l in lines if l]
        n = len(lines)
        i = 0

        while i < n:
            clean = lines[i]

            doc_type = None
            for dtype, pat in DOCUMENT_PATTERNS.items():
                if pat.match(clean):
                    doc_type = dtype
                    break

            if doc_type:
                if doc_type == "annexe":
                    self._flush()
                    i += 1
                    continue
                self._flush()
                # ✅ FIX 1 (existant) — Reset l'article courant à chaque nouveau document
                self._current_article = ""
                self._doc_type   = doc_type
                self._doc_title  = clean
                self._doc_number = extract_document_number(clean)
                self.ctx.set("document", clean)
                i += 1
                continue

            level = None
            for lvl, pat in STRUCTURE_PATTERNS.items():
                if pat.match(clean):
                    level = lvl
                    break

            if level in FLUSH_LEVELS:
                self._flush()
                # ✅ FIX 1 (existant) — Reset l'article courant à chaque nouvelle section
                self._current_article = ""

                # ────────────────────────────────────────────────────────────
                # ✅ FIX BCT-1 — rattache le titre porté par la ligne suivante
                # (ex. "SECTION II" puis "MONTANT DE L'ALLOCATION") au
                # breadcrumb/contexte, au lieu de laisser ctx_section figé
                # sur la valeur de la section précédente.
                # ────────────────────────────────────────────────────────────
                header_value = clean
                consumed_next = False
                if i + 1 < n and self._looks_like_title_line(lines[i + 1]):
                    nxt = lines[i + 1]
                    header_value = f"{clean} - {nxt}"
                    consumed_next = True

                self.ctx.set(level, header_value)
                self._buffer.append(header_value)

                i += 2 if consumed_next else 1
                continue

            if level in ("sous_section", "point_num"):
                self._flush()
                self.ctx.set(level, clean)
                self._buffer.append(clean)
                i += 1
                continue

            self._buffer.append(clean)
            i += 1

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

    # ────────────────────────────────────────────────────────────────────────
    # ✅ Vérification FIX BCT-1/2/3/5 — comptages rapides pour valider les
    # nouveaux patterns sur le corpus réel.
    # ────────────────────────────────────────────────────────────────────────
    print("\n── ✅ Vérification FIX BCT-2 (Article premier / suffixes) ──")
    art_premier = [c for c in all_children if c.article_title and
                   re.match(r"^Article\s+premier", c.article_title, re.IGNORECASE)]
    art_suffixes = [c for c in all_children if c.article_title and
                     re.search(r"\b(bis|ter|quater|quinquies|sexto|septies|octies)\b",
                                c.article_title, re.IGNORECASE)]
    print(f"  Chunks 'Article premier' : {len(art_premier)}")
    print(f"  Chunks avec suffixe ordinal (bis/ter/quater/...) : {len(art_suffixes)}")

    print("\n── ✅ Vérification FIX BCT-5 (numéro Avis de Change sans tiret) ──")
    avis_avec_num = [c for c in all_children
                      if c.document_type == "avis" and c.document_number]
    avis_sans_num = [c for c in all_children
                      if c.document_type == "avis" and not c.document_number]
    print(f"  Avis de Change AVEC document_number : {len(avis_avec_num)}")
    print(f"  Avis de Change SANS document_number : {len(avis_sans_num)}")

    print("\n── ✅ Vérification FIX BCT-4 (overlap) ──")
    overlap_chunks = [c for c in all_children if OVERLAP_MARKER in c.text]
    print(f"  Chunks portant un overlap : {len(overlap_chunks)}")

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