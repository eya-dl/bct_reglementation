# decomposer.py

import re
from typing import List

# Mots interrogatifs typiques du domaine réglementaire
INTERROGATIVE_MARKERS = [
    r"quel(?:le)?s?", r"comment", r"pourquoi", r"où", r"quand", 
    r"qui", r"est-ce que", r"combien", r"dans quel(?:le)?s? cas"
]

# Connecteurs qui PEUVENT introduire une sous-question (à valider ensuite)
SPLIT_CONNECTORS = [
    r"\bet\b", r"\bainsi que\b", r"\bde plus\b", r"\bpar ailleurs\b",
    r"\bégalement\b", r"\bde même\b", r"\?\s*(?=[A-ZÉÈ])"  # nouvelle question après un "?"
]

INTERROGATIVE_PATTERN = re.compile(
    r"(" + "|".join(INTERROGATIVE_MARKERS) + r")", re.IGNORECASE
)

def count_interrogative_structures(text: str) -> int:
    """Compte les structures interrogatives distinctes dans le texte."""
    return len(INTERROGATIVE_PATTERN.findall(text))


def is_composite(query: str) -> bool:
    """
    Une requête est composite si elle contient au moins 2 structures 
    interrogatives séparées par un connecteur de rupture, OU plusieurs '?'.
    """
    # Cas 1 : plusieurs points d'interrogation = plusieurs questions explicites
    if query.count("?") >= 2:
        return True

    # Cas 2 : au moins 2 mots interrogatifs dans le texte
    interrogative_count = count_interrogative_structures(query)
    if interrogative_count < 2:
        return False

    # Cas 3 : vérifier qu'il y a bien un connecteur de rupture ENTRE deux marqueurs interrogatifs
    positions = [m.start() for m in INTERROGATIVE_PATTERN.finditer(query)]
    for connector in SPLIT_CONNECTORS:
        for m in re.finditer(connector, query, re.IGNORECASE):
            conn_pos = m.start()
            # y a-t-il un marqueur interrogatif avant ET après le connecteur ?
            before = any(p < conn_pos for p in positions)
            after = any(p > conn_pos for p in positions)
            if before and after:
                return True

    return False


def split_into_subquestions(query: str) -> List[str]:
    """
    Découpe une requête composite en sous-questions atomiques.
    Stratégie : split sur les connecteurs de rupture VALIDÉS (entre deux 
    structures interrogatives), sinon split sur '?'.
    """
    if not is_composite(query):
        return [query]

    # Priorité 1 : split sur '?' si plusieurs présents
    if query.count("?") >= 2:
        parts = [p.strip() + "?" for p in query.split("?") if p.strip()]
        return _clean_subquestions(parts)

    # Priorité 2 : split sur le connecteur trouvé entre deux marqueurs interrogatifs
    positions = [m.start() for m in INTERROGATIVE_PATTERN.finditer(query)]
    for connector in SPLIT_CONNECTORS:
        for m in re.finditer(connector, query, re.IGNORECASE):
            conn_pos = m.start()
            before = any(p < conn_pos for p in positions)
            after = any(p > conn_pos for p in positions)
            if before and after:
                left = query[:conn_pos].strip(" ,.")
                right = query[m.end():].strip(" ,.")
                return _clean_subquestions([left, right])

    return [query]


def _clean_subquestions(parts: List[str]) -> List[str]:
    """Nettoie et filtre les sous-questions trop courtes ou vides."""
    cleaned = []
    for p in parts:
        p = p.strip()
        if len(p) > 5:  # évite les fragments vides ou résiduels
            if not p.endswith("?"):
                p += " ?"
            cleaned.append(p)
    return cleaned if cleaned else [parts[0]] if parts else []