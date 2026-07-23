"""
state.py — Définition de l'état partagé du graphe LangGraph BCT
================================================================
BCTState est le seul objet qui transite entre tous les nœuds.
Chaque nœud lit ce dont il a besoin et retourne un dict partiel
qui met à jour l'état global.
"""

from typing import TypedDict, Optional, Dict, List


class BCTState(TypedDict):
    # ── Entrée ────────────────────────────────────────────────────────────────
    question:            str            # Question brute de l'utilisateur

    # ── Décomposition ─────────────────────────────────────────────────────────
    is_composite:         bool            # True si la question contient plusieurs sous-questions
    sub_questions:         List[str]       # Liste des sous-questions atomiques (ou [question] si simple)
    current_sub_index:     int             # Index de la sous-question en cours de traitement dans la boucle
    sub_answers:            Dict[str, str]  # {sous_question: réponse_validée} au fil de la boucle

    # ── Routing ───────────────────────────────────────────────────────────────
    intent:              Optional[str]  # "reglementaire" | "hors_scope" | "salutation"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    rewritten_query:     Optional[str]  # Question reformulée par le query rewriter
    ranked_chunks:       Optional[list] # Sortie de rerank() : list[dict]
    context:             Optional[str]  # Contexte formaté pour le LLM générateur

    # ── Génération ────────────────────────────────────────────────────────────
    generation:          Optional[str]  # Réponse brute du LLM

    # ── Grading ───────────────────────────────────────────────────────────────
    hallucination_score: Optional[str]  # "oui" | "non"
    answer_score:        Optional[str]  # "utile" | "non_utile"

    # ── Contrôle de boucle ────────────────────────────────────────────────────
    retry_count:         int            # Nombre de retries retrieval (max MAX_RETRIES)
    replan_count:        int            # Nombre de replanifications (max MAX_REPLANS)

    # ── Sortie finale ─────────────────────────────────────────────────────────
    final_answer:        Optional[str]
    sources:             Optional[list] # list[dict] au format hybrid_rag sources

    conversation_history: list  # [{"role": "user"/"assistant", "content": "..."}]
    original_question: str      # question brute avant contextualisation
    intent:  Optional[str]  # "reglementaire_precis" | "notion_generale" | "hors_scope" | "salutation"
    
def initial_state(question: str, conversation_history: list) -> BCTState:
    """Crée un état initial propre pour une nouvelle question."""
    return BCTState(
        question=question,
        is_composite=False,
        sub_questions=[question],
        current_sub_index=0,
        sub_answers={},
        intent=None,
        rewritten_query=None,
        ranked_chunks=None,
        context=None,
        generation=None,
        hallucination_score=None,
        answer_score=None,
        retry_count=0,
        replan_count=0,
        final_answer=None,
        sources=None,
        conversation_history=conversation_history,   # ← corrigé (avant : [])
        original_question=question,
        
    )