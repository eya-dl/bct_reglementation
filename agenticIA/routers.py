"""
routers.py — Fonctions de routage conditionnel du graphe LangGraph BCT
======================================================================
Contient :
  1. IntentRouter                   — classifie l'intention
  2. route_intent()                 — edge conditionnel post intent_router
  3. route_hallucination()          — edge conditionnel post hallucination_grader
  4. route_answer()                 — edge conditionnel post answer_grader
  5. route_after_store_sub_answer() — edge conditionnel post store_sub_answer

Les routers lisent l'état BCTState et retournent le nom du nœud suivant.
"""

import json
from typing import Literal

from agenticIA.config import MAX_RETRIES, MAX_REPLANS
from agenticIA.tools import INTENT_PROMPT, invoke_with_full_fallback
from agenticIA.state import BCTState


# ══════════════════════════════════════════════════════════════════════════════
#  INTENT ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class IntentRouter:
    """
    Classifie l'intention de la question utilisateur en quatre catégories :
      - "reglementaire_precis" : question nécessitant une référence légale précise
        (loi, décret, circulaire, article, délai, montant) — doit venir du corpus.
      - "notion_generale"      : définition/explication générale d'une notion
        financière liée au thème du change — peut être expliquée hors corpus
        si le retrieval échoue.
      - "salutation"           : formule de politesse ou salutation.
      - "hors_scope"           : toute autre question.

    Utilisé comme nœud LangGraph dans nodes.py.
    Utilise invoke_with_full_fallback() pour basculer automatiquement
    entre les deux clés Groq en cas de rate-limit.
    """

    def classify(self, question: str) -> str:
        result = invoke_with_full_fallback(INTENT_PROMPT, {"question": question})

        try:
            intent = json.loads(result.content).get("intent", "hors_scope")
        except (json.JSONDecodeError, AttributeError):
            intent = "hors_scope"

        if intent not in ("reglementaire_precis", "notion_generale", "salutation", "hors_scope"):
            intent = "hors_scope"

        return intent


intent_router = IntentRouter()


# ══════════════════════════════════════════════════════════════════════════════
#  EDGES CONDITIONNELS
# ══════════════════════════════════════════════════════════════════════════════

def route_intent(
    state: BCTState,
) -> Literal["query_rewriter", "hors_scope", "salutation"]:
    """
    Edge conditionnel après le nœud intent_router.
    Note : la valeur "query_rewriter" est mappée dans graph.py vers
    "decomposition" — toute question réglementaire passe d'abord
    par la décomposition avant le rewriter.

    "reglementaire_precis" et "notion_generale" suivent tous deux ce chemin
    initial : le retrieval est toujours tenté en premier (le corpus peut très
    bien contenir la notion générale demandée). La bascule vers la
    connaissance générale hors corpus ne se décide qu'en cas d'échec du
    retrieval, dans route_hallucination() / route_answer().
    """
    intent = state.get("intent", "hors_scope")
    if intent in ("reglementaire_precis", "notion_generale"):
        return "query_rewriter"
    elif intent == "salutation":
        return "salutation"
    return "hors_scope"


def route_hallucination(
    state: BCTState,
) -> Literal["answer_grader", "hybrid_retriever", "final_output", "general_knowledge_generator"]:
    """
    Edge conditionnel après le nœud hallucination_grader.

    Logique :
      - hallucination "oui" + retries restants → retry retrieval
      - hallucination "oui" + max retries atteint + intent "notion_generale"
        → bascule vers la génération par connaissance générale (hors corpus)
      - hallucination "oui" + max retries atteint + intent "reglementaire_precis"
        → sortie forcée (on ne bascule jamais hors corpus sur du réglementaire précis)
      - hallucination "non" → answer_grader
    """
    if state.get("hallucination_score") == "oui":
        retry_count = state.get("retry_count", 0)
        if retry_count < MAX_RETRIES:
            print(f"   ⚠️  Hallucination détectée → retry {retry_count}/{MAX_RETRIES}")
            return "hybrid_retriever"
        if state.get("intent") == "notion_generale":
            print("   ℹ️  Corpus insuffisant sur une notion générale → bascule connaissance générale")
            return "general_knowledge_generator"
        print(f"   ⚠️  Max retries ({MAX_RETRIES}) atteint → sortie forcée")
        return "final_output"
    return "answer_grader"


def route_answer(
    state: BCTState,
) -> Literal["store_sub_answer", "query_rewriter", "final_output", "general_knowledge_generator"]:
    """
    Edge conditionnel après le nœud answer_grader.

    Logique :
      - réponse "non_utile" + replans restants → replan (reformule + re-retrieval)
      - réponse "non_utile" + max replans atteint + intent "notion_generale"
        → bascule vers la génération par connaissance générale (hors corpus)
      - réponse "non_utile" + max replans atteint + intent "reglementaire_precis"
        → sortie forcée
      - réponse "utile" → stockage de la sous-réponse (puis boucle ou sortie)
    """
    if state.get("answer_score") == "non_utile":
        replan_count = state.get("replan_count", 0)
        if replan_count < MAX_REPLANS:
            print(f"   ⚠️  Réponse non pertinente → replan {replan_count + 1}/{MAX_REPLANS}")
            return "query_rewriter"
        if state.get("intent") == "notion_generale":
            print("   ℹ️  Corpus insuffisant sur une notion générale → bascule connaissance générale")
            return "general_knowledge_generator"
        print(f"   ⚠️  Max replans ({MAX_REPLANS}) atteint → sortie forcée")
        return "final_output"
    return "store_sub_answer"


def route_after_store_sub_answer(
    state: BCTState,
) -> Literal["select_current_subquestion", "final_output"]:
    """
    Edge conditionnel après store_sub_answer.
    S'il reste des sous-questions non traitées → on boucle
    sur la sous-question suivante, sinon → sortie finale (synthèse).
    """
    if state["current_sub_index"] < len(state["sub_questions"]):
        return "select_current_subquestion"
    return "final_output"