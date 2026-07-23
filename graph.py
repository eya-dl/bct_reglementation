"""
graph.py — Construction et compilation du graphe LangGraph BCT
================================================================
Assemble tous les nœuds (nodes.py) et tous les edges conditionnels
(routers.py) en un StateGraph compilé.

Ce fichier ne contient AUCUNE logique métier — uniquement le câblage.
"""

from langgraph.graph import StateGraph, END

from state import BCTState
from nodes import (
    node_contextualize,
    node_general_knowledge,
    node_intent_router,
    node_decomposition,
    node_select_current_subquestion,
    node_query_rewriter,
    node_hybrid_retriever,
    node_cross_reference,
    node_response_generator,
    node_hallucination_grader,
    node_answer_grader,
    node_store_sub_answer,
    node_final_output,
    node_hors_scope,
    node_salutation,
)
from routers import (
    route_intent,
    route_hallucination,
    route_answer,
    route_after_store_sub_answer,
)


def build_bct_graph():
    """
    Construit et compile le graphe LangGraph du pipeline agentique BCT.

    Flux nominal :
        contextualize → intent_router → decomposition → select_current_subquestion →
        query_rewriter → hybrid_retriever → cross_reference → response_generator →
        hallucination_grader → answer_grader → store_sub_answer →
        [boucle sous-question suivante OU final_output]

    Boucles :
        hallucination_grader --(oui, retry<2)--> hybrid_retriever
        hallucination_grader --(oui, retry épuisé, intent=notion_generale)--> general_knowledge_generator
        answer_grader --(non_utile, replan<2)--> query_rewriter
        answer_grader --(non_utile, replan épuisé, intent=notion_generale)--> general_knowledge_generator
        store_sub_answer --(sous-questions restantes)--> select_current_subquestion

    Branches courtes :
        intent_router --(hors_scope)--> hors_scope --> END
        intent_router --(salutation)--> salutation --> END
        general_knowledge_generator --> store_sub_answer
    """
    g = StateGraph(BCTState)

    # ── Nœuds ────────────────────────────────────────────────────────────────
    g.add_node("contextualize",              node_contextualize)
    g.add_node("intent_router",              node_intent_router)
    g.add_node("decomposition",              node_decomposition)
    g.add_node("select_current_subquestion", node_select_current_subquestion)
    g.add_node("query_rewriter",             node_query_rewriter)
    g.add_node("hybrid_retriever",           node_hybrid_retriever)
    g.add_node("cross_reference",            node_cross_reference)
    g.add_node("response_generator",         node_response_generator)
    g.add_node("hallucination_grader",       node_hallucination_grader)
    g.add_node("answer_grader",              node_answer_grader)
    g.add_node("store_sub_answer",           node_store_sub_answer)
    g.add_node("final_output",               node_final_output)
    g.add_node("hors_scope",                 node_hors_scope)
    g.add_node("salutation",                 node_salutation)
    g.add_node("general_knowledge_generator", node_general_knowledge)

    # ── Point d'entrée : contextualisation avant tout le reste ─────────────
    g.set_entry_point("contextualize")
    g.add_edge("contextualize", "intent_router")

    # ── Edge conditionnel : routage par intention ──────────────────────────
    g.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "query_rewriter": "decomposition",  # toute question réglementaire passe d'abord par la décomposition
            "hors_scope":     "hors_scope",
            "salutation":     "salutation",
        }
    )

    # ── Décomposition (traite aussi le cas simple : 1 seule "sous-question") ──
    g.add_edge("decomposition",              "select_current_subquestion")
    g.add_edge("select_current_subquestion", "query_rewriter")

    # ── Flux nominal ─────────────────────────────────────────────────────────
    g.add_edge("query_rewriter",     "hybrid_retriever")
    g.add_edge("hybrid_retriever",   "cross_reference")
    g.add_edge("cross_reference",    "response_generator")
    g.add_edge("response_generator", "hallucination_grader")

    # ── Edge conditionnel : boucle hallucination ────────────────────────────
    g.add_conditional_edges(
        "hallucination_grader",
        route_hallucination,
        {
            "answer_grader":               "answer_grader",
            "hybrid_retriever":            "hybrid_retriever",
            "final_output":                "final_output",  # cas d'échec définitif (retry épuisé, réglementaire précis)
            "general_knowledge_generator": "general_knowledge_generator",  # cas d'échec sur notion générale
        }
    )

    # ── Edge conditionnel : boucle pertinence ───────────────────────────────
    g.add_conditional_edges(
        "answer_grader",
        route_answer,
        {
            "store_sub_answer":            "store_sub_answer",
            "query_rewriter":              "query_rewriter",
            "final_output":                "final_output",  # cas d'échec définitif (replan épuisé, réglementaire précis)
            "general_knowledge_generator": "general_knowledge_generator",  # cas d'échec sur notion générale
        }
    )

    # ── Sortie de la branche connaissance générale : rejoint le flux normal ──
    g.add_edge("general_knowledge_generator", "store_sub_answer")

    # ── Edge conditionnel : boucle de sous-questions ────────────────────────
    g.add_conditional_edges(
        "store_sub_answer",
        route_after_store_sub_answer,
        {
            "select_current_subquestion": "select_current_subquestion",
            "final_output":                "final_output",
        }
    )

    # ── Fins ─────────────────────────────────────────────────────────────────
    g.add_edge("final_output", END)
    g.add_edge("hors_scope",   END)
    g.add_edge("salutation",   END)

    return g.compile()