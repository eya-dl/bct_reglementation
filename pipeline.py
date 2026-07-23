"""
pipeline.py — Point d'entrée unique du pipeline agentique BCT
================================================================
C'est le SEUL fichier que ton app.py Streamlit doit importer.

Utilisation :
    from pipeline import ask_bct
    result = ask_bct("Quel est le délai de rapatriement des devises ?")
    print(result["final_answer"])
    for s in result["sources"]:
        print(s["type"], s["number"], "—", s["article"])
"""

import time
from retriever import load_resources
from nodes import inject_resources
from graph import build_bct_graph
from state import initial_state

# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION (exécutée une seule fois à l'import du module)
# ══════════════════════════════════════════════════════════════════════════════

print("⏳ Chargement des ressources RAG...")
_model, _reranker, _collection, _bm25, _children = load_resources()
inject_resources(_model, _reranker, _collection, _bm25, _children)
print("✅ Ressources chargées.\n")

print("⏳ Construction du graphe LangGraph...")
_graph = build_bct_graph()
print("✅ Graphe compilé.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFACE PUBLIQUE
# ══════════════════════════════════════════════════════════════════════════════

def ask_bct(question: str, conversation_history: list = None) -> dict:
    """
    Point d'entrée principal du pipeline agentique BCT.

    Args:
        question              : question de l'utilisateur en français
        conversation_history  : historique [{"role": "user"/"assistant", "content": "..."}]

    Returns:
        {
          "final_answer":  str,
          "sources":       list[dict],
          "intent":        str,
          "retry_count":   int,
          "replan_count":  int,
          "timings":       dict,
        }
    """
    t0 = time.perf_counter()
    state = _graph.invoke(initial_state(question, conversation_history or []))
    total = time.perf_counter() - t0

    return {
        "final_answer": state.get("final_answer", ""),
        "sources":      state.get("sources", []),
        "intent":       state.get("intent", ""),
        "retry_count":  state.get("retry_count", 0),
        "replan_count": state.get("replan_count", 0),
        "timings":      {"pipeline": total},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODE INTERACTIF (test en ligne de commande)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("━" * 80)
    print("  PIPELINE AGENTIQUE BCT — LangGraph + Hybrid RAG")
    print("  Tape 'quit' pour quitter")
    print("━" * 80)

    while True:
        try:
            question = input("\n❓ Question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        result = ask_bct(question)

        print("\n" + "═" * 80)
        print("💬 RÉPONSE FINALE")
        print("─" * 80)
        print(result["final_answer"])
        print("═" * 80)

        if result["sources"]:
            print("\n📚 Sources retenues :")
            for s in result["sources"]:
                print(
                    f"  [{s['rank']}] rerank={s['rerank_score']:.4f} | "
                    f"{s['type']} {s['number']} — {s['article'] or s['breadcrumb'][:55]}"
                )

        print(
            f"\n  ℹ️  intent={result['intent']} | "
            f"retries={result['retry_count']} | replans={result['replan_count']}"
        )


if __name__ == "__main__":
    main()