"""
nodes.py — Nœuds du graphe LangGraph BCT
==========================================
Chaque fonction est un nœud LangGraph :
  - prend BCTState en entrée
  - retourne un dict partiel qui met à jour l'état

Nœuds :
  1. node_intent_router              → classifie l'intention
  2. node_decomposition              → décompose les questions composites
  3. node_select_current_subquestion → pointe 'question' vers la sous-question active
  4. node_query_rewriter             → reformule la question
  5. node_hybrid_retriever           → récupère et reranke les chunks
  6. node_response_generator         → génère la réponse LLM
  7. node_hallucination_grader       → évalue l'ancrage
  8. node_answer_grader              → évalue la pertinence
  9. node_store_sub_answer           → stocke la sous-réponse validée
  10.node_final_output              → formate/synthétise la sortie finale
  + node_hors_scope / node_salutation → réponses statiques
"""

from agenticIA.config import LLM_PROVIDER, HORS_SCOPE_REPONSE, NOTION_GENERALE_DISCLAIMER, SALUTATION_REPONSE, MAX_SUBQUESTIONS
from agenticIA.state import BCTState
from agenticIA.tools import GENERAL_KNOWLEDGE_PROMPT, REWRITER_PROMPT, generate, invoke_with_full_fallback, format_subanswers_context
from agenticIA.retriever import extract_cross_references, fetch_chunks_by_reference, run_retrieval, format_sources
from agenticIA.evaluators import hallucination_grader, answer_grader
from agenticIA.routers import intent_router
from agenticIA.decomposer import is_composite, split_into_subquestions


# ── Ressources RAG (injectées depuis pipeline.py au démarrage) ────────────────
_model      = None
_reranker   = None
_collection = None
_bm25       = None
_children   = None


def inject_resources(model, reranker, collection, bm25, children):
    """
    Injecte les ressources RAG dans ce module.
    Appelé une seule fois au démarrage dans pipeline.py.
    """
    global _model, _reranker, _collection, _bm25, _children
    _model      = model
    _reranker   = reranker
    _collection = collection
    _bm25       = bm25
    _children   = children


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 1 — INTENT ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def node_intent_router(state: BCTState) -> dict:
    """Classifie l'intention de la question."""
    print(f"\n [Intent Router] '{state['question'][:70]}'")
    intent = intent_router.classify(state["question"])
    print(f"   → intent = {intent}")
    return {"intent": intent}


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 2 — DÉCOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def node_decomposition(state: BCTState) -> dict:
    """
    Analyse la question brute et la découpe en sous-questions atomiques
    si elle est composite. Toujours en amont du Query Rewriter.
    """
    question = state["question"]
    composite = is_composite(question)
    sub_questions = split_into_subquestions(question) if composite else [question]

    # Garde-fou : si le regex génère trop de fragments, on retombe sur la question brute
    if len(sub_questions) > MAX_SUBQUESTIONS:
        sub_questions = [question]
        composite = False

    print(f" [Decomposition] composite={composite} → {len(sub_questions)} sous-question(s)")
    if composite:
        for i, sq in enumerate(sub_questions, 1):
            print(f"   {i}. {sq}")

    return {
        "original_question": question,
        "is_composite": composite,
        "sub_questions": sub_questions,
        "current_sub_index": 0,
        "sub_answers": {},
        "collected_sources": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 3 — SÉLECTION DE LA SOUS-QUESTION COURANTE
# ══════════════════════════════════════════════════════════════════════════════

def node_select_current_subquestion(state: BCTState) -> dict:
    """
    Pointe 'question' vers la sous-question à traiter dans cette itération.
    Permet à query_rewriter/answer_grader/response_generator existants
    de fonctionner sans aucune modification.
    """
    idx = state["current_sub_index"]
    current_q = state["sub_questions"][idx]
    print(f" [Select Subquestion {idx + 1}/{len(state['sub_questions'])}] {current_q[:70]}")
    return {"question": current_q}


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 4 — QUERY REWRITER
# ══════════════════════════════════════════════════════════════════════════════

from agenticIA.tools import REWRITER_PROMPT, REWRITER_ANAPHORA_PROMPT, generate, invoke_with_full_fallback, format_subanswers_context


def node_query_rewriter(state: BCTState) -> dict:
    """
    Reformule la question avec les termes juridiques BCT pour
    maximiser la précision du retrieval.

    Deux chemins distincts, sans ambiguïté de prompt :
      - idx == 0 (ou question simple) → REWRITER_PROMPT classique, enrichissement libre
      - idx > 0 (sous-question dérivée) → REWRITER_ANAPHORA_PROMPT, résolution de
        référence stricte vers la sous-question précédente puis enrichissement

    Incrémente replan_count si on revient d'un replan.
    Utilise invoke_with_full_fallback() pour basculer automatiquement
    entre les deux clés Groq en cas de rate-limit.
    """
    print("  [Query Rewriter]")

    idx = state.get("current_sub_index", 0)
    sub_questions = state.get("sub_questions") or [state["question"]]

    if idx > 0:
        reference_question = sub_questions[idx - 1]
        result = invoke_with_full_fallback(REWRITER_ANAPHORA_PROMPT, {
            "question": state["question"],
            "reference_question": reference_question,
        })
        print(f"   Référence     : {reference_question[:70]}")
    else:
        result = invoke_with_full_fallback(REWRITER_PROMPT, {
            "question": state["question"],
        })

    rewritten = result.content.strip()
    print(f"   Sous-question : {state['question'][:70]}")
    print(f"   Rewritten     : {rewritten[:70]}")

    return {
        "rewritten_query": rewritten,
        "replan_count":    state.get("replan_count", 0) + 1
        if state.get("answer_score") == "non_utile"
        else state.get("replan_count", 0),
    }
# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 5 — HYBRID RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def node_hybrid_retriever(state: BCTState) -> dict:
    """
    Exécute le pipeline retrieval hybride complet :
    BM25 + BGE-M3 → RRF → CrossEncoder reranking.
    Pour les sous-questions dérivées (index > 0), biaise le retrieval
    vers les documents déjà utilisés pour la sous-question précédente,
    afin d'éviter une dispersion vers des textes réglementaires non liés.
    Incrémente retry_count à chaque appel.
    """
    query = state.get("rewritten_query") or state["question"]
    print(f" [Hybrid Retriever] '{query[:70]}'")

    prior_doc_refs = None
    if state.get("is_composite") and state.get("current_sub_index", 0) > 0:
        prior_sources = state.get("collected_sources", [])
        prior_doc_refs = {(s["type"], s["number"]) for s in prior_sources if s.get("type")}
        if prior_doc_refs:
            print(f"   📎 Biais documentaire actif : {prior_doc_refs}")
    prior_doc_refs = None
    if state.get("is_composite") and state.get("current_sub_index", 0) > 0:
        prior_sources = state.get("collected_sources", [])
        print(f"   🐛 DEBUG collected_sources = {prior_sources}")
        prior_doc_refs = {(s["type"], s["number"]) for s in prior_sources if s.get("type")}
        print(f"   🐛 DEBUG prior_doc_refs = {prior_doc_refs}")


    ranked, context = run_retrieval(
        query, _model, _reranker, _collection, _bm25, _children,
        prior_doc_refs=prior_doc_refs,
    )

    print(f"   {len(ranked)} chunks retenus après reranking :")
    for r in ranked:
        c = r["chunk"]
        boost_flag = " ⭐" if r.get("prior_boost_applied") else ""
        print(
            f"   rerank={r['rerank_score']:.4f}{boost_flag} | "
            f"{c.document_type} {c.document_number} — "
            f"{(c.article_title or c.breadcrumb)[:50]}"
        )

    return {
        "ranked_chunks": ranked,
        "context":       context,
        "retry_count":   state.get("retry_count", 0) + 1,
    }
# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 6 — RESPONSE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def node_response_generator(state: BCTState) -> dict:
    """
    Génère la réponse finale via le LLM (Groq ou Gemini).
    Utilise le contexte formaté produit par le retriever.
    """
    print(f" [Response Generator] provider={LLM_PROVIDER}")
    answer = generate(
        query=state["question"],
        context=state.get("context", ""),
        provider=LLM_PROVIDER,
    )
    print(f"   Réponse : {len(answer)} caractères")
    return {"generation": answer}


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 7 — HALLUCINATION GRADER
# ══════════════════════════════════════════════════════════════════════════════

def node_hallucination_grader(state: BCTState) -> dict:
    """
    Évalue si la réponse est ancrée dans les chunks récupérés.
    Délègue à evaluators.HallucinationGrader.
    """
    print(" [Hallucination Grader]")
    score = hallucination_grader.grade(
        context=state.get("context", ""),
        generation=state.get("generation", ""),
    )
    print(f"   → hallucination = {score}")
    return {"hallucination_score": score}


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 8 — ANSWER GRADER
# ══════════════════════════════════════════════════════════════════════════════

def node_answer_grader(state: BCTState) -> dict:
    """
    Évalue si la réponse répond vraiment à la question.
    Délègue à evaluators.AnswerGrader.
    """
    print(" [Answer Grader]")
    score = answer_grader.grade(
        question=state["question"],
        generation=state.get("generation", ""),
    )
    print(f"   → pertinence = {score}")
    return {"answer_score": score}


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 9 — STOCKAGE DE LA SOUS-RÉPONSE VALIDÉE
# ══════════════════════════════════════════════════════════════════════════════

def node_store_sub_answer(state: BCTState) -> dict:
    """
    Enregistre la réponse validée de la sous-question courante,
    accumule ses sources, puis réinitialise les champs de travail
    pour la prochaine itération.
    """
    current_q = state["sub_questions"][state["current_sub_index"]]

    updated_sub_answers = dict(state["sub_answers"])
    updated_sub_answers[current_q] = state.get("generation", "")

    updated_sources = list(state.get("collected_sources", []))
    updated_sources.extend(format_sources(state.get("ranked_chunks") or []))

    print(f" [Store Sub-Answer] '{current_q[:50]}' enregistrée")

    return {
        "sub_answers": updated_sub_answers,
        "collected_sources": updated_sources,
        "current_sub_index": state["current_sub_index"] + 1,
        # Reset impératif : sinon les champs de travail de la sous-question
        # précédente polluent la suivante
        "rewritten_query": None,
        "ranked_chunks": None,
        "context": None,
        "generation": None,
        "hallucination_score": None,
        "answer_score": None,
        "retry_count": 0,
        "replan_count": 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD 10 — FINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def node_final_output(state: BCTState) -> dict:
    """
    Formate la sortie finale avec les sources.
    Gère deux cas :
      - flux normal : toutes les sous-questions sont passées par store_sub_answer
      - sortie d'urgence (max retries/replans atteint) : la sous-question courante
        n'a pas encore été stockée, on la récupère ici en secours pour ne pas
        perdre sa réponse/sources.
    """
    sub_answers = dict(state.get("sub_answers", {}))
    collected_sources = list(state.get("collected_sources", []))

    sub_questions = state.get("sub_questions") or [state["question"]]
    idx = state.get("current_sub_index", 0)

    if idx < len(sub_questions):
        current_q = sub_questions[idx]
        if current_q not in sub_answers and state.get("generation"):
            sub_answers[current_q] = state["generation"]
            collected_sources.extend(format_sources(state.get("ranked_chunks") or []))

    if len(sub_answers) > 1:
        final_answer = format_subanswers_context(sub_answers)
    else:
        final_answer = next(iter(sub_answers.values()), state.get("generation", ""))

    print(f" [Final Output] {len(sub_answers)} sous-réponse(s) synthétisée(s)")

    return {
        "final_answer": final_answer,
        "sources": collected_sources,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUDS STATIQUES — HORS SCOPE / SALUTATION
# ══════════════════════════════════════════════════════════════════════════════

def node_hors_scope(state: BCTState) -> dict:
    """Retourne une réponse de refus poli pour les questions hors périmètre."""
    print(" [Hors scope]")
    return {"final_answer": HORS_SCOPE_REPONSE, "sources": []}


def node_salutation(state: BCTState) -> dict:
    """Retourne une réponse de bienvenue pour les salutations."""
    print(" [Salutation]")
    return {"final_answer": SALUTATION_REPONSE, "sources": []}


from agenticIA.tools import CONTEXTUALIZE_PROMPT, invoke_with_full_fallback


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUDS 11 — CONTEXTUALISATION DE LA QUESTION
# ══════════════════════════════════════════════════════════════════════════════

def node_contextualize(state: BCTState) -> dict:
    """Reformule la question en résolvant les références à l'historique de conversation."""
    print(f"\n [Contextualisation] '{state['question'][:70]}'")

    history = state.get("conversation_history", [])
    question = state["question"]

    # Pas d'historique = rien à résoudre, on court-circuite (pas d'appel LLM)
    if not history:
        print("   → pas d'historique, question inchangée")
        return {"question": question, "original_question": question}

    # Ne garde que les 3 derniers échanges (6 messages) pour ne pas saturer le prompt
    recent_history = history[-6:]
    history_text = "\n".join(
        f"{'Utilisateur' if turn['role'] == 'user' else 'Assistant'} : {turn['content']}"
        for turn in recent_history
    )

    try:
        result = invoke_with_full_fallback(CONTEXTUALIZE_PROMPT, {
            "history": history_text,
            "question": question,
        })
        reformulated = result.content.strip()

        if reformulated != question:
            print(f"   → reformulée : '{reformulated[:70]}'")
        else:
            print("   → inchangée (déjà autonome)")

        return {"question": reformulated, "original_question": question}

    except Exception as e:
        # Ne jamais bloquer le pipeline : fallback sur la question brute
        print(f"   ⚠️ Erreur contextualisation, fallback sur question brute : {e}")
        return {"question": question, "original_question": question}
    
# ══════════════════════════════════════════════════════════════════════════════
#  NŒUDS 12 — CROSS-REFERENCE DETECTION
# ══════════════════════════════════════════════════════════════════════════════


def node_cross_reference(state: BCTState) -> dict:
    """
    Détecte les références croisées explicites dans les chunks retenus
    (ex: "conformément à l'article 12 de la loi n°76-18") et enrichit
    le contexte avec les chunks correspondants s'ils manquent.
    """
    chunks = state.get("ranked_chunks") or []
    print(f"\n [Cross Reference] analyse de {len(chunks)} chunks retenus")

    if not chunks:
        return {}

    references = extract_cross_references(chunks)

    if not references:
        print("   → aucune référence croisée détectée")
        return {}

    print(f"   → {len(references)} référence(s) détectée(s) : {references}")

    existing_ids = {c.get("id") or c.get("metadata", {}).get("chunk_id") for c in chunks}
    additional_chunks = fetch_chunks_by_reference(references, existing_ids, _collection)  # _collection injecté comme le reste

    if not additional_chunks:
        print("   → aucune référence supplémentaire trouvée dans la base")
        return {}

    print(f"   ⭐ {len(additional_chunks)} chunk(s) ajouté(s) par cross-référence")

    return {"ranked_chunks": chunks + additional_chunks}



def node_general_knowledge(state):
    result = invoke_with_full_fallback(GENERAL_KNOWLEDGE_PROMPT, {"question": state["question"]})
    answer = result.content + NOTION_GENERALE_DISCLAIMER
    return {**state, "generation": answer}