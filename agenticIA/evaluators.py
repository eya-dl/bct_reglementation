"""
evaluators.py — Graders de qualité du pipeline agentique BCT
=============================================================
"""

import json
import logging
from agenticIA.config import CONTEXT_LIMIT
from agenticIA.tools import (
    HALLUCINATION_PROMPT_V2,
    ANSWER_GRADER_PROMPT_V2,
    invoke_with_full_fallback,  # Groq + fallback Gemini pour plus de résilience
)

logger = logging.getLogger("evaluators")


class HallucinationGrader:
    """
    Vérifie que la réponse générée est entièrement ancrée dans
    les chunks de référence (pas d'hallucination).

    Retourne :
        "oui"  → au moins une affirmation absente des chunks (ou incertitude → fail-safe)
        "non"  → toutes les affirmations sont traçables (confirmé)
    """

    def grade(self, context: str, generation: str, max_retries: int = 2) -> str:
        context_short = context[:CONTEXT_LIMIT]

        for attempt in range(max_retries):
            try:
                result = invoke_with_full_fallback(HALLUCINATION_PROMPT_V2, {
                    "context": context_short,
                    "generation": generation,
                })
                parsed = json.loads(result.content)
                score = parsed.get("hallucination")

                if score not in ("oui", "non"):
                    raise ValueError(f"Verdict invalide reçu : {score}")

                logger.info(
                    f"[HallucinationGrader] verdict={score} confiance={parsed.get('confiance', 'N/A')} "
                    f"analyse={parsed.get('analyse', '')[:150]}"
                )
                return score

            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                logger.warning(f"[HallucinationGrader] tentative {attempt+1}/{max_retries} échouée : {e}")
                continue

        # Fail-safe : après épuisement des tentatives, on assume le PIRE
        # (possible hallucination) plutôt que de laisser passer une réponse non vérifiée
        logger.error("[HallucinationGrader] Échec après toutes les tentatives — fail-safe sur 'oui'")
        return "oui"


class AnswerGrader:
    """
    Vérifie que la réponse répond vraiment à la question posée
    avec des informations réglementaires précises.

    Retourne :
        "utile"     → réponse directe et précise (confirmé)
        "non_utile" → réponse vague, refus injustifié, hors-sujet, ou incertitude → fail-safe
    """

    def grade(self, question: str, generation: str, max_retries: int = 2) -> str:
        for attempt in range(max_retries):
            try:
                result = invoke_with_full_fallback(ANSWER_GRADER_PROMPT_V2, {
                    "question": question,
                    "generation": generation,
                })
                parsed = json.loads(result.content)
                score = parsed.get("pertinence")

                if score not in ("utile", "non_utile"):
                    raise ValueError(f"Verdict invalide reçu : {score}")

                logger.info(
                    f"[AnswerGrader] verdict={score} analyse={parsed.get('analyse', '')[:150]}"
                )
                return score

            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                logger.warning(f"[AnswerGrader] tentative {attempt+1}/{max_retries} échouée : {e}")
                continue

        # Fail-safe : après épuisement des tentatives, on assume que la réponse
        # N'est PAS utile, pour forcer une régénération plutôt que de laisser passer
        logger.error("[AnswerGrader] Échec après toutes les tentatives — fail-safe sur 'non_utile'")
        return "non_utile"


# ── Instances partagées (singleton léger) ─────────────────────────────────────
hallucination_grader = HallucinationGrader()
answer_grader = AnswerGrader()