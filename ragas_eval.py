"""
BCT RAG - Évaluation RAGAS sur les lignes déjà remplies (juge = Groq)
=========================================================================
Compatible avec : ragas==0.1.21, langchain==0.2.17, langchain-core==0.2.43

Lit BCT_RAGAS_Test_Table_FILLED.xlsx (produit par run_ragas_eval.py, qui
appelle ask_bct() de pipeline.py), ne garde que les lignes où "answer" ET
"contexts" sont non vides, construit le Dataset RAGAS correspondant, puis
lance l'évaluation avec Groq comme LLM juge (rotation sur 5 clés, comme
_groq_cycle dans tools.py) et des embeddings HuggingFace locaux (pas de
dépendance OpenAI).

Installation (versions figées pour compatibilité) :
    pip install ragas==0.1.21 langchain==0.2.17 langchain-core==0.2.43 \
                langchain-groq langchain-huggingface --break-system-packages

⚠️ SÉCURITÉ : les clés ci-dessous sont codées en dur à la demande explicite
du projet, pour rester cohérent avec la logique de rotation du pipeline
principal. Si ce fichier part sur un dépôt Git (même privé), ces clés sont
compromises. Ajoute au minimum ce fichier à .gitignore, et régénère
immédiatement toute clé qui aurait déjà été committée ou partagée ailleurs.

⚠️ RAPPEL : chez Groq, les rate limits (RPM/TPM/RPD/TPD) sont au niveau du
COMPTE, pas de la clé individuelle. Si ces 5 clés appartiennent au même
compte que celles déjà utilisées par le pipeline (config.py), elles
partagent le même quota — la rotation répartit le code proprement mais ne
multiplie pas la capacité réelle. Voir tools.py pour le détail.
"""

import itertools
import os
from typing import Any, List, Optional

import pandas as pd
from datasets import Dataset

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ----------------------------------------------------------------------
# CLÉS GROQ — remplace les placeholders par tes 5 clés
# ----------------------------------------------------------------------
GROQ_JUDGE_API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
]
GROQ_JUDGE_API_KEYS = [k for k in GROQ_JUDGE_API_KEYS if k and not k.startswith("gsk_KEY_")]

if not GROQ_JUDGE_API_KEYS:
    raise RuntimeError(
        "Remplace les placeholders GROQ_JUDGE_API_KEYS par tes vraies clés avant de lancer."
    )

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_XLSX = "BCT_RAGAS_Test_Table_FILLED.xlsx"
SHEET_NAME = "RAGAS_Test_Set"
OUTPUT_CSV = "bct_ragas_scores.csv"
SKIPPED_CSV = "bct_ragas_skipped_questions.csv"

GROQ_MODEL = "llama-3.3-70b-versatile"  # aligne sur le modèle utilisé dans config.py / tools.py
EMBED_MODEL_NAME = "BAAI/bge-m3"  # même famille que ton pipeline ; remplace par un modèle plus léger si besoin

# Parallélisme réduit : RAGAS lance ses appels au juge en concurrence,
# ce qui peut re-déclencher le rate-limit Groq indépendamment du pipeline.
RUN_CONFIG = RunConfig(max_workers=2, timeout=120)


# ══════════════════════════════════════════════════════════════════════════
#  ROTATION DE CLÉS POUR LE JUGE RAGAS (même logique que _groq_cycle)
# ══════════════════════════════════════════════════════════════════════════

class RotatingChatGroq(BaseChatModel):
    """
    Wrapper LangChain qui bascule en round-robin entre plusieurs instances
    ChatGroq (une par clé API), à la manière de _groq_cycle/invoke_with_fallback
    dans tools.py — mais compatible avec l'interface BaseChatModel attendue
    par LangchainLLMWrapper (RAGAS).

    Sur erreur (rate limit ou autre), essaie la clé suivante ; lève une
    RuntimeError seulement si TOUTES les clés ont échoué pour cet appel.
    """

    clients: List[Any]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, api_keys: List[str], model: str, temperature: float = 0, **kwargs):
        clients = [
            ChatGroq(api_key=key, model=model, temperature=temperature, **kwargs)
            for key in api_keys
        ]
        super().__init__(clients=clients)
        object.__setattr__(self, "_cycle", itertools.cycle(clients))

    @property
    def _llm_type(self) -> str:
        return "rotating-chat-groq"

    def _generate(self, messages, stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> ChatResult:
        last_error = None
        for _ in range(len(self.clients)):
            client = next(self._cycle)
            try:
                return client._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                print(f"⚠️ Clé Groq (juge) limitée/indisponible, rotation... ({e})")
                last_error = e
        raise RuntimeError("Toutes les clés Groq du juge RAGAS sont indisponibles.") from last_error

    async def _agenerate(self, messages, stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> ChatResult:
        last_error = None
        for _ in range(len(self.clients)):
            client = next(self._cycle)
            try:
                return await client._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                print(f"⚠️ Clé Groq (juge) limitée/indisponible, rotation... ({e})")
                last_error = e
        raise RuntimeError("Toutes les clés Groq du juge RAGAS sont indisponibles.") from last_error


def build_judge() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    llm = RotatingChatGroq(api_keys=GROQ_JUDGE_API_KEYS, model=GROQ_MODEL, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def load_filled_dataset() -> tuple[Dataset, pd.DataFrame]:
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

    answer_col = "Réponse générée (answer)"
    contexts_col = "Contextes récupérés (contexts)"

    filled_mask = df[answer_col].fillna("").str.strip().ne("") & \
                  df[contexts_col].fillna("").str.strip().ne("")

    skipped = df[~filled_mask]
    filled = df[filled_mask].copy()

    print(f"Lignes totales   : {len(df)}")
    print(f"Lignes remplies  : {len(filled)}")
    print(f"Lignes ignorées  : {len(skipped)} (voir {SKIPPED_CSV})")

    if len(skipped) > 0:
        skipped[["N°", "Question"]].to_csv(SKIPPED_CSV, index=False)

    # "contexts" a été sauvegardé comme une seule chaîne jointe par " ||| "
    # dans le xlsx (Excel ne stocke pas de listes) -> on la re-découpe
    contexts_list = filled[contexts_col].apply(
        lambda s: [c.strip() for c in str(s).split("|||") if c.strip()]
    ).tolist()

    ragas_dataset = Dataset.from_dict({
        "question": filled["Question"].tolist(),
        "answer": filled[answer_col].tolist(),
        "contexts": contexts_list,
        "ground_truth": filled["Ground Truth (réponse officielle)"].tolist(),
    })

    return ragas_dataset, filled


def main():
    dataset, filled_df = load_filled_dataset()

    if len(dataset) == 0:
        print("Aucune ligne remplie à évaluer — relance run_ragas_eval.py d'abord.")
        return

    judge_llm, judge_embeddings = build_judge()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RUN_CONFIG,
        raise_exceptions=False,  # un échec isolé (rate-limit pendant le scoring)
                                  # donne NaN sur cet exemple au lieu de stopper tout le run
    )

    scores_df = result.to_pandas()
    scores_df.insert(0, "N°", filled_df["N°"].tolist())
    scores_df.insert(1, "Type", filled_df["Type"].tolist())
    scores_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nScores sauvegardés : {OUTPUT_CSV}")
    print("\nMoyennes globales :")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in scores_df.columns:
            print(f"  {metric:20s} : {scores_df[metric].mean():.3f}  "
                  f"(NaN sur {scores_df[metric].isna().sum()} lignes)")


if __name__ == "__main__":
    main()