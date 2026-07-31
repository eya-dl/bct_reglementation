"""
BCT RAG - Génération du dataset RAGAS à partir du tableau de test
===================================================================

Lit BCT_RAGAS_Test_Table.xlsx (100 questions simples + 10 composées),
appelle ask_bct() du pipeline BCT pour chaque question, remplit les
colonnes "answer" et "contexts", puis construit et sauvegarde un
Dataset (HuggingFace datasets) prêt pour l'évaluation RAGAS.

Reprise automatique : si OUTPUT_XLSX existe déjà (run précédent
interrompu, ou lignes déjà remplies manuellement), le script les
détecte et NE RAPPELLE PAS le pipeline pour ces questions — seules
les lignes sans réponse sont traitées. Sauvegarde incrémentale après
chaque question (pas seulement à la fin) pour ne rien perdre en cas
de coupure/rate-limit.

Prérequis : lancer ce script depuis le dossier où se trouve pipeline.py
(même niveau que retriever.py, nodes.py, graph.py, state.py), pour que
l'import fonctionne.
"""

import os
import time
import pandas as pd
from datasets import Dataset

from pipeline import ask_bct  # charge les ressources RAG à l'import (peut prendre du temps)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_XLSX = "BCT_RAGAS_Test_Table.xlsx"
SHEET_NAME = "RAGAS_Test_Set"
OUTPUT_XLSX = "BCT_RAGAS_Test_Table_FILLED.xlsx"
OUTPUT_DATASET_JSON = "bct_ragas_dataset.json"
SLEEP_BETWEEN_CALLS = 1.0  # secondes, pour ménager les clés Groq/Gemini (rate-limit)

ANSWER_COL = "Réponse générée (answer)"
CONTEXTS_COL = "Contextes récupérés (contexts)"


def sources_to_contexts(sources: list) -> list[str]:
    """
    Convertit result["sources"] en liste de chaînes de contexte pour RAGAS.

    Utilise le champ "content" (texte complet du chunk) ajouté dans
    format_sources() (retriever.py). Repli sur "extrait" (300 premiers
    caractères) ou sur les métadonnées de citation si "content" est absent
    — utile si tu évalues une version du pipeline antérieure au patch.
    """
    contexts = []
    for s in sources:
        if s.get("content"):
            contexts.append(s["content"])
        elif s.get("extrait"):
            contexts.append(s["extrait"])
        else:
            parts = [s.get("type", ""), s.get("number", ""), s.get("article") or s.get("breadcrumb", "")]
            contexts.append(" — ".join(p for p in parts if p))
    return contexts


def call_rag_pipeline(question: str) -> tuple[str, list[str]]:
    """Appelle le pipeline BCT réel et retourne (answer, contexts)."""
    result = ask_bct(question, conversation_history=[])
    answer = result.get("final_answer", "")
    contexts = sources_to_contexts(result.get("sources", []))
    return answer, contexts


def is_filled(value) -> bool:
    """Une cellule est considérée 'déjà remplie' si elle n'est ni NaN ni vide/blanche."""
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def load_working_dataframe() -> pd.DataFrame:
    """
    Charge le tableau de travail :
      - reprend OUTPUT_XLSX s'il existe déjà (run précédent, ou lignes
        pré-remplies manuellement) → on garde ce qui est déjà fait
      - sinon repart de INPUT_XLSX et ajoute les colonnes vides
    """
    if os.path.exists(OUTPUT_XLSX):
        print(f"Reprise depuis {OUTPUT_XLSX} (déjà existant)")
        df = pd.read_excel(OUTPUT_XLSX, sheet_name=SHEET_NAME)
    else:
        print(f"Nouveau run depuis {INPUT_XLSX}")
        df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

    if ANSWER_COL not in df.columns:
        df[ANSWER_COL] = ""
    if CONTEXTS_COL not in df.columns:
        df[CONTEXTS_COL] = ""

    return df


def main():
    df = load_working_dataframe()

    already_done = df[ANSWER_COL].apply(is_filled).sum()
    print(f"{already_done}/{len(df)} question(s) déjà remplie(s) — seront sautées.\n")

    for i, row in df.iterrows():
        question = row["Question"]

        if is_filled(row[ANSWER_COL]):
            print(f"[{i+1}/{len(df)}] déjà rempli, skip : {str(question)[:60]}...")
            continue

        print(f"[{i+1}/{len(df)}] {str(question)[:80]}...")

        try:
            answer, contexts = call_rag_pipeline(question)
        except Exception as e:
            print(f"   ! Erreur sur la question {row.get('N°', i)}: {e}")
            answer, contexts = "", []

        df.at[i, ANSWER_COL] = answer
        df.at[i, CONTEXTS_COL] = " ||| ".join(contexts)

        # Sauvegarde incrémentale : si ça plante ou rate-limit à la question
        # 60, les 59 précédentes ne sont pas perdues et ne seront pas rejouées.
        df.to_excel(OUTPUT_XLSX, sheet_name=SHEET_NAME, index=False)

        if SLEEP_BETWEEN_CALLS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nTableau Excel rempli sauvegardé : {OUTPUT_XLSX}")

    contexts_list = [
        (c.split(" ||| ") if c else []) for c in df[CONTEXTS_COL].fillna("")
    ]

    ragas_dataset = Dataset.from_dict({
        "question": df["Question"].tolist(),
        "answer": df[ANSWER_COL].fillna("").tolist(),
        "contexts": contexts_list,  # list[list[str]]
        "ground_truth": df["Ground Truth (réponse officielle)"].tolist(),
    })

    ragas_dataset.to_json(OUTPUT_DATASET_JSON)
    print(f"Dataset RAGAS sauvegardé : {OUTPUT_DATASET_JSON}")
    print(f"Nombre d'exemples : {len(ragas_dataset)}")

    return ragas_dataset


if __name__ == "__main__":
    main()