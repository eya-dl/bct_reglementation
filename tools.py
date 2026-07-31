"""
tools.py — LLM Groq et prompts agentiques
==========================================
Contient :
  - Instance ChatGroq avec failover automatique entre plusieurs clés
  - Tous les prompts ChatPromptTemplate
  - Fonction generate() pour la génération finale (Groq ou Gemini)
"""

import itertools
import os
from google import genai
from groq import Groq, RateLimitError
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ── LLM Groq pour les nœuds agentiques ───────────────────────────────────────
# température 0 pour des décisions déterministes (graders, router, rewriter)

"""tools.py"""



from config import (
    GROQ_API_KEYS,
    GEMINI_API_KEYS,   # ← à créer dans config.py, liste au lieu d'une seule clé
    GROQ_MODEL,
    GEMINI_MODEL,
    LLM_PROVIDER,
    SYSTEM_PROMPT,
)

# ── Rotation Groq (utilisée par get_llm, invoke_with_fallback ET generate) ──
# Un seul itertools.cycle PARTAGÉ par tous les appelants : c'est ce qui garantit
# une vraie répartition round-robin de la charge sur les N clés, plutôt que de
# retomber systématiquement sur la clé n°1 en premier à chaque appel.
_groq_cycle = itertools.cycle(GROQ_API_KEYS)

def get_llm(api_key: str | None = None) -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=api_key or next(_groq_cycle))


class LLMResult:
    """Objet minimal imitant l'interface .content de LangChain, pour ne rien casser dans routers.py/nodes.py"""
    def __init__(self, content: str):
        self.content = content


def _build_messages(prompt_template, variables: dict) -> list[dict]:
    """Convertit un ChatPromptTemplate + variables en liste de messages role/content."""
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    messages = prompt_template.format_messages(**variables)
    return [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]


def invoke_with_fallback(prompt_template, variables: dict, max_retries: int = None) -> str:
    """Appelle Groq avec rotation de clés. Accepte un ChatPromptTemplate + variables."""
    messages = _build_messages(prompt_template, variables)
    max_retries = max_retries or len(GROQ_API_KEYS)
    last_error = None
    for _ in range(max_retries):
        key = next(_groq_cycle)
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            print("⚠️ Clé Groq limitée, rotation...")
            last_error = e
        except Exception as e:
            last_error = e
    raise RuntimeError("Toutes les clés Groq sont indisponibles.") from last_error


# ── Rotation Gemini (nouveau SDK google-genai, cohérent avec requirements.txt) ──
_gemini_cycle = itertools.cycle([k for k in GEMINI_API_KEYS if k])

def invoke_gemini_with_fallback(prompt_template, variables: dict, max_retries: int = None) -> str:
    """Appelle Gemini avec rotation de clés. Accepte un ChatPromptTemplate + variables."""
    messages = _build_messages(prompt_template, variables)
    # Gemini prend un texte unique : on concatène les rôles
    flat_prompt = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)

    max_retries = max_retries or len(GEMINI_API_KEYS)
    last_error = None
    for _ in range(max_retries):
        key = next(_gemini_cycle)
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=flat_prompt)
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                print("⚠️ Clé Gemini limitée, rotation...")
                last_error = e
                continue
            raise
    raise RuntimeError("Toutes les clés Gemini sont indisponibles.") from last_error


def invoke_with_full_fallback(prompt_template, variables: dict) -> LLMResult:
    """
    Point d'entrée utilisé par routers.py et nodes.py.
    Même signature que l'ancien code : (template, variables) → objet avec .content
    """
    try:
        text = invoke_with_fallback(prompt_template, variables)
    except Exception as groq_error:
        print(f"Groq épuisé, bascule sur Gemini : {groq_error}")
        text = invoke_gemini_with_fallback(prompt_template, variables)
    return LLMResult(text)
# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un classifieur d'intention pour l'assistant réglementaire
de la Banque Centrale de Tunisie (BCT).

Classifie la question dans EXACTEMENT une des quatre catégories :

- "reglementaire_precis" : question dont la réponse EXIGE une information
  réglementaire précise issue du corpus BCT (loi, décret, circulaire, article,
  délai officiel, montant réglementaire, seuil, condition ou procédure
  administrative). Sans cette information du corpus, il est IMPOSSIBLE de
  répondre correctement à la question — une réponse générale ou de culture
  générale serait insuffisante ou fausse.
  Ex: "Quel est le délai de rapatriement des devises ?" (exige un délai précis
  du corpus), "Que dit l'article 15 de la loi n°76-18 ?" (exige le texte exact),
  "Un non-résident peut-il ouvrir un compte en devises ?" (exige les conditions
  précises du corpus), "Quelles sont les règles de change pour les paiements
  d'hôtel à l'étranger ?" (exige la règle précise applicable)

- "notion_generale" : question dont la réponse NE dépend PAS d'une information
  précise du corpus — juste la définition d'un terme technique RÉELLEMENT
  financier, économique ou monétaire, qu'un financier ou un économiste
  utiliserait dans son vocabulaire professionnel. Le corpus n'est pas
  indispensable pour répondre correctement.
  Ex: "C'est quoi une devise ?", "Qu'est-ce que le taux de change ?",
  "Qu'est-ce que l'inflation ?", "C'est quoi une balance des paiements ?"

- "salutation" : bonjour, merci, comment ça va, au revoir, etc.

- "hors_scope" : TOUT le reste. Cela inclut :
  - les mots du langage courant sans sens économique propre (ex: "maison", "voiture")
  - les questions PRATIQUES ou GÉNÉRALES qui pourraient, de loin, toucher au
    change, mais qui NE MENTIONNENT PAS explicitement le change, les devises,
    les transferts ou les paiements internationaux, ET dont la réponse n'exige
    pas une information précise du corpus.
    Ex: "Comment voyager en avion ?", "Comment réserver un hôtel ?" → hors_scope,
    MÊME SI le corpus BCT contient des règles de change applicables aux voyages
    ou aux hôtels. L'utilisateur doit poser la question réglementaire lui-même,
    explicitement, pour sortir de "hors_scope".

RÈGLE DE DÉCISION STRICTE :
1. Pose-toi la question : "Pour bien répondre, ai-je BESOIN d'une donnée précise
   du corpus (article, délai, montant, condition, référence légale) ?"
   - Si OUI → "reglementaire_precis"
   - Si NON, mais le terme est un concept financier/économique standard
     → "notion_generale"
   - Si NON, et ce n'est ni financier ni économique → "hors_scope"
2. Ne classe JAMAIS une question générale/pratique en "reglementaire_precis"
   au seul motif qu'elle pourrait toucher au change par un chemin indirect.
   La question doit elle-même appeler une réponse réglementaire précise, pas
   seulement pouvoir en croiser une par accident.

Réponds UNIQUEMENT avec ce JSON (rien d'autre) :
{{"intent": "<categorie>"}}"""),
    ("human", "{question}")
])
REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert en réglementation des changes tunisienne.
Reformule la question pour maximiser la récupération de chunks pertinents dans
une base ChromaDB indexée sur des textes réglementaires BCT.

RÈGLES STRICTES :
- Utilise les termes juridiques précis (ex: "avis de change", "intermédiaire agréé",
  "circulaire BCT", "décret", "article") SEULEMENT s'ils sont pertinents au sujet exact posé.
- Si un numéro de texte (loi, décret, circulaire) est mentionné dans la question, conserve-le
  EXACTEMENT. N'invente JAMAIS un numéro de référence qui n'est pas dans la question originale.
- Ne modifie PAS le périmètre de la question : ne l'élargis pas, ne la restreins pas.
- Si la question est déjà précise et bien formulée juridiquement, renvoie-la TELLE QUELLE
  sans changement.
- Reformulation concise : 25 mots maximum.
- Une seule phrase, sans explication.

Exemples :

Question originale : "Un non-résident peut-il ouvrir un compte en devises ?"
Reformulation : "Conditions d'ouverture d'un compte en devises par un non-résident selon la réglementation BCT"

Question originale : "Quel est le délai pour rapatrier les devises ?"
Reformulation : "Délai réglementaire de rapatriement des devises pour les exportateurs, article et circulaire BCT applicables"

Question originale : "Que dit l'article 15 de la loi n°76-18 sur les intermédiaires agréés ?"
Reformulation : "Que dit l'article 15 de la loi n°76-18 sur les intermédiaires agréés ?"
(déjà précise et référencée → renvoyée telle quelle)

Réponds UNIQUEMENT avec la question reformulée."""),
    ("human", "Question originale : {question}")
])

REWRITER_ANAPHORA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert en réglementation des changes tunisienne.
La sous-question ci-dessous contient un pronom ou une ellipse ("le faire",
"s'en charger", "dans ce cas", sujet sous-entendu) qui fait référence à la
question de référence fournie.

Ta tâche : remplace le pronom/l'ellipse par le sujet EXACT de la question de
référence — ne le change pas, ne le remplace pas par un synonyme technique,
ne devine pas un autre sujet réglementaire. Tu peux ensuite enrichir légèrement
avec le vocabulaire juridique pertinent à CE sujet précis (pas un autre).

Exemple :
Question de référence : "Comment déclarer un compte en devises ?"
Sous-question : "quand faut-il le faire ?"
Reformulation correcte : "Quel est le délai réglementaire pour déclarer un compte en devises à la BCT ?"
Reformulation INCORRECTE : "Quand faut-il effectuer les démarches relatives à l'avis de change ?"

- Une seule phrase reformulée, sans explication
- Réponds UNIQUEMENT avec la question reformulée"""),
    ("human", "Question de référence (sujet exact à conserver) : {reference_question}\n\nSous-question à reformuler : {question}")
])

HALLUCINATION_PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", """Tu es un contrôleur qualité rigoureux d'un système RAG juridique BCT.

Analyse la réponse affirmation par affirmation par rapport aux chunks de référence.

Chunks de référence :
{context}

Réponse à évaluer :
{generation}

Procède en 2 étapes :
1. Liste chaque affirmation factuelle de la réponse (articles, montants, délais, obligations)
   et indique si elle est explicitement traçable dans les chunks.
2. Donne un verdict global.

Règle stricte :
- "oui" = AU MOINS UNE affirmation factuelle n'est pas explicitement présente dans les chunks
- "non" = TOUTES les affirmations factuelles sont traçables dans les chunks
  (formules de politesse et mise en forme tolérées, pas les faits juridiques)

Réponds UNIQUEMENT avec ce JSON :
{{"analyse": "<ton analyse affirmation par affirmation en 2-3 phrases>", "hallucination": "oui" ou "non", "confiance": "haute" ou "moyenne" ou "basse"}}"""),
    ("human", "Évalue.")
])

ANSWER_GRADER_PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", """Tu es un évaluateur de pertinence rigoureux pour un assistant réglementaire BCT.

Question posée : {question}
Réponse générée : {generation}

Analyse d'abord si la réponse contient les éléments réglementaires précis
attendus pour ce type de question (articles, délais, montants, obligations, procédures).

Critères :
- "utile"     : réponse directe, précise, avec au moins un élément réglementaire concret
- "non_utile" : réponse vague, refus injustifié, hors-sujet, ou qui tourne autour du sujet sans répondre

Réponds UNIQUEMENT avec ce JSON :
{{"analyse": "<pourquoi la réponse répond ou non à la question, 1-2 phrases>", "pertinence": "utile" ou "non_utile"}}"""),
    ("human", "Évalue.")
])


CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant qui reformule les questions en tenant compte de
l'historique de conversation, pour l'assistant réglementaire de la Banque Centrale
de Tunisie (BCT).

RÈGLES STRICTES :
- Si la question actuelle fait référence à des éléments mentionnés précédemment
  (pronoms, ellipses, termes implicites comme "cette loi", "et pour...", "qu'en est-il de...",
  "dans ce cas"), reformule-la en question complète et autonome, compréhensible SANS
  l'historique.
- Si la question est déjà autonome et claire, renvoie-la TELLE QUELLE, sans modification.
- Ne réponds JAMAIS à la question, reformule-la seulement.
- Préserve la langue originale (français) et l'intention exacte de l'utilisateur.
- N'invente aucun numéro d'article, de loi ou de circulaire absent de l'historique.
- Une seule phrase reformulée, sans explication.

Exemples :

Historique :
Utilisateur : Quelles sont les obligations de rapatriement de devises pour les exportateurs ?
Assistant : Selon la circulaire XYZ, les exportateurs doivent rapatrier...
Question actuelle : Et pour les banques étrangères ?
Reformulation : Quelles sont les obligations de rapatriement de devises pour les banques étrangères ?

Historique :
Utilisateur : Comment déclarer un compte en devises ?
Assistant : La déclaration se fait via un avis de change...
Question actuelle : Quel est le délai maximum pour le faire ?
Reformulation : Quel est le délai maximum pour déclarer un compte en devises à la BCT ?

Historique : (aucun)
Question actuelle : Quelles sont les conditions d'ouverture d'un compte en devises par un non-résident ?
Reformulation : Quelles sont les conditions d'ouverture d'un compte en devises par un non-résident ?
(déjà autonome → renvoyée telle quelle)

Réponds UNIQUEMENT avec la question reformulée."""),
    ("human", "Historique de conversation :\n{history}\n\nQuestion actuelle : {question}")
])


GENERAL_KNOWLEDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant BCT. La question porte sur une notion financière générale
liée au thème du change, mais aucune définition n'a été trouvée dans le corpus réglementaire.

Explique la notion avec tes connaissances générales, de façon claire et pédagogique.

RÈGLES OBLIGATOIRES :
- Précise explicitement au début que cette explication est une notion générale
  et ne provient pas d'un texte réglementaire BCT spécifique.
- N'invente aucune référence légale (pas de numéro de loi, décret ou circulaire).
- Reste factuel et neutre, sans donner de conseil financier personnalisé.
- Si la question mélange une notion générale ET un aspect réglementaire précis,
  indique clairement que la partie réglementaire nécessite de consulter les textes BCT."""),
    ("human", "{question}")
])
# ══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATION LLM (Groq ou Gemini)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_groq(query: str, context: str) -> str:
    """
    Génération finale via Groq.
    Bascule automatiquement sur la clé suivante si la première
    atteint son quota (RateLimitError) ou échoue.

    IMPORTANT : utilise le même _groq_cycle partagé que get_llm() et
    invoke_with_fallback(), plutôt que de re-parcourir GROQ_API_KEYS depuis
    le début à chaque appel. Sans ça, la clé n°1 est systématiquement
    sollicitée en premier à CHAQUE génération, ce qui la fait plus vite
    passer en rate-limit et gaspille un appel raté avant de basculer —
    au lieu de vraiment répartir la charge sur les 10 clés en round-robin.
    """
    last_error = None
    max_retries = len(GROQ_API_KEYS)

    for _ in range(max_retries):
        api_key = next(_groq_cycle)

        try:
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Contexte réglementaire :\n{context}\n\nQuestion : {query}",
                    },
                ],
                temperature=0.1,
                max_tokens=1024,
            )

            return response.choices[0].message.content

        except RateLimitError as e:
            print("⚠️ Clé Groq limitée, passage à la suivante...")
            last_error = e

        except Exception as e:
            print(f"Erreur avec une clé Groq : {e}")
            last_error = e

    raise RuntimeError(
        "Toutes les clés Groq sont indisponibles."
    ) from last_error


def _generate_gemini(query: str, context: str) -> str:
    """Génère une réponse via l'API Gemini."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(
        f"Contexte réglementaire :\n{context}\n\nQuestion : {query}"
    )
    return response.text


def generate(query: str, context: str, provider: str = LLM_PROVIDER) -> str:
    """
    Point d'entrée unique pour la génération LLM.
    Sélectionne automatiquement Groq ou Gemini selon LLM_PROVIDER.
    """
    if provider == "groq":
        return _generate_groq(query, context)
    elif provider == "gemini":
        return _generate_gemini(query, context)
    else:
        raise ValueError(f"Provider inconnu : '{provider}'. Utilise 'groq' ou 'gemini'.")
    


def format_subanswers_context(sub_answers: dict) -> str:
    """
    Formate les sous-réponses validées en un contexte structuré
    pour le Final Generator, quand la question était composite.
    """
    if len(sub_answers) == 1:
        # Cas simple : une seule sous-question = la question originale
        return list(sub_answers.values())[0]

    parts = []
    for i, (question, answer) in enumerate(sub_answers.items(), 1):
        parts.append(f"Sous-question {i} : {question}\nRéponse validée : {answer}")
    return "\n\n".join(parts)