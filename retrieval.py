"""
ÉTAPE 4 : RETRIEVAL AVANCÉ
============================
Pipeline : Multi-query → Cosinus large → Re-ranking → Dédoublonnage
Optimisé pour corpus réduit (4 PDF : NGAP + 3 circulaires).

Changements clés vs version précédente :
- retrieval_k : 20 → 30 (moins de chunks totaux, on cherche plus large)
- min_cosine : 0.75 → 0.68 (chunks plus petits = scores parfois plus bas)
- weight_cosine : 0.90 → 0.95 (2 niveaux de priorité seulement → peu d'impact)
- Prompt de décomposition enrichi avec patterns circulaires + NGAP
- dedup_threshold : 0.97 → 0.93 (chunks plus petits = doublons plus proches)
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass

from embeddings import (
    EmbeddedChunk,
    EmbeddingEngine,
    cosine_similarity,
    search_similar,
)


# ============================================================
# 1. APPEL API MISTRAL (Chat + Rerank)
# ============================================================

def call_mistral_chat(
    messages: list[dict],
    api_key: str,
    model: str = "mistral-large-latest",
    temperature: float = 0.1,
    max_tokens: int = 300,
) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        raise RuntimeError(f"Erreur API Mistral ({e.code}) : {body}")
    except Exception as e:
        raise RuntimeError(f"Erreur connexion API Mistral : {e}")

    return result["choices"][0]["message"]["content"]


def call_mistral_rerank(
    query: str,
    documents: list[str],
    api_key: str,
    model: str = "mistral-rerank-latest",
    top_k: int | None = None,
) -> list[dict]:
    """
    Cross-encoder Mistral : lit query + chunk ENSEMBLE → score de pertinence.
    Plus précis que le cosinus, mais plus lent → utilisé sur top 30 seulement.
    """
    payload = {"model": model, "query": query, "documents": documents}
    if top_k is not None:
        payload["top_k"] = top_k

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/rerank",
        data=data, headers=headers, method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        raise RuntimeError(f"Erreur API Mistral Rerank ({e.code}) : {body}")
    except Exception as e:
        raise RuntimeError(f"Erreur connexion API Mistral Rerank : {e}")

    return result["results"]


# ============================================================
# 2. MULTI-QUERY DECOMPOSITION
# ============================================================

DECOMPOSE_SYSTEM_PROMPT = """Tu es un expert en cotation NGAP pour les infirmiers libéraux en France.

Ta tâche : transformer une situation clinique ou une question en 3 à 6 sous-requêtes de recherche pertinentes pour retrouver les règles de cotation exactes.

IMPORTANT :
- La question peut contenir plusieurs actes, plusieurs jours, ou une situation complexe (BSI, dépendance, cumul…)
- Tu dois couvrir TOUS les aspects nécessaires pour déterminer la cotation finale

────────────────────────────────────────
🎯 OBJECTIF
Produire des sous-requêtes permettant de retrouver :
1. Les actes cotables (avec coefficient + lettre clé)
2. Les règles de cumul (article 11B ou dérogations)
3. Les majorations (MCI, MAU, MIE…)
4. Les indemnités (IFD, IFI…)
5. Les cas particuliers (BSI, dépendance, enfant, psychiatrie…)
6. Les règles spécifiques (actes à 50%, cumul à taux plein…)

────────────────────────────────────────
🧠 ANALYSE À FAIRE
Avant d’écrire les requêtes, identifie :
- Les actes présents (ex : pansement, injection, prélèvement…)
- Le contexte :
  - domicile ?
  - dépendance / BSI ?
  - enfant ?
  - pathologie ?
- Le timing :
  - même séance ?
  - plusieurs jours ?
- Les règles implicites :
  - cumul
  - dérogation
  - acte principal / secondaire

────────────────────────────────────────
🧩 TYPES DE SOUS-REQUÊTES À PRODUIRE

1. ACTES NGAP (OBLIGATOIRE)
→ une requête par acte important
Ex :
"Article 2 pansement de stomie Chapitre I Titre XVI coefficient lettre clé AMI"
"Article 1 injection sous-cutanée coefficient AMI NGAP"

2. RÈGLES DE CUMUL (OBLIGATOIRE)
→ toujours inclure une requête sur :
"article 11B dispositions générales NGAP cumul actes même séance deuxième acte 50%"

3. DÉROGATIONS (si pertinent)
Ex :
"prélèvement sanguin cumul à taux plein dérogation article 11B NGAP"

4. MAJORATIONS / DÉPLACEMENTS
Ex :
"MCI majoration coordination infirmière article 23 NGAP"
"IFD indemnité forfaitaire déplacement article 13 NGAP"
"MAU majoration acte unique NGAP"

5. CAS COMPLEXES (si présent)
- BSI :
"BSI forfait BSA BSB BSC article 12 NGAP cumul pansement dépendance"
- enfant :
"MIE majoration jeune enfant avenant 6 NGAP"
- psychiatrie :
"AMI 1,2 administration traitement oral troubles cognitifs NGAP"

6. CAS MULTI-JOURS
Si la question implique plusieurs moments :
→ créer une requête sur :
"différence cotation premier passage bilan plaie AMI 11 NGAP"
"cotation jour prélèvement sanguin en plus des soins NGAP"

────────────────────────────────────────
📏 RÈGLES DE RÉDACTION

- 1 sous-requête = 1 ligne
- pas de numérotation
- pas de texte explicatif
- phrases naturelles en français
- inclure les mots-clés exacts :
  AMI, AMX, AIS, NGAP, Titre XVI, Chapitre I, article 11B, article 13, article 23...
- utiliser les noms EXACTS des actes NGAP si possible
- être précis mais pas trop long

────────────────────────────────────────
⚠️ IMPORTANT

- NE PAS répondre à la question
- NE PAS calculer la cotation
- NE PAS donner de résultat
- FAIRE UNIQUEMENT des requêtes de recherche

────────────────────────────────────────
EXEMPLE

Question :
"pansement + injection à domicile"

Réponse :
Article 2 pansement plaies opératoires coefficient AMI Chapitre I Titre XVI NGAP
Article 1 injection sous-cutanée coefficient AMI NGAP
article 11B dispositions générales NGAP cumul actes même séance deuxième acte 50%
IFD indemnité forfaitaire déplacement article 13 NGAP
MCI majoration coordination infirmière article 23 NGAP
"""

def decompose_query(query: str, api_key: str) -> list[str]:
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = call_mistral_chat(messages, api_key, max_tokens=500).strip()

    sub_queries = [
        line.strip().lstrip("-•0123456789.) ")
        for line in response.split("\n")
        if line.strip() and len(line.strip()) > 10
    ]

    return sub_queries if sub_queries else [query]


REWRITE_SYSTEM_PROMPT = """Tu es un expert en recherche documentaire sur la réglementation des infirmiers libéraux en France (NGAP, circulaires Ameli, convention nationale).

Ta tâche : réécrire la question de l'utilisateur en UNE SEULE requête de recherche optimisée, naturelle et précise.

Objectif :
- améliorer la recherche documentaire dans un corpus juridique / conventionnel
- reformuler la question avec les bons termes techniques
- conserver strictement l'intention de l'utilisateur
- ne PAS décomposer en sous-questions
- ne PAS répondre à la question
- ne produire qu'UNE seule reformulation

Consignes de réécriture :
- Réécris la question comme une phrase naturelle en français
- Garde le sujet exact demandé par l'utilisateur
- Ajoute les termes techniques utiles si pertinents :
  AMI, AIS, AMX, IFD, MAU, MCI, NGAP, Titre XVI, Chapitre I, article 11B, dispositions générales, circulaire, convention nationale
- Pour un acte concret (stomie, agrafes, prélèvement, perfusion...), utilise le NOM EXACT tel qu'il apparaît dans la NGAP si possible
- Si la question porte sur un acte, ajoute si pertinent :
  coefficient, lettre clé, cotation, facturation, cumul, article NGAP
- Si la question implique une règle de cumul ou de facturation, ajoute explicitement :
  article 11B des dispositions générales NGAP
- Si la question semble porter sur une revalorisation ou une règle récente, ajoute si pertinent :
  circulaire Ameli, avenant, convention nationale
- N'invente aucun élément non lié à la question
- Ne mets ni numérotation, ni puce, ni explication

Exemples :
Question : "combien vaut un pansement de stomie ?"
Réécriture : "Quelle est la cotation NGAP du pansement de stomie dans le Titre XVI Chapitre I Article 2, avec son coefficient, sa lettre clé AMI ou SFI et ses règles éventuelles de cumul selon l'article 11B des dispositions générales ?"

Question : "est ce qu'on peut cumuler un prélèvement veineux"
Réécriture : "Quelles sont les règles de cumul NGAP du prélèvement par ponction veineuse directe dans le Titre XVI Chapitre I Article 1er, avec sa cotation, sa lettre clé AMI ou AMX ou SFI et la dérogation éventuelle à l'article 11B des dispositions générales ?"

Question : "revalorisation des actes infirmiers"
Réécriture : "Quelles sont les revalorisations des actes infirmiers libéraux prévues par la NGAP, les avenants, la convention nationale et les circulaires Ameli, notamment sur les coefficients AMI, AIS et les modalités de facturation ?"

Réponds UNIQUEMENT avec la requête réécrite, sans commentaire.
"""


def rewrite_query(query: str, api_key: str) -> str:
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = call_mistral_chat(messages, api_key, max_tokens=300).strip()

    # Nettoyage minimal pour garantir une seule requête propre
    rewritten = response.split("\n")[0].strip()
    rewritten = rewritten.lstrip("-•0123456789.) ").strip()

    return rewritten if rewritten else query




# ============================================================
# 3. SCORING ET RE-RANKING
# ============================================================

@dataclass
class ScoredResult:
    embedded_chunk: EmbeddedChunk
    cosine_score: float
    priority_score: float
    recency_score: float
    rerank_score: float = 0.0
    final_score: float = 0.0

    @property
    def chunk(self):
        return self.embedded_chunk.chunk

    @property
    def metadata(self):
        return self.embedded_chunk.chunk.metadata


def compute_priority_score(priorite: int | None) -> float:
    """
    Priorité juridique → score.
    Corpus actuel : NGAP=P1, circulaires=P2.
    """
    if priorite is None:
        return 0.5
    mapping = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.6,
               6: 0.5, 7: 0.4, 8: 0.3, 9: 0.2}
    return mapping.get(priorite, 0.5)


def compute_recency_score(annee: int | None, current_year: int = 2026) -> float:
    if annee is None:
        return 0.5
    age = current_year - annee
    if age <= 2:
        return 1.0
    elif age <= 5:
        return 0.8
    elif age <= 10:
        return 0.6
    else:
        return 0.4


def rerank(
    results: list[tuple[EmbeddedChunk, float]],
    query: str | None = None,
    api_key: str | None = None,
    use_mistral_rerank: bool = False,
    weight_cosine: float = 0.95,
    weight_rerank: float = 0.00,
    weight_priority: float = 0.025,
    weight_recency: float = 0.025,
) -> list[ScoredResult]:
    """
    Re-rank combinant jusqu'à 4 signaux.

    SANS Mistral Rerank (par défaut) :
        final = 0.95 * cosine + 0.025 * priority + 0.025 * recency
        Avec 4 PDF et 2 niveaux de priorité, le cosinus décide.

    AVEC Mistral Rerank (optionnel, use_mistral_rerank=True) :
        final = 0.50 * cosine + 0.40 * rerank + 0.05 * priority + 0.05 * recency
        Le cross-encoder apporte la compréhension fine query↔chunk.
    """
    if use_mistral_rerank:
        weight_cosine = 0.50
        weight_rerank = 0.40
        weight_priority = 0.05
        weight_recency = 0.05
    else:
        weight_cosine = weight_cosine + weight_rerank
        weight_rerank = 0.0

    # Appel Mistral Rerank si activé
    rerank_scores = {}
    if use_mistral_rerank and query and api_key:
        documents = [ec.chunk.text for ec, _ in results]
        if documents:
            rerank_results = call_mistral_rerank(query, documents, api_key)
            for item in rerank_results:
                rerank_scores[item["index"]] = item["relevance_score"]

    scored = []
    for i, (ec, cos_score) in enumerate(results):
        meta = ec.chunk.metadata
        prio_score = compute_priority_score(meta.priorite)
        rec_score = compute_recency_score(meta.annee)
        rr_score = rerank_scores.get(i, 0.0)

        final = (weight_cosine * cos_score
                 + weight_rerank * rr_score
                 + weight_priority * prio_score
                 + weight_recency * rec_score)

        scored.append(ScoredResult(
            embedded_chunk=ec,
            cosine_score=cos_score,
            priority_score=prio_score,
            recency_score=rec_score,
            rerank_score=rr_score,
            final_score=final,
        ))

    scored.sort(key=lambda x: x.final_score, reverse=True)
    return scored


# ============================================================
# 4. FILTRAGE ET DÉDOUBLONNAGE
# ============================================================

def deduplicate(
    results: list[ScoredResult],
    similarity_threshold: float = 0.93,
) -> list[ScoredResult]:
    """
    Supprime les chunks quasi-identiques.
    Seuil 0.93 (vs 0.97 avant) : chunks plus petits → doublons plus proches.
    """
    filtered = []
    for result in results:
        is_duplicate = False
        for kept in filtered:
            sim = cosine_similarity(
                result.embedded_chunk.embedding,
                kept.embedded_chunk.embedding,
            )
            if sim > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(result)
    return filtered


def filter_by_score(
    results: list[ScoredResult],
    min_cosine: float = 0.68,
) -> list[ScoredResult]:
    """
    Filtre par score cosinus minimum.
    Seuil 0.68 (vs 0.75 avant) : chunks plus petits peuvent avoir
    un score cosinus plus bas tout en étant pertinents.
    """
    return [r for r in results if r.cosine_score >= min_cosine]


# ============================================================
# 5. RETRIEVER AVANCÉ
# ============================================================

class AdvancedRetriever:
    """
    Pipeline de retrieval :
    1. Décompose la question en sous-questions (multi-query)
    2. Recherche cosinus large (30 par sous-query)
    3. Fusionne (meilleur score par chunk)
    4. Re-rank (cosinus 95% + métadonnées 5%)
    5. Filtre cosinus ≥ 0.68 + dédoublonne (seuil 0.93)
    6. Top K
    """

    def __init__(
        self,
        embedded_chunks: list[EmbeddedChunk],
        api_key: str | None = None,
        embedding_engine: EmbeddingEngine | None = None,
    ):
        self.embedded_chunks = embedded_chunks
        self.engine = embedding_engine or EmbeddingEngine(api_key=api_key)
        self.api_key = self.engine.api_key

    def retrieve(
    self,
    query: str,
    top_k: int = 10,
    retrieval_k: int = 30,
    use_multi_query: bool = True,
    use_rewrite: bool = True,
    use_mistral_rerank: bool = False,
    weight_cosine: float = 0.95,
    weight_rerank: float = 0.00,
    weight_priority: float = 0.025,
    weight_recency: float = 0.025,
    min_cosine: float = 0.68,
    dedup_threshold: float = 0.93,
    verbose: bool = True,
) -> list[ScoredResult]:

        # sécurité logique : multi-query prioritaire sur rewrite
        if use_multi_query and use_rewrite:
            use_rewrite = False

        # ── Étape 1 : Préparation des requêtes ──
        if use_multi_query:
            if verbose:
                print("  🔀 Décomposition multi-query...")
            sub_queries = decompose_query(query, self.api_key)

            if verbose:
                for i, sq in enumerate(sub_queries):
                    print(f"     [{i+1}] {sq[:90]}...")

        elif use_rewrite:
            if verbose:
                print("  ✍️ Rewriting query...")
            rewritten_query = rewrite_query(query, self.api_key)
            sub_queries = [rewritten_query]   # <- impératif

            if verbose:
                print(f"     [rewrite] {rewritten_query[:120]}...")

        else:
            sub_queries = [query]

            if verbose:
                print("  🔎 Recherche directe sans rewrite...")

        print(type(sub_queries))
        print(len(sub_queries))
        print(sub_queries[:3] if isinstance(sub_queries, list) else sub_queries[:50])
        
        # garde-fou
        sub_queries = [sq.strip() for sq in sub_queries if isinstance(sq, str) and sq.strip()]
        if not sub_queries:
            sub_queries = [query]

        # garde-fou
        sub_queries = [sq.strip() for sq in sub_queries if isinstance(sq, str) and sq.strip()]
        if not sub_queries:
            sub_queries = [query]

        # ── Étape 2 : Recherche cosinus large ──
        all_results = {}  # chunk_id → (EmbeddedChunk, best_score)

        for i, sq in enumerate(sub_queries):
            if verbose:
                print(f"  🔍 Recherche [{i+1}/{len(sub_queries)}] : {sq[:120]}...")

            query_vec = self.engine.embed_query(sq)
            results = search_similar(query_vec, self.embedded_chunks, top_k=retrieval_k)

            for ec, score in results:
                chunk_id = id(ec)
                if chunk_id not in all_results or score > all_results[chunk_id][1]:
                    all_results[chunk_id] = (ec, score)

        merged_results = list(all_results.values())

        if verbose:
            print(
                f"     → {len(merged_results)} chunks uniques "
                f"(depuis {len(sub_queries)} requête(s) × {retrieval_k})"
            )

        # ── Étape 3 : Re-ranking ──
        if verbose:
            mode = "Mistral Rerank + cosinus" if use_mistral_rerank else "cosinus dominant (95%)"
            print(f"  ⚖️  Re-ranking ({mode})...")

        ranked = rerank(
            merged_results,
            query=query,   # on rerank toujours sur la question originale
            api_key=self.api_key,
            use_mistral_rerank=use_mistral_rerank,
            weight_cosine=weight_cosine,
            weight_rerank=weight_rerank,
            weight_priority=weight_priority,
            weight_recency=weight_recency,
        )

        # ── Étape 4 : Filtrage + dédoublonnage ──
        ranked = filter_by_score(ranked, min_cosine)
        if verbose:
            print(f"     {len(ranked)} après filtre cosinus ≥ {min_cosine}")
        """
        ranked = deduplicate(ranked, dedup_threshold)
        if verbose:
            print(f"     {len(ranked)} après dédoublonnage (seuil {dedup_threshold})")
        """
        final = ranked[:top_k]
        if verbose:
            print(f"  ✅ {len(final)} résultats finaux")

        return final


# ============================================================
# 6. TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 TEST RETRIEVAL AVANCÉ")
    print("=" * 60)

    try:
        from embeddings import load_embedded_chunks
        embedded = load_embedded_chunks("data/embedded_chunks.json")
        print(f"  📦 {len(embedded)} chunks chargés")

        retriever = AdvancedRetriever(embedded)

        queries = [
            "Pansement de colostomie + ablation de 7 agrafes au domicile, quelle cotation ?",
            "Prélèvement sanguin + ECBU au domicile, comment coter ?",
        ]

        for query in queries:
            print(f"\n{'─' * 60}")
            print(f"🔎 {query}\n")
            results = retriever.retrieve(query, top_k=8)

            for i, r in enumerate(results):
                m = r.metadata
                src = m.nom_pdf or m.source or "?"
                preview = r.chunk.text[:120].replace("\n", " ").strip()
                print(f"  [{i+1}] cos={r.cosine_score:.3f} final={r.final_score:.3f} "
                      f"| {src} P{m.priorite}")
                print(f"      {preview}...")
                print()

    except Exception as e:
        print(f"  ⚠️  {e}")
        print("  → Lance d'abord le pipeline d'ingestion + embeddings")