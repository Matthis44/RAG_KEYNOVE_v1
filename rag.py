"""
ÉTAPE 5 : GÉNÉRATION RAG
==========================
Assemble tout le pipeline : retrieval avancé + génération avec Mistral Large.

Le LLM reçoit :
- La question de l'utilisateur
- Les chunks pertinents retrouvés (avec métadonnées)
- Des instructions pour citer ses sources et respecter la hiérarchie juridique

Usage :
    from step5_rag import RAGAgent
    agent = RAGAgent(embedded_chunks)
    response = agent.ask("Quels vaccins un infirmier peut-il prescrire ?")
"""

import json
from dataclasses import dataclass

from embeddings import EmbeddedChunk, EmbeddingEngine, load_embedded_chunks
from retrieval import AdvancedRetriever, ScoredResult, call_mistral_chat


# ============================================================
# 1. PROMPT ENGINEERING
# ============================================================

SYSTEM_PROMPT = """Tu es un assistant juridique expert en réglementation des infirmiers libéraux en France.

Tu réponds aux questions en te basant EXCLUSIVEMENT sur les extraits de documents fournis dans le contexte.

RÈGLES IMPÉRATIVES :
1. RÉPONDS D'ABORD DIRECTEMENT à la question en 2-3 phrases, détaille si nécessaire.
2. SI LA QUESTION PORTE SUR UNE COTATION, affiche la formule. Exemple de format attendu :
   → AMI 3 + AMI 2 / 2 + IFD (+/- IK)
   Explique chaque élément (quel acte, quel article). S'il manque des informations, précise-le.
3. CORRESPONDANCE EXACTE DES ACTES : utilise la DÉSIGNATION EXACTE de l'acte dans la NGAP, pas une interprétation par analogie. Par exemple :
   - Une colostomie est une STOMIE → utilise "Pansement de stomie" (pas "Pansement de fistule digestive")
   - Une ablation de 7 agrafes → utilise "Ablation de fils ou d'agrafes, dix ou moins"
   - Un prélèvement sanguin → utilise "Prélèvement par ponction veineuse directe"
   Ne confonds PAS deux actes différents de la NGAP même s'ils sont proches anatomiquement.
4. ACTES MULTIPLES : quand la situation mentionne PLUSIEURS actes distincts, cote CHAQUE acte SÉPARÉMENT avec son propre coefficient AMI. Ne fusionne JAMAIS deux actes différents en une seule cotation.
5. RÈGLE DE CUMUL (article 11B) : quand deux actes AMI sont réalisés dans la même séance, le deuxième acte est généralement facturé à 50% (noté "/ 2" dans la formule), SAUF dérogations explicites prévues par la NGAP. Cherche dans les extraits si une dérogation s'applique.
6. BASE-TOI UNIQUEMENT sur les extraits fournis. Si l'information exacte manque, dis-le — ne devine pas et ne donne pas de "valeurs indicatives".
7. CITE TES SOURCES entre crochets après chaque affirmation : [Titre (Type, Année)]
8. HIÉRARCHIE JURIDIQUE : quand plusieurs sources existent, la plus autoritaire prime :
   Loi > Code > Décret > Arrêté > Avenant > Circulaire > Avis
9. PRÉCISION : utilise les termes techniques exacts (AMI, AIS, BSI, NGAP, IFD, IK, MAU, MCI...).
10. Pour les COTATIONS : donne la formule exacte avec les coefficients et lettres-clés trouvés dans les extraits. Ne donne que les valeurs chiffrées qui apparaissent explicitement dans le contexte.
11. CONCISION : ne répète pas les mêmes informations. Pas de tableaux récapitulatifs sauf si demandé.
12. Si tu ne trouves PAS l'information dans les extraits, dis simplement "Cette information n'apparaît pas dans les documents fournis" au lieu d'inventer.
"""


def build_context(results: list[ScoredResult]) -> str:
    """
    Construit le contexte à injecter dans le prompt du LLM.
    
    Chaque chunk est présenté avec ses métadonnées pour que le LLM
    puisse citer ses sources et respecter la hiérarchie juridique.
    """
    context_parts = []

    for i, r in enumerate(results):
        m = r.metadata
        header_parts = []

        if m.is_enriched:
            header_parts.append(f"Titre: {m.titre}")
            header_parts.append(f"Type: {m.type_document} (priorité: {m.priorite_label})")
            header_parts.append(f"Année: {m.annee}")
            header_parts.append(f"Source: {m.source_juridique}")
            header_parts.append(f"URL: {m.url}")
        else:
            header_parts.append(f"Fichier: {m.source}")

        header = " | ".join(header_parts)

        context_parts.append(
            f"--- Extrait {i+1} [{header}] ---\n"
            f"{r.chunk.text}\n"
        )

    return "\n".join(context_parts)


def build_user_prompt(query: str, context: str) -> str:
    """Construit le message utilisateur avec le contexte."""
    return (
        f"CONTEXTE (extraits de documents juridiques) :\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"QUESTION : {query}\n\n"
        f"CONSIGNES : Commence par la réponse directe en 2-3 phrases. "
        f"Puis détaille uniquement si nécessaire. "
        f"Ne donne que les valeurs qui apparaissent explicitement dans les extraits. "
        f"Cite tes sources [Titre (Type, Année)] après chaque affirmation."
    )


# ============================================================
# 2. AGENT RAG
# ============================================================

@dataclass
class RAGResponse:
    """Réponse complète du RAG avec traçabilité."""
    answer: str                          # Réponse générée par le LLM
    query: str                           # Question originale
    rewritten_query: str                 # Question reformulée
    sources: list[ScoredResult]          # Chunks utilisés comme contexte
    model: str                           # Modèle utilisé pour la génération

    def print_sources(self):
        """Affiche les sources utilisées."""
        print(f"\n📚 Sources utilisées ({len(self.sources)}) :\n")
        for i, r in enumerate(self.sources):
            m = r.metadata
            if m.is_enriched:
                print(f"  [{i+1}] {m.titre[:65]}...")
                print(f"      {m.type_document} | P{m.priorite} ({m.priorite_label}) | {m.annee}")
                print(f"      Score: cos={r.cosine_score:.4f} final={r.final_score:.4f}")
                print(f"      URL: {m.url}")
            else:
                print(f"  [{i+1}] {m.source} (score={r.cosine_score:.4f})")
            print()


class RAGAgent:
    """
    Agent RAG complet : retrieval avancé + génération.

    Usage :
        agent = RAGAgent(embedded_chunks)
        response = agent.ask("Quels vaccins un infirmier peut-il prescrire ?")
        print(response.answer)
        response.print_sources()
    """

    def __init__(
        self,
        embedded_chunks: list[EmbeddedChunk],
        api_key: str | None = None,
        generation_model: str = "mistral-large-latest",
    ):
        self.engine = EmbeddingEngine(api_key=api_key)
        self.retriever = AdvancedRetriever(
            embedded_chunks,
            embedding_engine=self.engine,
        )
        self.generation_model = generation_model
        self.api_key = self.engine.api_key

    def ask(
        self,
        query: str,
        top_k: int = 100,
        retrieval_k: int = 150,
        use_multi_query: bool = True,
        verbose: bool = True,
        **retrieval_kwargs,
    ) -> RAGResponse:
        """
        Pose une question au RAG.

        Pipeline complet :
        1. Multi-query decomposition + retrieval parallèle
        2. Re-ranking + dédoublonnage
        3. Construction du contexte
        4. Génération de la réponse avec Mistral Large
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"🤖 RAG — Traitement de la question")
            print(f"{'=' * 60}\n")

        # Étape 1 : Retrieval avancé (multi-query géré à l'intérieur)
        if verbose:
            print("📥 Étape 1 : Retrieval\n")

        results = self.retriever.retrieve(
            query,
            top_k=top_k,
            retrieval_k=retrieval_k,
            use_multi_query=use_multi_query,
            verbose=verbose,
            **retrieval_kwargs,
        )

        # Capture les sub-queries pour la traçabilité
        rewritten = query

        # Étape 2 : Construction du contexte
        if verbose:
            print(f"\n📄 Étape 2 : Construction du contexte ({len(results)} chunks)...")
        context = build_context(results)
        user_prompt = build_user_prompt(query, context)

        if verbose:
            context_size = len(context)
            print(f"     Contexte : {context_size} caractères (~{context_size // 4} tokens)")

        # Étape 3 : Génération
        if verbose:
            print(f"\n🧠 Étape 3 : Génération ({self.generation_model})...\n")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        answer = call_mistral_chat(
            messages,
            api_key=self.api_key,
            model=self.generation_model,
            temperature=0.2,
            max_tokens=2000,
        )

        # Construit la réponse
        response = RAGResponse(
            answer=answer,
            query=query,
            rewritten_query=rewritten,
            sources=results,
            model=self.generation_model,
        )

        if verbose:
            print(f"{'─' * 60}")
            print(f"📝 RÉPONSE :\n")
            print(answer)
            print(f"\n{'─' * 60}")

        return response



# ============================================================
# 3. TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 TEST ÉTAPE 5 : RAG COMPLET")
    print("=" * 60)

    try:
        # Charge les embeddings
        embedded = load_embedded_chunks("data/embedded_chunks.json")

        # Crée l'agent
        agent = RAGAgent(embedded)

        # Question de test
        query = "Vous prenez en charge la petite Camille, 4 ans, qui suite à une chute s’est fracturé l’avant-bras, a été opérée, et est rentré ce jour à la maison avec un fixateur externe. L’ordonnance vous demande de passer 3 fois par semaine pour réfection de pansement et surveillance des broches du fixateur : Quelle sera votre cotation ?"
        # Lance le RAG
        response = agent.ask(query, top_k=100)

        # Affiche les sources
        response.print_sources()

    except Exception as e:
        print(f"\n⚠️  Erreur : {e}")
        print("\nVérifie :")
        print("  1. Tes embeddings sont dans data/embedded_chunks.json")
        print("  2. Ta clé API est dans .env (MISTRAL_API_KEY=...)")