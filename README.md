# RAG KEYNOVE

Système de **Retrieval-Augmented Generation (RAG)** spécialisé dans la réglementation infirmière française et les règles de tarification NGAP (Nomenclature Générale des Actes Professionnels).

Il permet de répondre à des questions complexes sur la cotation et la facturation des actes infirmiers en combinant recherche sémantique dans les textes officiels et génération par LLM.

---

## Fonctionnalités

- **Ingestion de PDF** : extraction et nettoyage du texte, détection automatique des tableaux NGAP, enrichissement avec les métadonnées du registre
- **Chunking flexible** : 3 stratégies — naive (taille fixe), overlap (fenêtre glissante), semantic (limites légales : articles, titres, chapitres)
- **Embedding** : via Mistral (`mistral-embed`, 1024 dims) ou OpenAI (`text-embedding-3-small/large`), avec cache SHA256 sur disque
- **Retrieval avancé** :
  - Réécriture de la requête + décomposition multi-query via Mistral LLM
  - Recherche cosinus sur les vecteurs
  - Re-ranking via `mistral-rerank-latest` (cross-encoder)
  - Score combiné : similarité cosinus + priorité légale + récence + rerank
  - Déduplication par similarité vectorielle
- **Génération** : `mistral-large-latest` avec prompt expert NGAP, température basse (0.2), citations sources obligatoires

---

## Architecture

```
data/RAW/           →   ingestion.py   →   chunking.py   →   embeddings.py   →   data/embedded_chunks.json
   (PDFs)               (extraction)       (segmentation)     (vectorisation)       (index vectoriel)

                                                                                          │
                                                                                    Question utilisateur
                                                                                          │
                                                                                    retrieval.py
                                                                                 (multi-query + rerank)
                                                                                          │
                                                                                       rag.py
                                                                                    (génération)
                                                                                          │
                                                                                     RAGResponse
```

---

## Structure du projet

```
RAG_KEYNOVE/
├── rag.py                       # Agent RAG end-to-end (point d'entrée principal)
├── pipeline.py                  # CLI : ingestion → chunking → embedding → sauvegarde
├── ingestion.py                 # Chargement et nettoyage des PDFs
├── chunking.py                  # Segmentation des documents
├── cleaning_text.py             # Normalisation du texte (JORF, coupures, espaces)
├── embeddings.py                # Moteur d'embedding + cache + recherche cosinus
├── retrieval.py                 # Retrieval avancé, re-ranking, scoring
├── data/
│   ├── RAW/                     # PDFs source (circulaires, NGAP, avenants…)
│   ├── documents_registry.xlsx  # Registre des documents (titre, type, année, priorité)
│   ├── embedded_chunks.json     # Index vectoriel pré-calculé (743 chunks)
│   └── embeddings_cache/        # Cache des embeddings par hash
└── notebook/
    ├── POC_API_LEGIFRANCE.ipynb
    ├── exploration_cleaning_pdf.ipynb
    └── test_search.ipynb
```

---

## Installation

```bash
# Cloner le repo
git clone <url>
cd RAG_KEYNOVE

# Créer l'environnement virtuel (uv recommandé)
uv venv
source .venv/bin/activate

# Installer les dépendances
uv pip install pdfplumber pandas numpy scipy openpyxl mistralai openai python-dotenv
```

Créer un fichier `.env` à la racine :

```env
MISTRAL_API_KEY=<votre_clé>
OPENAI_API_KEY=<votre_clé>   # optionnel
```

---

## Utilisation

### 1. Construire l'index vectoriel

```bash
# Avec les paramètres par défaut (Mistral, chunking sémantique, taille 800)
python pipeline.py

# Avec OpenAI et chunking avec overlap
python pipeline.py --provider openai --model text-embedding-3-large --strategy overlap --chunk-size 800 --overlap 150
```

### 2. Interroger le système

```python
from rag import RAGAgent
from embeddings import load_embedded_chunks

embedded = load_embedded_chunks("data/embedded_chunks.json")
agent = RAGAgent(embedded)

query = """
Vous prenez en charge la petite Camille, 4 ans, qui suite à une chute
s'est fracturé l'avant-bras, a été opérée, et est rentrée ce jour à la maison
avec un fixateur externe. L'ordonnance demande 3 passages par semaine pour
réfection de pansement et surveillance des broches. Quelle sera votre cotation ?
"""

response = agent.ask(query, top_k=100)
print(response.answer)
response.print_sources()
```

---

## Paramètres clés

| Composant | Paramètre | Valeur par défaut |
|-----------|-----------|-------------------|
| Chunking | `chunk_size` | 800 caractères |
| Chunking | `overlap` | 150 caractères |
| Chunking | `strategy` | `SEMANTIC` |
| Embedding | `provider` | `mistral` |
| Retrieval | `retrieval_k` | 30 candidats |
| Retrieval | `min_cosine` | 0.68 |
| Retrieval | `dedup_threshold` | 0.93 |
| Retrieval | `top_k` | 100 résultats |
| Génération | `model` | `mistral-large-latest` |
| Génération | `temperature` | 0.2 |
| Génération | `max_tokens` | 2000 |

---

## Hiérarchie légale

Les documents sont pondérés selon leur rang dans la hiérarchie des normes :

| Priorité | Type |
|----------|------|
| 1 | Loi |
| 2 | Code |
| 3 | Décret |
| 4 | Arrêté |
| 6 | Avenant / Décision |
| 7 | Circulaire |
| 8 | Charte |
| 9 | Avis |

---

## Documents sources

Le corpus initial couvre :
- Circulaire CIR-34/2019 — Avenant n°6
- Circulaire CIR-30/2024 — Prescriptions médicales
- Document NGAP de référence (lettre-clés, coefficients, actes infirmiers)
- Textes réglementaires complémentaires

Ajoutez vos propres PDFs dans `data/RAW/` et référencez-les dans `data/documents_registry.xlsx`, puis relancez `pipeline.py`.

---

## État du projet

- Version : v1
- Corpus : 4 documents, 743 chunks
- Stack : Python 3.12, Mistral AI, uv
