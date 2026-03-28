"""
PIPELINE COMPLET RAG
=====================
Ingestion → Chunking → Embedding → Sauvegarde

Usage :
    python pipeline.py
    python pipeline.py --provider openai
    python pipeline.py --provider openai --model text-embedding-3-large
    python pipeline.py --strategy overlap --chunk-size 800 --overlap 150
    python pipeline.py --output data/my_embeddings.json
"""

import argparse
import time
from pathlib import Path

from ingestion import ingest_directory
from chunking import chunk_documents, ChunkingStrategy, Chunk
from embeddings import EmbeddingEngine, EmbeddingConfig, save_embedded_chunks


# ============================================================
# CONFIG PAR DÉFAUT
# ============================================================

DATA_DIR      = "data/RAW"
REGISTRY_PATH = "data/documents_registry.xlsx"
OUTPUT_PATH   = "data/embedded_chunks.json"

STRATEGY      = ChunkingStrategy.SEMANTIC
CHUNK_SIZE    = 800
OVERLAP       = 150
MIN_CHUNK     = 150
MAX_CHUNK     = 1500
PROVIDER      = "mistral"


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    data_dir: str = DATA_DIR,
    registry_path: str = REGISTRY_PATH,
    output_path: str = OUTPUT_PATH,
    strategy: ChunkingStrategy = STRATEGY,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    min_chunk_size: int = MIN_CHUNK,
    max_chunk_size: int = MAX_CHUNK,
    provider: str = PROVIDER,
    model: str | None = None,
):
    t0 = time.time()

    # ── Étape 1 : Ingestion ──────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — INGESTION")
    print("=" * 60)

    reg = registry_path if Path(registry_path).exists() else None
    if not reg:
        print(f"  Registre '{registry_path}' absent, métadonnées basiques uniquement.")

    documents = ingest_directory(data_dir, registry_path=reg)

    if not documents:
        print("Aucun document trouvé. Vérifie le dossier data/RAW.")
        return

    t1 = time.time()
    print(f"\n  {len(documents)} document(s) ingérés en {t1 - t0:.1f}s")

    # ── Étape 2 : Chunking ───────────────────────────────────
    print("\n" + "=" * 60)
    print(f"ÉTAPE 2 — CHUNKING  (stratégie : {strategy.value.upper()})")
    print("=" * 60)

    chunks = chunk_documents(
        documents,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )

    sizes = [len(c) for c in chunks]
    t2 = time.time()
    print(f"\n  {len(chunks)} chunks générés en {t2 - t1:.1f}s")
    print(f"  Tailles — min: {min(sizes)}, max: {max(sizes)}, moy: {sum(sizes)//len(sizes)}")

    # ── Étape 3 : Embedding ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"ÉTAPE 3 — EMBEDDING  (provider : {provider.upper()})")
    print("=" * 60)

    config = EmbeddingConfig(model=model) if model else None
    engine = EmbeddingEngine(provider=provider, config=config)
    embedded_chunks = engine.embed_chunk(chunks)

    t3 = time.time()
    print(f"\n  {len(embedded_chunks)} chunks vectorisés en {t3 - t2:.1f}s")
    print(f"  Dimensions : {len(embedded_chunks[0].embedding)}")

    # ── Étape 4 : Sauvegarde ─────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 4 — SAUVEGARDE")
    print("=" * 60)

    save_embedded_chunks(embedded_chunks, filepath=output_path)

    total = time.time() - t0
    print(f"\n  Pipeline terminé en {total:.1f}s")
    print(f"  Fichier : {output_path}")

    return embedded_chunks


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG : ingestion → chunking → embedding")

    parser.add_argument("--data-dir",      default=DATA_DIR,      help="Dossier contenant les PDF")
    parser.add_argument("--registry",      default=REGISTRY_PATH, help="Registre Excel des métadonnées")
    parser.add_argument("--output",        default=OUTPUT_PATH,   help="Fichier JSON de sortie")
    parser.add_argument("--provider",      default=PROVIDER,      choices=["mistral", "openai"])
    parser.add_argument("--model",         default=None,          help="Modèle d'embedding (ex: text-embedding-3-large)")
    parser.add_argument("--strategy",      default="semantic",    choices=["naive", "overlap", "semantic"])
    parser.add_argument("--chunk-size",    default=CHUNK_SIZE,    type=int)
    parser.add_argument("--overlap",       default=OVERLAP,       type=int)
    parser.add_argument("--min-chunk",     default=MIN_CHUNK,     type=int)
    parser.add_argument("--max-chunk",     default=MAX_CHUNK,     type=int)

    args = parser.parse_args()

    strategy_map = {
        "naive":    ChunkingStrategy.NAIVE,
        "overlap":  ChunkingStrategy.OVERLAP,
        "semantic": ChunkingStrategy.SEMANTIC,
    }

    run_pipeline(
        data_dir=args.data_dir,
        registry_path=args.registry,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        strategy=strategy_map[args.strategy],
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_size=args.min_chunk,
        max_chunk_size=args.max_chunk,
    )
