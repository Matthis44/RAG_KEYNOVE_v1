"""
ÉTAPE 3 : EMBEDDINGS
======================
Transforme les chunks de texte en vecteurs numériques via l'API Mistral ou OpenAI.

Concepts clés :
- Un embedding = une liste de floats représentant le "sens" du texte
- Des textes sémantiquement proches → vecteurs proches (cosinus élevé)
- On vectorise à la fois les chunks (offline, une seule fois)
  et la query utilisateur (online, à chaque question)

Modèles supportés :
  Mistral : mistral-embed (1024 dimensions)
  OpenAI  : text-embedding-3-small (1536 dim) | text-embedding-3-large (3072 dim)
            text-embedding-ada-002 (1536 dim)

Usage :
    engine = EmbeddingEngine()                          # Mistral par défaut
    engine = EmbeddingEngine(provider="openai")         # OpenAI text-embedding-3-small
    engine = EmbeddingEngine(provider="openai",
                config=EmbeddingConfig(model="text-embedding-3-large"))
    vectors = engine.embed_chunk(chunks)
    query_vec = engine.embed_query("quels vaccins peut faire un infirmier ?")
"""
import os
import json
import math
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
 
import urllib.request
import urllib.error
 
from chunking import Chunk

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROVIDER_DEFAULTS = {
    "mistral": {
        "model":      "mistral-embed",
        "api_url":    "https://api.mistral.ai/v1/embeddings",
        "dimensions": 1024,
    },
    "openai": {
        "model":      "text-embedding-3-small",
        "api_url":    "https://api.openai.com/v1/embeddings",
        "dimensions": 1536,
    },
}


@dataclass
class EmbeddingConfig:
    model:             str   = "mistral-embed"
    api_url:           str   = "https://api.mistral.ai/v1/embeddings"
    dimensions:        int   = 1024
    batch_size:        int   = 10
    rate_limit_delay:  float = 0.5
    cache_dir:         str   = "data/embeddings_cache"


# ============================================================
# 2. SIMILARITÉ COSINUS 
# ==========================================================

def dot_product(vec_a : list[float], vec_b : list[float]):
    return sum(a*b for a,b in zip(vec_a,vec_b))

def vector_norm(vec : list[float]):
    return math.sqrt(sum(v*v for v in vec))

def cosine_similarity(vec_a : list[float], vec_b : list[float]):
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product(vec_a,vec_b) / (norm_a*norm_b)


# ============================================================
# 3. APPELS API
# ============================================================

def _post_embedding_api(
    texts: list[str],
    api_key: str,
    config: EmbeddingConfig,
    provider: str,
) -> list[list[float]]:
    """Appel générique vers n'importe quelle API compatible OpenAI/Mistral."""
    payload = json.dumps({
        "model": config.model,
        "input": texts,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(
        config.api_url,
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(
            f"Erreur API {provider} ({e.code}) : {body}\n"
            f"Vérifie ta clé API."
        )

    sorted_data = sorted(result["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


# Aliases nommés pour la lisibilité
def call_mistral_embedding_api(
    texts: list[str],
    api_key: str,
    config: EmbeddingConfig = EmbeddingConfig(),
) -> list[list[float]]:
    return _post_embedding_api(texts, api_key, config, provider="Mistral")


def call_openai_embedding_api(
    texts: list[str],
    api_key: str,
    config: EmbeddingConfig | None = None,
) -> list[list[float]]:
    if config is None:
        defaults = PROVIDER_DEFAULTS["openai"]
        config = EmbeddingConfig(
            model=defaults["model"],
            api_url=defaults["api_url"],
            dimensions=defaults["dimensions"],
        )
    return _post_embedding_api(texts, api_key, config, provider="OpenAI")


# ============================================================
# 4. MOTEUR D'EMBEDDINGS
# ============================================================

@dataclass
class EmbeddedChunk:
    
    chunk : Chunk
    embedding : list[float]

    def __repr__(self):
        src = self.chunk.metadata.source
        return (f"EmbeddedChunk('{src}' "
                f"[{self.chunk.chunk_index}/{self.chunk.total_chunks}], "
                f"dim={len(self.embedding)})")
    
class EmbeddingEngine:

    PROVIDERS = ("mistral", "openai")

    def __init__(
        self,
        provider: str = "mistral",
        api_key: str | None = None,
        config: EmbeddingConfig | None = None,
    ):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Provider '{provider}' inconnu. Choix : {self.PROVIDERS}")

        self.provider = provider
        self.api_key = api_key or self._load_api_key()

        if config is None:
            defaults = PROVIDER_DEFAULTS[provider]
            config = EmbeddingConfig(
                model=defaults["model"],
                api_url=defaults["api_url"],
                dimensions=defaults["dimensions"],
            )
        self.config = config
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def _load_api_key(self) -> str:
        env_var = "MISTRAL_API_KEY" if self.provider == "mistral" else "OPENAI_API_KEY"
        key = os.environ.get(env_var)
        if key:
            return key

        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{env_var}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

        raise ValueError(
            f"Clé API introuvable pour le provider '{self.provider}'.\n"
            f"Définis {env_var} dans ton environnement ou dans un fichier .env"
        )
    

    
    def _cache_key(self, text: str) -> str:
        """Hash MD5 du texte pour identifier un embedding en cache."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
 
    def _get_from_cache(self, text: str) -> list[float] | None:
        """Cherche un embedding en cache. Retourne None si absent."""
        cache_file = Path(self.config.cache_dir) / f"{self._cache_key(text)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None
 
    def _save_to_cache(self, text: str, embedding: list[float]):
        """Sauvegarde un embedding en cache."""
        cache_file = Path(self.config.cache_dir) / f"{self._cache_key(text)}.json"
        cache_file.write_text(json.dumps(embedding))

    def embed_texts(self, texts : list[str])-> list[list[float]]:
        
        embeddings = [None]* len(texts)
        to_embed = []

        for i, text in enumerate(texts):
            cached=self._get_from_cache(text)
            if cached is not None:
                embeddings[i]=cached
            else:
                to_embed.append((i,text))
        
        cached_count = len(texts) - len(to_embed)

        if cached_count > 0:
            print(f"{cached_count} embeddings trouvés en cache")

        if not to_embed:
            return embeddings

        print(f"{len(to_embed)} textes à vectoriser via API {self.provider}...")
        batch_size = self.config.batch_size

        for batch_start in range(0, len(to_embed), batch_size):
            batch = to_embed[batch_start:batch_start+batch_size]
            batch_texts = [text for _, text in batch]

            batch_embeddings = _post_embedding_api(
                batch_texts, self.api_key, self.config, provider=self.provider
            )

            for (orig_idx, text), embedding in zip(batch, batch_embeddings):
                embeddings[orig_idx]=embedding
                self._save_to_cache(text,embedding)

            done = min(batch_start+batch_size, len(to_embed))
            print(f"-->{done}/{len(to_embed)} vectorisés")

            if batch_start + batch_size < len(to_embed):
                    time.sleep(self.config.rate_limit_delay)

        return embeddings
    
    def embed_chunk(self, chunks : list[Chunk]) -> list[EmbeddedChunk]:
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        return[
            EmbeddedChunk(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]
    
    def embed_query(self, query: str) -> list[float]:
        result = call_mistral_embedding_api(
            [query], self.api_key, self.config
        )
        return result[0]


# ============================================================
# 5. RECHERCHE PAR SIMILARITÉ
# ============================================================

def search_similar(
        query_embedding : list[float],
        embedded_chunks : list[EmbeddedChunk],
        top_k : int = 5,
)-> list[tuple[EmbeddedChunk,float]]:
    
    scored=[]
    for ec in embedded_chunks:
        score = cosine_similarity(query_embedding, ec.embedding)
        scored.append((ec, score))
    
    scored.sort(key=lambda x : x[1], reverse = True)

    return scored[:top_k]



# ============================================================
# 6. SAUVEGARDE / CHARGEMENT DES EMBEDDINGS
# ============================================================
 

def save_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
    filepath: str = "data/embedded_chunks.json"
):
    """
    Sauvegarde les chunks vectorisés en JSON.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
 
    data = []
    for ec in embedded_chunks:
        data.append({
            "text": ec.chunk.text,
            "embedding": ec.embedding,
            "chunk_index": ec.chunk.chunk_index,
            "total_chunks": ec.chunk.total_chunks,
            "strategy": ec.chunk.strategy,
            "metadata": ec.chunk.metadata.to_dict(),
        })
 
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
 
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"💾 {len(data)} embeddings sauvegardés → {filepath} ({size_mb:.1f} MB)")
 
 
def load_embedded_chunks(filepath: str = "data/embedded_chunks.json") -> list[EmbeddedChunk]:
    """Charge les chunks vectorisés depuis un fichier JSON."""
    from ingestion import DocumentMetadata
 
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    embedded = []
    for item in data:
        meta = DocumentMetadata(**item["metadata"])
        chunk = Chunk(
            text=item["text"],
            metadata=meta,
            chunk_index=item["chunk_index"],
            total_chunks=item["total_chunks"],
            strategy=item["strategy"],
        )
        embedded.append(EmbeddedChunk(chunk=chunk, embedding=item["embedding"]))
 
    print(f"{len(embedded)} embeddings chargés depuis {filepath}")
    return embedded

# ============================================================
# 7. TEST
# ============================================================
 
if __name__ == "__main__":
    print("=" * 60)
    print("TEST : EMBEDDINGS")
    print("=" * 60)
 
    # Test de la similarité cosinus (sans API)
    print("\n--- Test similarité cosinus (sans API) ---\n")
 
    # Vecteurs fictifs pour démontrer le concept
    vec_vaccination = [0.8, 0.1, -0.3, 0.5, 0.2]
    vec_vaccin_2 =    [0.75, 0.15, -0.25, 0.45, 0.18]  # similaire
    vec_tarif =       [-0.2, 0.7, 0.4, -0.1, 0.6]       # différent
 
    sim_1 = cosine_similarity(vec_vaccination, vec_vaccin_2)
    sim_2 = cosine_similarity(vec_vaccination, vec_tarif)
 
    print(f"  vaccination ↔ vaccination_2 : {sim_1:.4f} (devrait être ~1.0)")
    print(f"  vaccination ↔ tarif         : {sim_2:.4f} (devrait être ~0.0)")
 
    # Test avec l'API (nécessite la clé)
    print("\n--- Test API Mistral ---\n")
    try:
        engine = EmbeddingEngine()
        print(f"  ✅ Clé API chargée")
 
        # Vectorise 3 textes de test
        test_texts = [
            "L'infirmier peut vacciner contre la grippe.",
            "La vaccination antigrippale par les infirmiers est autorisée.",
            "Les tarifs des indemnités kilométriques sont fixés par convention.",
        ]
 
        embeddings = engine.embed_texts(test_texts)
        print(f"  ✅ {len(embeddings)} embeddings générés (dim={len(embeddings[0])})")
 
        # Similarités
        sim_vacc = cosine_similarity(embeddings[0], embeddings[1])
        sim_diff = cosine_similarity(embeddings[0], embeddings[2])
        print(f"\n  Vaccination ↔ Vaccination  : {sim_vacc:.4f} (devrait être élevé)")
        print(f"  Vaccination ↔ Tarifs       : {sim_diff:.4f} (devrait être bas)")
 
        # Test query
        query = "quels vaccins peut faire un infirmier ?"
        query_vec = engine.embed_query(query)
        print(f"\n  Query vectorisée : dim={len(query_vec)}")
 
        for i, text in enumerate(test_texts):
            sim = cosine_similarity(query_vec, embeddings[i])
            print(f"  Query ↔ texte[{i}] : {sim:.4f} — {text[:50]}...")
 
    except ValueError as e:
        print(f"  ⚠️  {e}")
        print("\n  Le test cosinus fonctionne. Pour le test API,")
        print("  crée un fichier .env avec : MISTRAL_API_KEY=ta_clé")