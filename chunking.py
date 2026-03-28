"""
ÉTAPE 2 : CHUNKING
====================
Découpe les documents en morceaux (chunks) pour le RAG.

3 stratégies : Naïf, Overlap, Sémantique.
Optimisé pour les textes juridiques infirmiers (NGAP + circulaires).
"""

import re
from enum import Enum
from dataclasses import dataclass
from ingestion import Document, DocumentMetadata


# ============================================================
# 1. DATACLASS CHUNK
# ============================================================

@dataclass
class Chunk:
    text: str
    metadata: DocumentMetadata
    chunk_index: int = 0
    total_chunks: int = 0
    strategy: str = ""

    def __repr__(self):
        src = self.metadata.source or "?"
        return f"Chunk({src} [{self.chunk_index}/{self.total_chunks}], {len(self.text)} car.)"

    def __len__(self):
        return len(self.text)


class ChunkingStrategy(Enum):
    NAIVE = "naive"
    OVERLAP = "overlap"
    SEMANTIC = "semantic"


# ============================================================
# 2. STRATÉGIES DE BASE
# ============================================================

def chunk_naive(text: str, chunk_size: int = 1000) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) >= 20:
            chunks.append(chunk)
    return chunks


def chunk_overlap(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError(f"L'overlap ({overlap}) doit être < chunk_size ({chunk_size})")
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= 20:
            chunks.append(chunk)
        start += step
    return chunks


# ============================================================
# 3. STRATÉGIE SÉMANTIQUE — PATTERNS DE SPLIT
# ============================================================


LEGAL_PATTERNS = [
    # Titres majeurs
    r"(?=(?:^|\n)\s*TITRE\s+[IVX\d]+)",
    r"(?=(?:^|\n)\s*CHAPITRE\s+[IVX\d]+)",
    r"(?=(?:^|\n)\s*PARTIE\s+[IVX\d]+)",
    r"(?=(?:^|\n)\s*SECTION\s+\d+)",
    r"(?=(?:^|\n)\s*Préambule\b)",
    # Articles NGAP / juridiques
    r"(?=(?:^|\n)\s*Art(?:icle)?\.?\s+\d+)",
    r"(?=(?:^|\n)\s*Art\.\s+1er\b)",
    r"(?=(?:^|\n)\s*Article\s+[LRD]\.\s*\d+)",
    r"(?=(?:^|\n)\s*Article\s+unique\b)",
    r"(?=(?:^|\n)\s*Article\s+préliminaire\b)",
    # Annexes
    r"(?=(?:^|\n)\s*ANNEXE[S]?\s*[IVX\d]*)",
    r"(?=(?:^|\n)\s*Annexe[s]?\s*[IVX\d]*)",
]

CIRCULAIRE_PATTERNS = [
    r"(?=(?:^|\n)\s*\d+\.\d+(?:\.\d+)?\s+[A-ZÉÈÀÊÔÎÛÇa-z])",
    r"(?=(?:^|\n)\s*\d+/\s+[A-ZÉÈÀÊÔÎÛÇa-z])",
]

LEGAL_SPLIT_REGEX = "|".join(LEGAL_PATTERNS + CIRCULAIRE_PATTERNS)



# ============================================================
# 5. HELPERS GÉNÉRAUX
# ============================================================

def _is_structural_header(text: str) -> bool:
    """
    Détecte si un texte est un titre structurel pur.
    Ces fragments doivent être fusionnés vers l'avant.
    """
    stripped = text.strip()
    if len(stripped) < 120:
        if re.match(r"^(TITRE|CHAPITRE|PARTIE|SECTION|ANNEXE)\s+", stripped, re.IGNORECASE):
            return True
        if re.match(r"^Art(icle)?\.?\s+\d+", stripped) and "\n" not in stripped:
            return True
    return False


def _looks_like_ngap_extracted_text(text: str) -> bool:
    """
    Détecte si le texte ressemble à une extraction NGAP structurée
    contenant des actes déjà reconstruits sous forme :
    - ... : coefficient X, lettre clé Y
    """
    if not text:
        return False

    t = text.lower()

    signals = [
        "lettre clé",
        "coefficient",
        "soins infirmiers",
        "article 1er",
        "ami",
        "sfi",
        "amx",
        "chapitre i - soins de pratique courante",
    ]
    score = sum(1 for s in signals if s in t)

    act_count = len(
        re.findall(
            r"^\s*-\s+.+?:\s*coefficient\s+\d+[,.]?\d*",
            text,
            flags=re.MULTILINE | re.IGNORECASE
        )
    )

    return score >= 3 and act_count >= 3


def _is_condition_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    starters = [
        "Condition :",
        "Conditions :",
        "Cet acte",
        "Cette cotation",
        "Dans le cadre",
        "Lorsque",
        "Par dérogation",
        "Les deux cotations",
        "L'acte comprend",
        "Prescription",
        "Indications",
        "Au-delà",
        "facturés,",
    ]
    return any(s.startswith(x) for x in starters)


def _is_article_or_header_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    return bool(
        re.match(r"^(TITRE|CHAPITRE|PARTIE|SECTION)\s+", s, re.IGNORECASE)
        or re.match(r"^Article\b", s, re.IGNORECASE)
    )


# Ligne d'acte NGAP reconstruite
ACT_LINE_REGEX = re.compile(
    r"^\s*-\s+.+?:\s*coefficient\s+\d+[,.]?\d*(?:,\s*lettre clé\s+.+)?\s*$",
    re.IGNORECASE
)


# ============================================================
# 6. CHUNKING SÉMANTIQUE GÉNÉRAL
# ============================================================

def chunk_semantic(
    text: str,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1500,
    fallback_chunk_size: int = 800,
    fallback_overlap: int = 150,
) -> list[str]:
    """
    Découpe aux frontières naturelles du texte juridique.
    """
    raw_sections = re.split(LEGAL_SPLIT_REGEX, text, flags=re.MULTILINE)
    raw_sections = [s for s in raw_sections if s.strip()]

    if len(raw_sections) <= 1:
        raw_sections = text.split("\n\n")
        raw_sections = [s for s in raw_sections if s.strip()]

    if len(raw_sections) <= 1:
        return chunk_overlap(text, fallback_chunk_size, fallback_overlap)

    merged = []
    buffer = ""

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        if not buffer:
            buffer = section
            continue

        if _is_structural_header(buffer):
            buffer = buffer + "\n\n" + section
            continue

        if len(section) < min_chunk_size and not _is_structural_header(section):
            buffer = buffer + "\n\n" + section
            continue

        merged.append(buffer)
        buffer = section

    if buffer:
        if merged and len(buffer) < min_chunk_size and not _is_structural_header(buffer):
            merged[-1] = merged[-1] + "\n\n" + buffer
        else:
            merged.append(buffer)

    final_chunks = []
    for chunk in merged:
        if len(chunk) > max_chunk_size:
            sub_chunks = chunk_overlap(chunk, fallback_chunk_size, fallback_overlap)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    final_chunks = [c for c in final_chunks if len(c.strip()) >= 30]
    return final_chunks


# ============================================================
# 7. CHUNKING SPÉCIFIQUE NGAP — PAR ACTE
# ============================================================

def chunk_ngap_acts(
    text: str,
    max_chunk_size: int = 1800,
    min_chunk_size: int = 80,
) -> list[str]:
    """
    Chunking spécialisé NGAP :
    - un acte = un chunk
    - conserve le contexte CHAPITRE / Article
    - rattache les conditions à l'acte précédent
    - garde aussi le texte orphelin (mode lossless)
    """
    lines = [line.rstrip() for line in text.splitlines()]

    chunks = []
    current_context = []
    current_act = None
    orphan_buffer = []

    def flush_act():
        nonlocal current_act
        if current_act:
            chunk = "\n".join(current_act).strip()
            if chunk:
                chunks.append(chunk)
            current_act = None

    def flush_orphans():
        nonlocal orphan_buffer
        if orphan_buffer:
            chunk = "\n".join(orphan_buffer).strip()
            if chunk:
                chunks.append(chunk)
            orphan_buffer = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Ignore certains séparateurs techniques de page
        if stripped.startswith("===== PAGE"):
            continue

        # Contexte structurel majeur
        if re.match(r"^(TITRE|CHAPITRE|PARTIE|SECTION)\s+", stripped, re.IGNORECASE):
            flush_act()
            flush_orphans()
            current_context = [stripped]
            continue

        # Article
        if re.match(r"^Article\b", stripped, re.IGNORECASE):
            flush_act()
            flush_orphans()

            if current_context:
                # garde le dernier contexte majeur + l'article courant
                current_context = current_context[:1] + [stripped]
            else:
                current_context = [stripped]
            continue

        # Nouvelle ligne d'acte
        if ACT_LINE_REGEX.match(stripped):
            flush_act()
            flush_orphans()

            current_act = []
            if current_context:
                current_act.extend(current_context)
            current_act.append(stripped)
            continue

        # Ligne de condition / continuation rattachée à l'acte courant
        if current_act is not None:
            current_act.append(stripped)
            continue

        # Texte hors acte => on le garde
        orphan_buffer.append(stripped)

    # flush final
    flush_act()
    flush_orphans()

    # Fusion légère si chunks trop courts
    merged = []
    buffer = ""

    for chunk in chunks:
        if not buffer:
            buffer = chunk
            continue

        # On évite de fusionner deux chunks "acte" différents
        buffer_is_act = bool(ACT_LINE_REGEX.search(buffer))
        chunk_is_act = bool(ACT_LINE_REGEX.search(chunk))

        if len(buffer) < min_chunk_size and not (buffer_is_act and chunk_is_act):
            buffer = buffer + "\n\n" + chunk
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    # Redécoupe si chunk trop gros
    final_chunks = []
    for chunk in merged:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # fallback prudent
            final_chunks.extend(chunk_overlap(chunk, chunk_size=max_chunk_size, overlap=150))

    final_chunks = [c.strip() for c in final_chunks if len(c.strip()) >= 30]
    return final_chunks

# ============================================================
# 8. FONCTIONS PRINCIPALES
# ============================================================

def chunk_document(
    doc,
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    chunk_size: int = 800,
    overlap: int = 150,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1500,
) -> list[Chunk]:
    """
    Point d'entrée principal.
    Si le document ressemble à une NGAP extraite proprement,
    on utilise le chunking métier par acte.
    Sinon, on reste sur le chunking juridique sémantique.
    """
    if strategy == ChunkingStrategy.NAIVE:
        raw_chunks = chunk_naive(doc.text, chunk_size)

    elif strategy == ChunkingStrategy.OVERLAP:
        raw_chunks = chunk_overlap(doc.text, chunk_size, overlap)

    elif strategy == ChunkingStrategy.SEMANTIC:
        if _looks_like_ngap_extracted_text(doc.text):
            raw_chunks = chunk_ngap_acts(
                doc.text,
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
            )
        else:
            raw_chunks = chunk_semantic(
                doc.text,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                fallback_chunk_size=chunk_size,
                fallback_overlap=overlap,
            )
    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")

    total = len(raw_chunks)
    return [
        Chunk(
            text=text,
            metadata=doc.metadata,
            chunk_index=i,
            total_chunks=total,
            strategy=strategy.value,
        )
        for i, text in enumerate(raw_chunks)
    ]


def chunk_documents(
    docs: list,
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    **kwargs
) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, strategy=strategy, **kwargs)
        all_chunks.extend(chunks)
    return all_chunks