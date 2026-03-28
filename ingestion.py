"""
ÉTAPE 1 : INGESTION DES DOCUMENTS — v5
======================================
But :
- Lire des PDF
- Extraire un texte propre
- Enrichir avec les métadonnées du registre Excel
- Gérer 2 modes :
    1) mode générique pour tous les PDF
    2) mode spécialisé NGAP pour les tableaux de cotations

AMÉLIORATIONS v5
────────────────
- Détection automatique NGAP / générique
- Extraction texte propre avec suppression des en-têtes/pieds répétitifs
- Séparateurs de pages explicites
- Schéma de colonnes dynamique
- Gestion des tableaux NGAP coupés
- Réparation des LK éclatées (ex: "AMI ou" + "AMX")
- Support des layouts :
    - cotation standard
    - anatomo-patho
    - thermalisme
- Fallback sûr pour les autres PDF
"""

import os
import re
import pdfplumber
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from cleaning_text import clean_text


# ============================================================
# 1. DATACLASSES
# ============================================================

@dataclass
class DocumentMetadata:
    source: str = ""
    filepath: str = ""
    extension: str = ""
    size_bytes: int = 0
    modified_time: str = ""
    registry_id: int | None = None
    type_document: str = ""
    titre: str = ""
    annee: int | None = None
    source_juridique: str = ""
    url: str = ""
    nom_pdf: str = ""
    theme: str = ""
    priorite: int | None = None
    commentaire: str = ""
    extraction_mode: str = "generic"   # generic | ngap

    @property
    def is_enriched(self) -> bool:
        return self.registry_id is not None

    @property
    def priorite_label(self) -> str:
        labels = {
            1: "Loi", 2: "Code", 3: "Décret", 4: "Arrêté",
            6: "Avenant / Décision", 7: "Circulaire", 8: "Charte", 9: "Avis",
        }
        return labels.get(self.priorite, "Inconnu") if self.priorite else "Non classé"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    text: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    def __repr__(self):
        src = self.metadata.source or "?"
        tag = " ✓registre" if self.metadata.is_enriched else ""
        mode = f" [{self.metadata.extraction_mode}]"
        return f"Document('{src}'{tag}{mode}, {len(self.text)} car.)"

    def __len__(self):
        return len(self.text)


# ============================================================
# 2. REGISTRE
# ============================================================

def load_registry(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Registre introuvable : {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Format '{ext}' non supporté. Utilise .xlsx ou .csv")

    colonnes_requises = {
        "id", "type_document", "titre", "annee",
        "source", "url", "nom_pdf", "theme", "priorite"
    }
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}\nTrouvées : {list(df.columns)}")

    df["id"] = df["id"].astype(int)
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce").astype("Int64")
    df["priorite"] = pd.to_numeric(df["priorite"], errors="coerce").astype("Int64")
    df["commentaire"] = df.get("commentaire", pd.Series(dtype=str)).fillna("")

    records = df.to_dict(orient="records")
    for r in records:
        if pd.isna(r["annee"]):
            r["annee"] = None
        if pd.isna(r["priorite"]):
            r["priorite"] = None
    return records


def build_registry_index(registry: list[dict]) -> dict[str, dict]:
    return {entry["nom_pdf"]: entry for entry in registry}


# ============================================================
# 3. CONSTANTES NGAP
# ============================================================

LETTRES_CLES = {
    "AMI", "AMX", "SFI", "AIS", "AMK", "AMO", "AMY",
    "AMP", "K", "SF", "SP", "Z", "TO", "ORT", "POD", "POT",
    "BSA", "BSB", "BSC", "DI", "PAI", "ARL", "NMI",
    "RAB", "RAV", "TER", "PLL", "RPE", "RPB", "DRA",
    "APM", "RAM", "RAO", "RSC", "RSM", "RIC", "RIM",
    "VIC", "VIM", "VSC", "VSM", "TMP", "TMO", "TMK",
    "FMN", "RQD", "TE2", "KMB", "P", "TLS", "TLD", "TLL"
}

_EXCLUSION_KEYWORDS = {
    "stations thermales", "orientations thérapeutiques",
    "annees", "décisions uncam", "decisions uncam",
    "publication au journal", "forfait bsi pour",
    "combinaisons", "groupe 1", "groupe 2", "groupe 3",
    "groupe 4", "interventions",
}

CONDITION_STARTERS = [
    "Par dérogation", "Une séance", "Ce bilan", "Cet acte",
    "La cotation", "Le contrôle", "En cas de", "Pour ",
    "Lorsque ", "Au-delà", "Toute", "Il est", "Les ",
    "Sous réserve", "Cette cotation", "Facturation",
    "La facturation", "La prescription", "Prescription",
    "Indications", "Ce protocole", "Conditions",
]


# ============================================================
# 4. HELPERS TEXTE
# ============================================================

def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_unicode(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\xad": "",
        "•": "- ",
        "◦": "- ",
        "▪": "- ",
        "■": "- ",
        "►": "- ",
        "": "- ",
        "": "- ",
        "\r": "\n",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _remove_repeated_headers_footers(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        s = line.strip()

        if not s:
            cleaned.append("")
            continue

        if re.fullmatch(r"Version en vigueur du \d{2}/\d{2}/\d{4}", s):
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue
        if re.match(r"^(Première|Deuxième|Troisième|Quatrième)\s+partie\s*[:\-–]", s):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def _fix_broken_hyphenation(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    text = re.sub(
        r"([a-zàâçéèêëîïôûùüÿñæœ,;])\n([a-zàâçéèêëîïôûùüÿñæœ])",
        r"\1 \2",
        text,
        flags=re.IGNORECASE
    )
    return text


def _normalize_linebreaks(text: str) -> str:
    if not text:
        return ""

    text = _normalize_unicode(text)
    text = _remove_repeated_headers_footers(text)
    text = _fix_broken_hyphenation(text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text_clean(page) -> str:
    text = page.extract_text(
        x_tolerance=2,
        y_tolerance=3,
        layout=True,
    )

    if not text:
        text = page.extract_text(
            x_tolerance=3,
            y_tolerance=4,
            layout=False,
        ) or ""

    return _normalize_linebreaks(text)


def _dedupe_near_duplicate_lines(text: str) -> str:
    if not text:
        return ""

    out = []
    prev = None
    for line in text.splitlines():
        s = line.strip()
        if s and s == prev:
            continue
        out.append(line)
        prev = s if s else prev

    return "\n".join(out).strip()


# ============================================================
# 5. HELPERS NGAP
# ============================================================

def _has_lettre_cle(text: str) -> bool:
    if not text:
        return False
    upper = text.upper()
    for lk in LETTRES_CLES:
        if re.search(rf"\b{re.escape(lk)}\b", upper):
            return True
    return False


def _is_numeric_coeff(value: str) -> bool:
    if not value:
        return False

    for line in value.split("\n"):
        cleaned = line.strip().replace(" ", "")
        if cleaned and re.match(r"^\d+[,.]?\d*$", cleaned):
            return True
    return False


def _is_anatpath_code(value: str) -> bool:
    if not value:
        return False
    return bool(re.match(r"^0{1,2}\d{2,3}$", value.strip()))


def _clean_lettre_cle(lk: str) -> str:
    if not lk:
        return ""
    lk = lk.replace("\n", " ")
    lk = re.sub(r"\s+", " ", lk).strip()
    lk = re.sub(r"[;,:\-–—]+$", "", lk).strip()
    return _normalize_multi_lk(lk)


def _is_partial_lk(text: str) -> bool:
    if not text:
        return False
    t = _cell(text).upper().strip()
    return bool(re.match(r"^(?:[A-Z]{1,4})\s+OU$", t))


def _is_single_lk_token(text: str) -> bool:
    if not text:
        return False
    t = _cell(text).upper().strip()
    return t in LETTRES_CLES


def _normalize_header_name(text: str) -> str:
    if not text:
        return ""
    t = _cell(text).lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = (
        t.replace("é", "e").replace("è", "e").replace("ê", "e")
         .replace("à", "a").replace("â", "a")
         .replace("ç", "c")
         .replace("ù", "u")
         .replace("î", "i").replace("ï", "i")
         .replace("ô", "o")
    )
    return t


def _looks_like_designation_header(text: str) -> bool:
    t = _normalize_header_name(text)
    if not t or len(t) > 60:
        return False
    return (
        "designation" in t
        or "designation de l'acte" in t
        or "libelle" in t
    )


def _looks_like_coeff_header(text: str) -> bool:
    t = _normalize_header_name(text)
    if not t:
        return False
    # Limiter aux textes courts pour éviter les faux positifs (ex: désignation contenant "coefficient")
    return ("coefficient" in t or t.startswith("coeff")) and len(t) < 30


def _looks_like_lk_header(text: str) -> bool:
    t = _normalize_header_name(text)
    if not t or len(t) > 30:
        return False
    return "lettre cle" in t or "lettre" in t or "cle" in t


def _is_ngap_document(text: str) -> bool:
    if not text:
        return False

    t = _normalize_header_name(text).upper()

    signals = [
        "NGAP",
        "LETTRE CLE",
        "COEFFICIENT",
        "AMI",
        "AMK",
        "AIS",
        "SFI",
        "AMX",
    ]
    score = sum(1 for s in signals if s in t)
    return score >= 2


# ============================================================
# 6. SCHÉMA DE TABLEAU DYNAMIQUE
# ============================================================

def _infer_table_schema(table: list, inherited_schema: dict | None = None) -> dict:
    default_schema = {
        "designation": 0,
        "coefficient": 1,
        "lettre_cle": 2,
        "header_row": None,
    }

    if not table:
        return inherited_schema or default_schema

    max_cols = max(len(row) for row in table if row)

    for ri, row in enumerate(table[:3]):
        if not row:
            continue

        row_extended = list(row) + [""] * (max_cols - len(row))
        col_designation = None
        col_coeff = None
        col_lk = None

        for ci, cell in enumerate(row_extended):
            if _looks_like_designation_header(cell):
                col_designation = ci
            elif _looks_like_coeff_header(cell):
                col_coeff = ci
            elif _looks_like_lk_header(cell):
                col_lk = ci

        if col_designation is not None or col_coeff is not None or col_lk is not None:
            return {
                "designation": 0 if col_designation is None else col_designation,
                "coefficient": 1 if col_coeff is None else col_coeff,
                "lettre_cle": 2 if col_lk is None else col_lk,
                "header_row": ri,
            }

    col_stats = []
    for ci in range(max_cols):
        values = []
        for row in table[:10]:
            if row and ci < len(row):
                values.append(_cell(row[ci]))
            else:
                values.append("")

        coeff_score = sum(1 for v in values if _is_numeric_coeff(v))
        lk_score = sum(1 for v in values if _has_lettre_cle(v))
        text_score = sum(1 for v in values if v and not _is_numeric_coeff(v) and not _has_lettre_cle(v))

        col_stats.append({
            "col": ci,
            "coeff_score": coeff_score,
            "lk_score": lk_score,
            "text_score": text_score,
        })

    coeff_col = max(col_stats, key=lambda x: x["coeff_score"])["col"]
    lk_col = max(col_stats, key=lambda x: x["lk_score"])["col"]

    candidates = [c for c in col_stats if c["col"] not in {coeff_col, lk_col}]
    designation_col = max(candidates, key=lambda x: x["text_score"])["col"] if candidates else 0

    schema = {
        "designation": designation_col,
        "coefficient": coeff_col,
        "lettre_cle": lk_col,
        "header_row": None,
    }

    if inherited_schema:
        # si pas d'header détecté, on hérite du précédent pour les tables coupées
        schema = {
            "designation": inherited_schema.get("designation", schema["designation"]),
            "coefficient": inherited_schema.get("coefficient", schema["coefficient"]),
            "lettre_cle": inherited_schema.get("lettre_cle", schema["lettre_cle"]),
            "header_row": None,
        }

    return schema


# ============================================================
# 7. CLASSIFICATION DES TABLEAUX
# ============================================================

def _classify_table(table: list, inherited_schema: dict | None = None) -> str:
    if not table or len(table) < 1:
        return "ignore"

    all_text = " ".join(
        _cell(c).lower()
        for row in table[:3] for c in (row or [])
    )
    if any(kw in all_text for kw in _EXCLUSION_KEYWORDS):
        return "ignore"

    for row in table:
        if not row or len(row) < 4:
            continue
        c0 = _cell(row[0])
        c3 = _cell(row[3])
        if _is_anatpath_code(c0) and _is_numeric_coeff(c3):
            return "anatpath"

    for row in table:
        if not row or len(row) < 2:
            continue
        c1 = _cell(row[1])
        if "maximum de" in c1.lower() and "séance" in c1.lower():
            return "thermal"

    schema = _infer_table_schema(table, inherited_schema=inherited_schema)
    c_col = schema["coefficient"]
    lk_col = schema["lettre_cle"]

    for row in table:
        if not row:
            continue
        needed = max(c_col, lk_col) + 1
        row_extended = list(row) + [""] * max(0, needed - len(row))
        coeff_cell = _cell(row_extended[c_col])
        lk_cell = _cell(row_extended[lk_col])

        if _is_numeric_coeff(coeff_cell) and _has_lettre_cle(lk_cell):
            return "cotation_ngap"

    for row in table:
        if not row:
            continue
        cells = [_cell(c) for c in row]
        if any(_is_numeric_coeff(c) for c in cells) and any(_has_lettre_cle(c) for c in cells):
            return "cotation_ngap"

    # Continuation d'un tableau NGAP depuis la page précédente :
    # si le schéma est hérité et qu'on trouve des coefficients numériques dans la colonne attendue
    if inherited_schema is not None:
        c_col_inh = inherited_schema.get("coefficient", 1)
        for row in table:
            if not row:
                continue
            if c_col_inh < len(row) and _is_numeric_coeff(_cell(row[c_col_inh])):
                return "cotation_ngap"

    return "ignore"


# ============================================================
# 8. RÉPARATION DES TABLEAUX
# ============================================================

def _extract_packed_lks(table: list, schema: dict) -> list[str]:
    if not table or not table[0]:
        return []

    lk_col = schema["lettre_cle"]

    for ri in range(min(2, len(table))):
        row = table[ri]
        if not row:
            continue
        if lk_col >= len(row):
            continue

        cell_str = _cell(row[lk_col])
        if not cell_str:
            continue

        lines = [l.strip() for l in cell_str.split("\n") if l.strip()]
        lk_lines = [
            l for l in lines
            if _has_lettre_cle(l)
            and "lettre" not in l.lower()
            and "désignation" not in l.lower()
            and "coefficient" not in l.lower()
        ]

        if len(lk_lines) >= 2:
            return lk_lines
        if len(lk_lines) >= 1 and "lettre" in cell_str.lower():
            return lk_lines

    return []


def _merge_continuation_lines(raw_lines: list[str]) -> list[str]:
    continuation_starters = (
        "de ", "la ", "par ", "y compris", "sur ", "et ", "à ", "ou ",
        "des ", "d'", "du ", "avec ", "dans ", "en ", "au ", "aux ",
        "chez ", "dont ", "pour ", "sous ", "sans ", "entre ",
    )

    merged = []
    for line in raw_lines:
        is_continuation = (
            merged and line and (
                line[0].islower()
                or any(line.startswith(s) for s in continuation_starters)
            )
        )
        if is_continuation:
            merged[-1] += " " + line
        else:
            merged.append(line)
    return merged


def _split_acts_conditions(merged: list[str], num_data: int) -> tuple[list[str], list[str]]:
    acts = []
    conditions = []

    for block in merged:
        is_cond = (
            any(block.startswith(s) for s in CONDITION_STARTERS)
            or (len(block) > 250 and "coefficient" not in block.lower())
        )
        if is_cond:
            conditions.append(block)
        else:
            acts.append(block)

    if len(acts) == num_data:
        return acts, conditions

    return merged[:num_data], merged[num_data:]

def _resolve_lk(
    lk_lines: list[str],
    idx: int,
    packed_lks: list[str],
    packed_idx: int
) -> str:
    """
    Résout la lettre clé pour une cotation donnée.

    Priorité :
    1. la LK de la ligne courante (lk_lines[idx])
    2. la LK empaquetée (packed_lks[packed_idx])
    3. la première LK valide trouvée dans lk_lines
    """
    # 1) LK explicite sur la ligne courante
    if idx < len(lk_lines):
        lk = lk_lines[idx]
        if _has_lettre_cle(lk):
            return _clean_lettre_cle(lk)

    # 2) LK empaquetée héritée de l'en-tête / bloc
    if packed_idx < len(packed_lks):
        lk = packed_lks[packed_idx]
        if _has_lettre_cle(lk):
            return _clean_lettre_cle(lk)

    # 3) fallback : première LK exploitable de la cellule
    for lk in lk_lines:
        if _has_lettre_cle(lk):
            return _clean_lettre_cle(lk)

    return ""

def _extract_packed_designations_v3(table: list, schema: dict) -> list[str]:
    if not table or not table[0]:
        return []

    d_col = schema["designation"]
    c_col = schema["coefficient"]

    if d_col >= len(table[0]):
        return []

    h0 = _cell(table[0][d_col])
    h1 = _cell(table[0][c_col]) if c_col < len(table[0]) else ""

    lines = [l.strip() for l in h0.split("\n") if l.strip()]
    if len(lines) < 2:
        return []

    is_header = "designation" in _normalize_header_name(lines[0])
    has_coeff_header = "coefficient" in _normalize_header_name(h1)

    if not (is_header or has_coeff_header):
        return []

    raw_lines = lines[1:] if is_header else lines

    num_data = 0
    for row in table[1:]:
        if not row:
            continue
        if c_col < len(row) and _is_numeric_coeff(_cell(row[c_col])):
            num_data += 1

    merged = _merge_continuation_lines(raw_lines)

    if len(merged) <= num_data:
        return merged

    acts, conditions = _split_acts_conditions(merged, num_data)
    return acts + conditions


def _repair_split_letter_keys(table: list, schema: dict) -> list:
    """
    Répare les LK éclatées sur plusieurs lignes :
      row i   : [designation, coeff, 'AMI ou']
      row i+1 : [condition ou '', '', 'AMX ou']
      row i+2 : ['', '', 'SFI']

    => row i : LK = 'AMI ou AMX ou SFI'
    """
    if not table:
        return table

    d_col = schema["designation"]
    c_col = schema["coefficient"]
    lk_col = schema["lettre_cle"]

    repaired = [list(row) if row else [] for row in table]

    i = 0
    while i < len(repaired):
        row = repaired[i]
        if not row:
            i += 1
            continue

        needed = max(d_col, c_col, lk_col) + 1
        while len(row) < needed:
            row.append("")

        curr_desc = _cell(row[d_col])
        curr_coeff = _cell(row[c_col])
        curr_lk = _cell(row[lk_col])

        if curr_desc and _is_numeric_coeff(curr_coeff) and curr_lk:
            lk_parts = [curr_lk.strip()]
            j = i + 1

            while j < len(repaired):
                nxt = repaired[j]
                if not nxt:
                    break

                while len(nxt) < needed:
                    nxt.append("")

                next_desc = _cell(nxt[d_col])
                next_coeff = _cell(nxt[c_col])
                next_lk = _cell(nxt[lk_col])

                # on continue tant qu'il n'y a pas de nouveau coeff
                if _is_numeric_coeff(next_coeff):
                    break

                # si la "LK" ressemble à une continuation
                if next_lk and (
                    _is_single_lk_token(next_lk)
                    or _is_partial_lk(next_lk)
                    or _has_lettre_cle(next_lk)
                ):
                    lk_parts.append(next_lk.strip())
                    nxt[lk_col] = ""
                    j += 1
                    continue

                # sinon on s'arrête
                break

            merged_lk = " ".join(lk_parts)
            merged_lk = re.sub(r"\s+", " ", merged_lk).strip()
            row[lk_col] = merged_lk

        i += 1

    return repaired

def _normalize_multi_lk(lk: str) -> str:
    if not lk:
        return ""

    tokens = re.split(r"\s+", lk.strip())
    out = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "ou":
            out.append("ou")
        elif tok.upper() in LETTRES_CLES:
            out.append(tok.upper())

    # reconstruire proprement
    cleaned = []
    prev = None
    for tok in out:
        if tok == prev:
            continue
        cleaned.append(tok)
        prev = tok

    return " ".join(cleaned).strip()


# ============================================================
# 9. FORMATEURS NGAP
# ============================================================

def _format_cotation_ngap(
    table: list,
    inherited_schema: dict | None = None
) -> tuple[list[str], dict]:
    if not table:
        return [], (inherited_schema or {"designation": 0, "coefficient": 1, "lettre_cle": 2, "header_row": None})

    schema = _infer_table_schema(table, inherited_schema=inherited_schema)
    table = _repair_split_letter_keys(table, schema)

    d_col = schema["designation"]
    c_col = schema["coefficient"]
    lk_col = schema["lettre_cle"]
    header_row = schema["header_row"]

    packed_lks = _extract_packed_lks(table, schema)
    packed_designations = _extract_packed_designations_v3(table, schema)

    results = []
    last_designation = ""
    packed_desig_idx = 0
    packed_lk_idx = 0

    start_idx = 0
    if header_row is not None:
        start_idx = header_row + 1

    for ri in range(start_idx, len(table)):
        row = table[ri]
        if not row:
            continue

        needed = max(d_col, c_col, lk_col) + 1
        row_extended = list(row) + [""] * max(0, needed - len(row))

        raw_desig_text = _cell(row_extended[d_col])
        raw_coeff = _cell(row_extended[c_col])
        raw_lk = _cell(row_extended[lk_col])

        if not raw_coeff or not _is_numeric_coeff(raw_coeff):
            text = raw_desig_text.replace("\n", " ").strip()

            # Si la cellule LK contient plusieurs LK empilées (ex: 'AMI\nAMI ou SFI'),
            # la première appartient à une continuation cross-page : consommer cet index.
            if raw_lk and _has_lettre_cle(raw_lk):
                lk_cell_lines = [
                    l.strip() for l in raw_lk.split("\n")
                    if l.strip() and _has_lettre_cle(l.strip())
                ]
                if len(lk_cell_lines) > 1:
                    packed_lk_idx += 1

            # Si LK présente, pas de coeff et pas de désignation → ligne LK-only cross-page
            if raw_lk and _has_lettre_cle(raw_lk) and not text:
                continue

            if text and len(text) > 10:
                is_condition = (
                    any(text.startswith(s) for s in CONDITION_STARTERS)
                    or text.startswith("Cet acte")
                    or text.startswith("Cette cotation")
                    or text.startswith("Dans le cadre")
                    or text.startswith("Par dérogation")
                )

                if results and is_condition:
                    results[-1] += f" | condition: {text}"
                elif results and text and text[0].islower():
                    # Continuation de désignation depuis la page précédente
                    results[-1] += " " + text
                else:
                    # texte libre hors cotation
                    results.append(text)
            continue

        coeff_lines = [
            c.strip() for c in raw_coeff.split("\n")
            
            if c.strip() and re.match(r"^\d+[,.]?\d*$", c.strip())
        ]

        lk_lines = [l.strip() for l in raw_lk.split("\n") if l.strip()] if raw_lk else []
        designation = raw_desig_text.replace("\n", " ").strip()

        if designation and "designation" in _normalize_header_name(designation):
            designation = ""

        if not designation:
            if packed_desig_idx < len(packed_designations):
                designation = packed_designations[packed_desig_idx]
                packed_desig_idx += 1
            elif last_designation:
                for ci, coeff_val in enumerate(coeff_lines):
                    lk = _resolve_lk(lk_lines, ci, packed_lks, packed_lk_idx)
                    packed_lk_idx += 1
                    lk_part = f", lettre clé {lk}" if lk else ""
                    results.append(f"- {last_designation} (variante) : coefficient {coeff_val}{lk_part}")
                continue

        if not designation:
            designation = "(acte sans libellé extrait)"

        if len(coeff_lines) > 1:
            desig_raw_lines = [d.strip() for d in raw_desig_text.split("\n") if d.strip()]
            short_lines = [d for d in desig_raw_lines if len(d) < 150]

            for ci, coeff_val in enumerate(coeff_lines):
                lk = _resolve_lk(lk_lines, ci, packed_lks, packed_lk_idx)
                packed_lk_idx += 1

                if ci < len(short_lines) and len(short_lines) >= len(coeff_lines):
                    offset = len(short_lines) - len(coeff_lines)
                    d = short_lines[offset + ci]
                elif ci < len(desig_raw_lines):
                    d = desig_raw_lines[ci]
                else:
                    d = designation

                lk_part = f", lettre clé {lk}" if lk else ""
                results.append(f"- {d} : coefficient {coeff_val}{lk_part}")
                last_designation = d
        else:
            coeff_val = coeff_lines[0]
            # Coeff unique : la cellule LK entière est une seule LK (peut s'étaler sur plusieurs lignes)
            if raw_lk and len(lk_lines) > 1:
                lk = _clean_lettre_cle(raw_lk)
            else:
                lk = _resolve_lk(lk_lines, 0, packed_lks, packed_lk_idx)
            packed_lk_idx += 1
            last_designation = designation
            lk_part = f", lettre clé {lk}" if lk else ""
            results.append(f"- {designation} : coefficient {coeff_val}{lk_part}")

    return results, schema


def _format_anatpath(table: list) -> list[str]:
    results = []
    for row in table:
        if not row or len(row) < 4:
            continue

        c0 = _cell(row[0])
        c1 = _cell(row[1]).replace("\n", " ")
        c3 = _cell(row[3])

        if _is_anatpath_code(c0) and _is_numeric_coeff(c3):
            desc = c1
            if len(desc) > 200:
                desc = desc[:200] + "…"
            results.append(f"- Acte {c0} : {desc} — coefficient P {c3}")

    return results


def _format_thermal(table: list) -> list[str]:
    results = []
    for row in table:
        if not row or len(row) < 2:
            continue

        c0 = _cell(row[0]).replace("\n", " ")
        c1 = _cell(row[1]).replace("\n", " ")

        if c0 and c1 and "séance" in c1.lower():
            results.append(f"- {c0} : honoraires {c1}")

    return results

def _extract_partial_lk_suffix(line: str) -> str | None:
    """
    Détecte une fin de ligne avec LK incomplète :
    ex: '... lettre clé AMI ou'
        '... lettre clé AMI ou AMX ou'
    """
    if not line:
        return None

    m = re.search(
        r"(lettre clé\s+(?:[A-Z]{1,4}\s+ou(?:\s+[A-Z]{1,4}\s+ou)*))\s*$",
        line,
        flags=re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    return None


def _line_is_lk_continuation(line: str) -> bool:
    """
    Détecte si une ligne de début de page ressemble à une continuation de LK.
    Ex:
      'SFI'
      'AMX ou'
      'SFI'
    """
    if not line:
        return False

    s = line.strip().upper()
    if s in LETTRES_CLES:
        return True
    if re.fullmatch(r"[A-Z]{1,4}\s+OU", s):
        return True
    return False


def _merge_page_boundary_lk(prev_page_text: str, next_page_text: str) -> tuple[str, str]:
    """
    Si la dernière ligne de la page N finit par une LK incomplète
    et que la première ligne utile de la page N+1 est une continuation,
    on les recolle.
    """
    if not prev_page_text or not next_page_text:
        return prev_page_text, next_page_text

    prev_lines = prev_page_text.splitlines()
    next_lines = next_page_text.splitlines()

    if not prev_lines or not next_lines:
        return prev_page_text, next_page_text

    # dernière ligne non vide de la page précédente
    prev_idx = None
    for i in range(len(prev_lines) - 1, -1, -1):
        if prev_lines[i].strip():
            prev_idx = i
            break

    if prev_idx is None:
        return prev_page_text, next_page_text

    partial = _extract_partial_lk_suffix(prev_lines[prev_idx])
    if not partial:
        return prev_page_text, next_page_text

    # première ligne utile de la page suivante (ignore en-têtes répétitifs de section)
    next_idx = None
    for i, line in enumerate(next_lines):
        s = line.strip()
        if not s:
            continue
        if re.match(r"^(Première|Deuxième|Troisième|Quatrième)\s+partie\s*[:\-–]", s):
            continue
        next_idx = i
        break

    if next_idx is None:
        return prev_page_text, next_page_text

    continuation = next_lines[next_idx].strip()
    if not _line_is_lk_continuation(continuation):
        return prev_page_text, next_page_text

    # recoller
    prev_lines[prev_idx] = prev_lines[prev_idx].rstrip() + " " + continuation
    next_lines[next_idx] = ""

    return "\n".join(prev_lines).strip(), "\n".join(next_lines).strip()


# ============================================================
# 10. EXTRACTION PAGE
# ============================================================

def extract_page_generic(page) -> str:
    raw_text = _extract_text_clean(page)
    parts = []

    if raw_text:
        parts.append(raw_text)

    return _dedupe_near_duplicate_lines("\n\n".join(p for p in parts if p.strip())).strip()


def _extract_leading_lk_continuation(tables: list, inherited_schema: dict | None) -> str | None:
    """
    Détecte si le premier tableau de la page commence par une ligne LK-seulement
    (désignation vide, coeff vide, LK présente) — continuation cross-page d'une LK
    incomplète. Retourne la valeur LK de continuation, ou None.
    """
    if not tables or inherited_schema is None:
        return None

    first_table = tables[0]
    if not first_table:
        return None

    lk_col = inherited_schema.get("lettre_cle", 2)
    c_col = inherited_schema.get("coefficient", 1)
    d_col = inherited_schema.get("designation", 0)

    row = first_table[0]
    if not row:
        return None

    needed = max(d_col, c_col, lk_col) + 1
    row_ext = list(row) + [""] * max(0, needed - len(row))

    desig = _cell(row_ext[d_col])
    coeff = _cell(row_ext[c_col])
    lk = _cell(row_ext[lk_col])

    # Cas 1 : désignation vide, pas de coeff, LK présente (ligne LK-only cross-page)
    if not desig and not _is_numeric_coeff(coeff) and _has_lettre_cle(lk):
        return _clean_lettre_cle(lk)

    # Cas 2 : désignation commence par une minuscule (suite de phrase), pas de coeff, LK présente
    # → c'est une ligne de continuation cross-page avec designation partielle ET LK de complétion
    if desig and desig[0].islower() and not _is_numeric_coeff(coeff) and _has_lettre_cle(lk):
        return _clean_lettre_cle(lk)

    return None


def extract_page_ngap(
    page,
    inherited_schema: dict | None = None
) -> tuple[str, dict | None, str | None]:
    """
    Retourne (texte_page, dernier_schema, lk_continuation_cross_page).
    lk_continuation_cross_page est la LK orpheline détectée en début de tableau
    qui doit être accolée à la dernière cotation de la page précédente.
    """
    # Récupérer les objets Table pour extraire les bboxes
    tables_objs = page.find_tables()

    if tables_objs:
        # Extraire le texte uniquement HORS des zones de tableaux (évite la duplication)
        def _not_in_any_table(obj):
            for t in tables_objs:
                bx0, by0, bx1, by1 = t.bbox
                if (
                    obj.get("x0", 0) >= bx0 - 2
                    and obj.get("top", 0) >= by0 - 2
                    and obj.get("x1", 0) <= bx1 + 2
                    and obj.get("bottom", 0) <= by1 + 2
                ):
                    return False
            return True

        non_table_page = page.filter(_not_in_any_table)
        raw_text = _extract_text_clean(non_table_page)
        tables = [t.extract() for t in tables_objs]
    else:
        raw_text = _extract_text_clean(page)
        tables = []

    # Détecter une LK orpheline de continuation cross-page
    leading_lk = _extract_leading_lk_continuation(tables, inherited_schema)

    parts = []
    if raw_text:
        parts.append(raw_text)

    formatted_parts = []
    last_schema = inherited_schema

    for table in tables:
        ttype = _classify_table(table, inherited_schema=last_schema)

        if ttype == "cotation_ngap":
            lines, detected_schema = _format_cotation_ngap(table, inherited_schema=last_schema)
            if lines:
                formatted_parts.append("\n".join(lines))
            last_schema = detected_schema

        elif ttype == "anatpath":
            lines = _format_anatpath(table)
            if lines:
                formatted_parts.append("\n".join(lines))

        elif ttype == "thermal":
            lines = _format_thermal(table)
            if lines:
                formatted_parts.append("\n".join(lines))

    if formatted_parts:
        parts.append("--- Cotations extraites ---")
        parts.extend(formatted_parts)

    final = "\n\n".join(p for p in parts if p.strip())
    final = _dedupe_near_duplicate_lines(final)
    return final.strip(), last_schema, leading_lk


def detect_pdf_mode(filepath: str, max_pages: int = 3) -> str:
    try:
        with pdfplumber.open(filepath) as pdf:
            sample_parts = []
            for page in pdf.pages[:max_pages]:
                sample_parts.append(_extract_text_clean(page))
            sample_text = "\n".join(sample_parts)
            return "ngap" if _is_ngap_document(sample_text) else "generic"
    except Exception:
        return "generic"


def load_pdf(filepath: str, mode: str | None = None) -> tuple[str, str]:
    chosen_mode = mode or detect_pdf_mode(filepath)

    with pdfplumber.open(filepath) as pdf:
        pages = []

        if chosen_mode == "generic":
            for page in pdf.pages:
                content = extract_page_generic(page)
                if content and content.strip():
                    pages.append(content)
            return "\n\n".join(pages).strip(), chosen_mode

        last_schema = None
        previous_page_content = None

        for page in pdf.pages:
            content, detected_schema, leading_lk = extract_page_ngap(
                page,
                inherited_schema=last_schema
            )

            if detected_schema:
                last_schema = detected_schema

            # Appliquer la LK orpheline de continuation à la dernière ligne de la page précédente
            if leading_lk and previous_page_content:
                prev_lines = previous_page_content.splitlines()
                for idx in range(len(prev_lines) - 1, -1, -1):
                    if "lettre clé" in prev_lines[idx].lower():
                        prev_lines[idx] = prev_lines[idx].rstrip() + " " + leading_lk
                        break
                previous_page_content = "\n".join(prev_lines)

            if not content or not content.strip():
                continue

            if previous_page_content is None:
                previous_page_content = content
                continue

            # >>> GESTION COUPURE ENTRE PAGES ICI <<<
            previous_page_content, content = _merge_page_boundary_lk(
                previous_page_content,
                content
            )

            pages.append(previous_page_content)
            previous_page_content = content

        if previous_page_content and previous_page_content.strip():
            pages.append(previous_page_content)

        return "\n\n".join(pages).strip(), chosen_mode


# ============================================================
# 11. MÉTADONNÉES
# ============================================================

def extract_pdf_name(filename: str) -> str | None:
    match = re.match(r"(pdf_\d+)", filename)
    return match.group(1) if match else None


def build_metadata(
    filepath: str,
    registry_index: dict[str, dict] | None = None,
    extraction_mode: str = "generic"
) -> DocumentMetadata:
    stat = os.stat(filepath)
    meta = DocumentMetadata(
        source=os.path.basename(filepath),
        filepath=os.path.abspath(filepath),
        extension=os.path.splitext(filepath)[1],
        size_bytes=stat.st_size,
        modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        extraction_mode=extraction_mode,
    )

    if registry_index:
        pdf_name = extract_pdf_name(meta.source)
        if pdf_name and pdf_name in registry_index:
            entry = registry_index[pdf_name]
            meta.registry_id = entry["id"]
            meta.type_document = entry["type_document"]
            meta.titre = entry["titre"]
            meta.annee = entry["annee"]
            meta.source_juridique = entry["source"]
            meta.url = entry["url"]
            meta.nom_pdf = entry["nom_pdf"]
            meta.theme = entry["theme"]
            meta.priorite = entry["priorite"]
            meta.commentaire = entry.get("commentaire", "")

    return meta


# ============================================================
# 12. INGESTION
# ============================================================

def ingest_file(
    filepath: str,
    registry_index: dict[str, dict] | None = None,
    force_mode: str | None = None
) -> Document:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".pdf":
        raise ValueError(f"Format '{ext}' non supporté. Seul .pdf est géré.")

    text, chosen_mode = load_pdf(filepath, mode=force_mode)
    text = clean_text(text)
    metadata = build_metadata(filepath, registry_index, extraction_mode=chosen_mode)

    return Document(text=text, metadata=metadata)


def ingest_directory(
    directory: str,
    registry_path: str | None = None,
    force_mode: str | None = None
) -> list[Document]:
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Pas un répertoire : {directory}")

    registry_index = None
    if registry_path:
        registry = load_registry(registry_path)
        registry_index = build_registry_index(registry)
        print(f"📋 Registre chargé : {len(registry)} entrées")

    documents = []
    skipped = []

    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath) or filename.startswith("."):
            continue

        if os.path.splitext(filename)[1].lower() == ".pdf":
            try:
                doc = ingest_file(filepath, registry_index, force_mode=force_mode)
                documents.append(doc)

                if doc.metadata.is_enriched:
                    m = doc.metadata
                    print(
                        f"  🔗 {filename} ({len(doc)} car.) "
                        f"→ {m.theme}, P{m.priorite} ({m.priorite_label}) "
                        f"[{m.extraction_mode}]"
                    )
                else:
                    print(f"  📄 {filename} ({len(doc)} car.) [{doc.metadata.extraction_mode}]")

            except Exception as e:
                print(f"  ❌ {filename} — {e}")
                skipped.append(filename)
        else:
            skipped.append(filename)

    enriched = sum(1 for d in documents if d.metadata.is_enriched)
    print(
        f"\n📊 {len(documents)} PDF chargés, "
        f"{enriched} enrichis via registre, "
        f"{len(skipped)} ignorés"
    )

    return documents


# ============================================================
# 13. EXPORT
# ============================================================

def save_document_text(doc: Document, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.text)


# ============================================================
# 14. TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TEST ÉTAPE 1 : INGESTION v5")
    print("=" * 60)

    DATA_DIR = "data/RAW"
    REGISTRY = "data/documents_registry.xlsx"

    if os.path.isdir(DATA_DIR):
        reg = REGISTRY if os.path.exists(REGISTRY) else None
        docs = ingest_directory(DATA_DIR, registry_path=reg)

        print("\n--- Détail des documents ---\n")
        for i, doc in enumerate(docs):
            m = doc.metadata
            print(f"[{i}] {doc}")
            if m.is_enriched:
                print(f"     titre     : {m.titre[:70]}...")
                print(f"     type      : {m.type_document} | thème : {m.theme}")
                print(f"     priorité  : {m.priorite} ({m.priorite_label})")
            print(f"     mode      : {m.extraction_mode}")
            print()

    else:
        print(f"\n⚠️  Dossier '{DATA_DIR}' introuvable.")
        print("   Test direct sur pdf_4.pdf...")

        test_path = "/mnt/user-data/uploads/pdf_4.pdf"
        if os.path.exists(test_path):
            doc = ingest_file(test_path)
            print(f"\n{doc}")
            print(f"Taille : {len(doc)} caractères")
            print(f"Mode   : {doc.metadata.extraction_mode}")

            out = "ngap_extraction_v5.txt"
            save_document_text(doc, out)
            print(f"✅ Export texte : {out}")