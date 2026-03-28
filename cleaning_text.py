"""
NETTOYAGE DU TEXTE EXTRAIT DES PDF
====================================
Pipeline de nettoyage pour textes juridiques (NGAP, JORF, circulaires Ameli).

CORRECTIONS v2 :
- FIX 1 : fix_hyphenation enrichi avec les mots composés médicaux/juridiques
           réels trouvés dans la NGAP (masseur-kinésithérapeute, chirurgien-dentiste, etc.)
- FIX 2 : remove_page_numbers protégé contre la suppression de coefficients isolés
           (ne supprime que si la ligne est UNIQUEMENT un nombre, pas dans un contexte de cotation)
- FIX 3 : remove_non_idel_sections sécurisé avec garde-fous de longueur maximale
           pour éviter qu'un regex DOTALL ne mange tout le texte si l'ancre change
- FIX 4 : remove_ingestion_markers supprime les séparateurs "--- Cotations extraites ---"
           ajoutés par l'ingestion hybride, inutiles pour le RAG

CORRECTIONS v3 :
- FIX 5 : strip_leading_whitespace supprime les espaces initiaux artefacts du layout PDF
           (79 % des lignes avaient des espaces de tête non significatifs)
- FIX 6 : remove_legislative_metadata supprime les lignes "(Modifié/Créé/Abrogé par décision
           UNCAM...)" pures métadonnées législatives, sans valeur pour le RAG
- FIX 7 : remove_noise_lines supprime les fragments parasites isolés : numéros de section
           seuls ("2.", "3."), conjonctions seules ("ou", "et"), marqueurs de liste seuls
- FIX 8 : normalize_inline_conditions reformate "| condition:" en bloc lisible par un LLM
"""

import re
import unicodedata


# ============================================================
# 1. EN-TÊTES / PIEDS DE PAGE RÉPÉTÉS
# ============================================================

JORF_HEADERS = [
    "JOURNAL OFFICIEL DE LA RÉPUBLIQUE FRANÇAISE",
    "Décrets, arrêtés, circulaires",
    "Avis et communications",
    "SOMMAIRE ANALYTIQUE",
    "textes généraux",
    "Sommaire",
    "mesures nominatives",
    "conventions collectives",
    "avis divers",
    "Informations parlementaires",
    "Assemblée nationale",
    "avis de concours et de vacance d'emplois",
    'République française.',
    'publié au Journal officiel de la République française.',
    'Journal officiel de la République française.',
    'Pour le ministre et par délégation :',
    'TEXTES GÉNÉRAUX',
]

CIRCULAIRE_HEADERS = {"Nouveau", "Modificatif", "Complémentaire", "Suivi"}


def remove_jorf_headers(text: str) -> str:
    """
    Supprime les en-têtes/pieds de page récurrents :
    - Journal Officiel (JORF)
    - NGAP (pieds de page "Version en vigueur", en-têtes de partie répétés)
    - Circulaires Ameli (cases à cocher)
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # En-têtes JORF exacts
        if stripped and any(stripped == header for header in JORF_HEADERS):
            continue

        # Pieds de page NGAP : "Version en vigueur du XX/XX/XXXX"
        if re.match(r"^Version\s+en\s+vigueur\s+du\s+\d{2}/\d{2}/\d{4}\s*$", stripped):
            continue

        # En-têtes de section NGAP répétés à chaque page (match souple)
        if re.match(r"^Premi.re\s+partie\s*:?\s*Dispositions\s+G.n.rales\s*$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Deuxi.me\s+partie\s*:?.*radiations\s+ionisantes\s*$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Cinqui.me\s+partie\s*:?.*biologie\s+m.dicale\s*$", stripped, re.IGNORECASE):
            continue
        if stripped == "Annexes":
            continue

        # En-têtes circulaires (cases à cocher)
        if stripped in CIRCULAIRE_HEADERS:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Patterns JORF inline
    JORF_INLINE_PATTERNS = [
        r"\d{1,2}\s+\w+\s+\d{4}\s+Texte\s+\d+\s+sur\s+\d+",
        r"\d{1,2}\s+\w+\s+\d{4}\s*\n\s*Texte\s+\d+\s+sur\s+\d+",
        r"^Texte\s+\d+\s+sur\s+\d+\s*$",
    ]
    for pattern in JORF_INLINE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    return text


# ============================================================
# 2. NUMÉROS DE PAGE ISOLÉS (FIX 2)
# ============================================================

def remove_page_numbers(text: str) -> str:
    """
    Supprime les numéros de page isolés.
    
    FIX v2 : Vérifie le CONTEXTE autour de la ligne pour ne pas supprimer
    un coefficient isolé qui serait sur sa propre ligne dans un tableau mal parsé.
    On ne supprime que si les lignes adjacentes ne sont pas des cotations.
    """
    lines = text.split("\n")
    cleaned = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Formats classiques de numéros de page
        if re.match(r"^[-–]\s*\d+\s*[-–]$", stripped):
            continue
        if re.match(r"^page\s+\d+", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\d+\s*/\s*\d+$", stripped):
            continue
        
        # Nombre isolé : vérifier le contexte
        if re.match(r"^\d{1,4}$", stripped):
            # Regarder les lignes adjacentes pour détecter un contexte de cotation
            prev_line = lines[i - 1].strip() if i > 0 else ""
            next_line = lines[i + 1].strip() if i < len(lines) - 1 else ""
            
            # Si la ligne précédente ou suivante contient des indicateurs de cotation,
            # on conserve le nombre (c'est peut-être un coefficient)
            cotation_indicators = ["AMI", "AMX", "SFI", "AIS", "AMK", "coefficient", 
                                   "lettre clé", "Désignation", "séance", "forfait"]
            context = prev_line + " " + next_line
            if any(ind in context for ind in cotation_indicators):
                cleaned.append(line)  # Garder : contexte de cotation
                continue
            
            # Sinon c'est probablement un numéro de page
            continue
        
        cleaned.append(line)
    
    return "\n".join(cleaned)


# ============================================================
# 3. LIGNES DE SÉPARATION
# ============================================================

def remove_separator_lines(text: str) -> str:
    """
    Supprime les lignes composées uniquement de caractères de séparation.
    Ne touche PAS aux lignes mixtes comme "--- Cotations extraites ---".
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) >= 3 and all(c in "-=_*~" for c in stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ============================================================
# 4. UNICODE
# ============================================================

UNICODE_REPLACEMENTS = {
    "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2212": "-", "\u2015": "—",
    "\xad": "", "\xa0": " ", "\u2009": " ",
    "\uf0fc": "", "\uf0a7": "", "\uf0d8": "",
    "\uf0b7": "", "\uf075": "", "\uf0cc": "",
    "\uf0b6": "",
    "\uf0a0": "",
}


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for old_char, new_char in UNICODE_REPLACEMENTS.items():
        text = text.replace(old_char, new_char)
    return text


# ============================================================
# 5. CÉSURES (FIX 1)
# ============================================================

# Préfixes de mots composés français courants
MOTS_COMPOSES_PREFIXES = {
    "vice", "sous", "demi", "semi", "anti", "auto", "co", "contre",
    "entre", "extra", "hyper", "infra", "inter", "intra", "macro",
    "méta", "micro", "mini", "multi", "néo", "non", "outre", "para",
    "poly", "post", "pré", "pro", "pseudo", "quasi", "re", "ré",
    "semi", "socio", "sub", "super", "supra", "sur", "trans", "tri",
    "ultra", "médico", "chirurgico", "technico",
}

# FIX v2 : Mots composés COMPLETS fréquents dans les textes NGAP/médicaux
# qui doivent TOUJOURS garder leur trait d'union.
# Construits à partir de l'analyse des 61 césures réelles du PDF NGAP.
MOTS_COMPOSES_CONNUS = {
    # Professions de santé
    "masseur-kinésithérapeute", "chirurgien-dentiste", "sage-femme",
    "anesthésiste-réanimateur", "pédicure-podologue",
    # Termes médicaux composés
    "maxillo-faciale", "psycho-comportemental", "neuro-développement",
    "neuro-psychologique", "neuro-ophtalmologique", "neuro-visuelles",
    "oesophago-gastrique", "médico-social", "médico-sociales",
    "médico-sociaux", "ostéo-articulaire", "ostéo-articulaires",
    "cardio-vasculaire", "cardio-vasculaires", "cardio-circulatoire",
    "bucco-dentaire", "bucco-dentaires", "bucco-linguales",
    "dento-faciale", "dento-faciales", "temporo-maxillaire",
    "tibio-tarsienne", "tibio-tarsiennes", "périnéo-sphinctériennes",
    "hygiéno-diététiques", "cognitivo-linguistiques",
    "musculo-articulaire", "musculo-articulaires",
    "vélo-tubo-tympanique", "vélo-tubo-tympaniques",
    "oro-myo-faciales", "oro-oesophagiennes",
    "trachéo-oesophagienne",
    # Anatomie
    "avant-bras", "avant-pied", "avant-pieds",
    "lombo-sacré", "pyélo-urétérale",
    # Juridique / administratif
    "ci-dessus", "ci-dessous", "ci-après", "ci-contre",
    "celui-ci", "celle-ci", "ceux-ci",
    "peut-être", "c'est-à-dire",
    # Matériel médical
    "lève-malade", "nutri-pompe",
    # Termes NGAP spécifiques
    "comptes-rendus", "compte-rendu",
}

# Seconds mots qui indiquent un vrai composé quand précédés d'un tiret
COMPOUND_SECOND_PARTS = {
    "ci", "dessus", "dessous", "après", "contre", "même",
    "femme", "dentiste", "kinésithérapeute", "réanimateur", "podologue",
    "faciale", "social", "sociale", "sociaux", "dentaire", "dentaires",
    "vasculaire", "vasculaires", "articulaire", "articulaires",
    "bras", "pied", "pieds", "malade",
}

COMPOUND_PATTERNS = [r"^\d+", r"^[A-Z]{2,}"]


def is_real_hyphenated_word(before: str, after: str) -> bool:
    """
    Détermine si 'before-after' est un vrai mot composé (garder le tiret)
    ou une césure de fin de ligne (joindre les deux parties).
    
    FIX v2 : Enrichi avec les mots composés médicaux/NGAP.
    """
    before_lower = before.lower()
    after_lower = after.lower()
    full_word = f"{before_lower}-{after_lower}"
    
    # 1. Mot composé connu exactement
    if full_word in MOTS_COMPOSES_CONNUS:
        return True
    # Aussi vérifier les variantes avec majuscules
    if f"{before}-{after}".lower() in {w.lower() for w in MOTS_COMPOSES_CONNUS}:
        return True
    
    # 2. Préfixe composé reconnu
    if before_lower in MOTS_COMPOSES_PREFIXES:
        return True
    
    # 3. Seconde partie caractéristique d'un composé
    if after_lower in COMPOUND_SECOND_PARTS:
        return True
    
    # 4. Patterns structurels (nombre-X, SIGLE-X)
    for pattern in COMPOUND_PATTERNS:
        if re.match(pattern, before):
            return True
    
    # 5. Nombres composés
    nombres = {"dix", "vingt", "trente", "quarante", "cinquante",
               "soixante", "quatre", "cent"}
    if before_lower in nombres:
        return True
    
    # 6. Noms propres composés (villes thermales type "Saint-Lary", "Bagnoles-de")
    if before[0].isupper() and after[0].isupper():
        return True
    
    return False


def fix_hyphenation(text: str) -> str:
    """Rejoint les césures de fin de ligne, en préservant les vrais mots composés."""
    def replace_hyphen(match):
        before = match.group(1)
        after = match.group(2)
        if is_real_hyphenated_word(before, after):
            return f"{before}-{after}"
        else:
            return f"{before}{after}"
    return re.sub(r"(\w+)-\s*\n\s*(\w+)", replace_hyphen, text)


# ============================================================
# 6. ESPACES ET SAUTS DE LIGNE
# ============================================================

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


# ============================================================
# 7. LIGNES D'UN SEUL CARACTÈRE
# ============================================================

def remove_single_char_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 1 and not stripped.isalnum():
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ============================================================
# 8. SUPPRESSION DES ARTICLES OBSOLÈTES
# ============================================================

def remove_obsolete_articles(text: str) -> str:
    """
    Supprime l'article 11 du Chapitre I, Titre XVI (soins infirmiers en AIS)
    → Remplacé par l'article 12 + forfaits BSI.
    
    NE TOUCHE PAS à l'article 11B des Dispositions Générales (cumul).
    NE TOUCHE PAS à l'article 11 du Titre XIV (soins palliatifs kiné).
    """
    pattern = (
        r"Article\s+11\s*[-–—]\s*Soins\s+infirmiers\s+.{0,5}domicile\s+pour\s+un\s+patient,\s+quel\s+que\s+soit\s+son\s+.ge.*?"
        r"(?=Article\s+12\s*[-–—])"
    )
    text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text


# ============================================================
# 9. SUPPRESSION DES SECTIONS NON PERTINENTES (FIX 3)
# ============================================================

# Longueurs maximales attendues pour chaque section à supprimer.
# Si le match dépasse ce seuil, on ne supprime PAS (sécurité anti-débordement).
_MAX_SECTION_LENGTHS = {
    "stations_thermales": 25_000,     # ~12K actuellement
    "references_reglementaires": 10_000,  # ~3K actuellement
    "tableau_bsi": 10_000,            # ~4K actuellement
}


def _safe_dotall_sub(pattern: str, text: str, section_name: str) -> str:
    """
    Applique un re.sub DOTALL avec garde-fou de longueur.
    Si le match est anormalement long, il est ignoré (log un warning).
    
    FIX v2 : Protège contre les regex DOTALL qui mangent trop de texte
    quand l'ancre de fin ne matche pas.
    """
    max_len = _MAX_SECTION_LENGTHS.get(section_name, 50_000)
    
    def safe_replace(match):
        if len(match.group()) > max_len:
            # Match trop gros → probablement un problème d'ancre
            # On ne supprime rien par sécurité
            return match.group()
        return ""
    
    return re.sub(pattern, safe_replace, text, flags=re.DOTALL)


def remove_non_idel_sections(text: str) -> str:
    """
    Supprime les sections de la NGAP non pertinentes pour les IDEL :
    - Liste des stations thermales (longue liste de villes)
    - Table des références réglementaires (dates JO)
    - Tableaux BSI détaillés (combinaisons HEM)
    
    FIX v2 : Chaque suppression est protégée par un garde-fou de longueur.
    """
    # Stations thermales (longue liste)
    text = _safe_dotall_sub(
        r"Stations\s+thermales\s+Orientations\s+th.rapeutiques.*?(?=CHAPITRE\s+V|TITRE\s+XVI|$)",
        text,
        "stations_thermales"
    )
    
    # Références réglementaires
    text = _safe_dotall_sub(
        r"REFERENCES\s+DES\s+TEXTES\s+REGLEMENTAIRES.*?(?=ANNEXE\s+1|ANNEXE\s+2|$)",
        text,
        "references_reglementaires"
    )
    
    # Tableaux BSI combinaisons HEM
    text = _safe_dotall_sub(
        r"Tableau\s+2\s*:\s*R.gles\s+de\s+classement\s+des\s+forfaits\s+BSI.*?(?=ANNEXE\s+2|$)",
        text,
        "tableau_bsi"
    )
    
    return text


# ============================================================
# 10. MARQUEURS D'INGESTION (FIX 4)
# ============================================================

def remove_ingestion_markers(text: str) -> str:
    """
    Supprime les marqueurs ajoutés par l'ingestion hybride
    (séparateurs "--- Cotations extraites ---") qui sont inutiles pour le RAG
    et ajoutent du bruit dans les chunks.
    
    FIX v2 : Nouveau nettoyage spécifique.
    """
    text = re.sub(r"^---\s*Cotations\s+extraites\s*---\s*$", "", text, flags=re.MULTILINE)
    return text


# ============================================================
# 11. ESPACES INITIAUX (FIX 5)
# ============================================================

def strip_leading_whitespace(text: str) -> str:
    """
    Supprime les espaces/tabulations en début de ligne.
    Artefacts du layout PDF : 79 % des lignes extraites ont des espaces initiaux
    non significatifs pour le sens du texte.
    Les lignes de cotation comme "  - Désignation : ..." deviennent "- Désignation : ...".
    """
    lines = text.split("\n")
    return "\n".join(line.lstrip(" \t") for line in lines)


# ============================================================
# 12. MÉTADONNÉES LÉGISLATIVES (FIX 6)
# ============================================================

# Ligne démarrant une métadonnée législative (ouverture de parenthèse)
_RE_LEGISLATIVE_START = re.compile(
    r"^\s*\((Modifi[eé]|Cr[eé]{1,2}|Abrog[eé])\b",
    re.IGNORECASE,
)


def remove_legislative_metadata(text: str) -> str:
    """
    Supprime les lignes pures de métadonnées législatives du type :
      (Modifié par décision UNCAM du 09/10/2023 - JO du 28/11/2023)
      (Créé par décision UNCAM du 01/01/2022)
      (Abrogé par arrêté du 15/06/2021)
    Gère aussi les entrées multi-lignes (parenthèse non fermée sur la première ligne) :
    les lignes de continuation sont supprimées jusqu'à la ligne qui ferme la parenthèse.
    Ces lignes sont du bruit pour le RAG.
    """
    lines = text.split("\n")
    cleaned = []
    in_meta_block = False
    for line in lines:
        if in_meta_block:
            # On est dans un bloc multi-ligne : supprimer jusqu'à la fermeture
            if ")" in line:
                in_meta_block = False
            continue
        if _RE_LEGISLATIVE_START.match(line):
            # Ligne de métadonnée : vérifier si elle est fermée sur cette ligne
            if ")" not in line:
                in_meta_block = True  # Multi-ligne : activer le mode bloc
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ============================================================
# 13. FRAGMENTS PARASITES ISOLÉS (FIX 7)
# ============================================================

# Fragments trop courts pour être utiles seuls, sur une ligne vide de contexte
_NOISE_LINES = re.compile(
    r"^("
    r"\d{1,2}\."           # "2." "12." (numéros de section orphelins)
    r"|[a-z°]{1,3}\)"      # "a)" "b)" "1°)" isolés
    r"|ou|et|ou\s+:"       # conjonctions seules
    r"|[-–•]\s*"           # puces vides
    r"|[ivxlIVXL]{1,5}\."  # numéros romains isolés (i. ii. iii.)
    r")$",
    re.IGNORECASE,
)


def remove_noise_lines(text: str) -> str:
    """
    Supprime les fragments parasites isolés sur leur propre ligne :
    numéros de section orphelins ("2."), conjonctions seules ("ou", "et"),
    puces vides, numéros romains seuls.
    Condition : la ligne doit être entourée de lignes vides (fragment vraiment isolé).
    """
    lines = text.split("\n")
    cleaned = []
    n = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _NOISE_LINES.match(stripped):
            # Vérifier que c'est bien isolé (lignes adjacentes vides)
            prev_empty = (i == 0) or (not lines[i - 1].strip())
            next_empty = (i == n - 1) or (not lines[i + 1].strip())
            if prev_empty and next_empty:
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ============================================================
# 14. CONDITIONS INLINE (FIX 8)
# ============================================================

def normalize_inline_conditions(text: str) -> str:
    """
    Reformate les conditions inline du type "| condition:" issues de l'extraction
    de tables NGAP en un bloc lisible par un LLM :
      AVANT : "- Désignation : texte | condition: valeur"
      APRÈS  : "- Désignation : texte\\n  Condition : valeur"
    Améliore la lisibilité pour le chunking RAG en séparant la valeur de sa condition.
    """
    # Pipe suivi de "condition:", "conditions:", "remarque:", "note:" → saut de ligne
    text = re.sub(
        r"\s*\|\s*(condition|conditions|remarque|note)\s*:",
        r"\n  \1 :",
        text,
        flags=re.IGNORECASE,
    )
    # Capitalise le mot-clé après le saut
    def _cap(m):
        return "\n  " + m.group(1).capitalize() + " :"
    text = re.sub(r"\n  (condition|conditions|remarque|note) :", _cap, text, flags=re.IGNORECASE)
    return text


# ============================================================
# 15. PIPELINE PRINCIPAL
# ============================================================

def clean_text(text: str) -> str:
    """
    Pipeline de nettoyage complet :
    1.  Unicode
    2.  En-têtes JORF + NGAP + Circulaires
    3.  Numéros de page (avec protection contexte cotation)
    4.  Lignes de séparation
    5.  Lignes d'un seul caractère
    6.  Espaces initiaux (artefacts layout PDF) [v3]
    7.  Métadonnées législatives UNCAM [v3]
    8.  Fragments parasites isolés [v3]
    9.  Conditions inline → bloc lisible LLM [v3]
    10. Césures (avec mots composés médicaux)
    11. Articles obsolètes (Article 11 Ch.I T.XVI)
    12. Sections non pertinentes (stations thermales, tables BSI)
    13. Marqueurs d'ingestion
    14. Espaces (nettoyage final)
    """
    text = normalize_unicode(text)
    text = remove_jorf_headers(text)
    text = remove_page_numbers(text)
    text = remove_separator_lines(text)
    text = remove_single_char_lines(text)
    text = strip_leading_whitespace(text)
    text = remove_legislative_metadata(text)
    text = remove_noise_lines(text)
    text = normalize_inline_conditions(text)
    text = fix_hyphenation(text)
    text = remove_obsolete_articles(text)
    text = remove_non_idel_sections(text)
    text = remove_ingestion_markers(text)
    text = normalize_whitespace(text)
    return text