"""

TANHA Ablation: No Lexical Matching

Removes the lexical rescue stage from TANHA.

Entity matching relies on exact canonical matches

and semantic/structured similarity.

"""
import re
import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
def canonicalize_provision(text):

    text_lower = text.lower()

    sections = re.findall(r'\d+[a-zA-Z]*', text_lower)
    sections = sorted(set(sections))

    if not sections:
        return None

    if "article" in text_lower:
        return "article_" + "_".join(sections)

    return "section_" + "_".join(sections)

def normalize_legal_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("sec.", "section")
    text = text.replace("u/s", "section")
    text = re.sub(r'\s*\(\s*', '(', text)
    text = re.sub(r'\s*\)\s*', ')', text)
    text = re.sub(r'[.,;:]+$', '', text)
    return text



def canonicalize_court(text):
    text = text.lower().strip()

    # remove punctuation
    text = re.sub(r'[.,;:]', '', text)

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # aliases
    text = text.replace("apex court", "supreme court")
    text = text.replace("supreme court of india", "supreme court")

    # normalize HC abbreviation
    text = re.sub(r'\bhc\b', 'high court', text)

    # remove leading "the"
    text = re.sub(r'^the ', '', text)

    # high court of X → X high court
    m = re.match(r'high court of (.+)', text)
    if m:
        place = m.group(1).strip()
        text = f"{place} high court"

    # high court at X → X high court
    m = re.match(r'high court at (.+)', text)
    if m:
        place = m.group(1).strip()
        text = f"{place} high court"

    # X bench of high court → X high court
    m = re.match(r'(.+) bench of high court', text)
    if m:
        place = m.group(1).strip()
        text = f"{place} high court"

    # remove duplicate spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def canonicalize_judge(text):
    text = text.lower()

    # remove titles
    text = re.sub(r'\b(justice|honble|dr|mr|mrs|ms|shri|smt)\b\.?', '', text)

    # remove punctuation
    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def canonicalize_statute(text):

    if not text:
        return ""

    text = text.lower().strip()

    text = text.replace("&", "and")

    ABBREV_MAP = {
        "ipc": "indian penal code",
        "crpc": "code of criminal procedure",
        "cpc": "code of civil procedure",
        "evidence act": "indian evidence act",
    }

    for k, v in ABBREV_MAP.items():
        if text == k or text.startswith(k + " "):
            text = text.replace(k, v)

    text = text.replace("anti social", "anti-social")

    # KEEP YEAR
    text = re.sub(r'[^\w\s-]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def token_overlap(a, b):
    a_set = set(a.split())
    b_set = set(b.split())

    inter = len(a_set & b_set)
    union = len(a_set | b_set)

    if union == 0:
        return 0.0

    return inter / union

def canonicalize_precedent_v2(text: str) -> str:

    if not text:
        return ""

    text = text.lower().strip()

    # -------------------------------------------------
    # 1️⃣ Normalize v / vs / versus
    # -------------------------------------------------
    text = re.sub(r'\bversus\b', 'v', text)
    text = re.sub(r'\bvs\.?\b', 'v', text)
    text = re.sub(r'\bv\.?\b', 'v', text)

    # -------------------------------------------------
    # 2️⃣ Remove citations inside [] or ()
    # -------------------------------------------------
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)

    # -------------------------------------------------
    # 3️⃣ Remove common citation patterns
    # -------------------------------------------------
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\b\d+\s*scc\b.*', '', text)
    text = re.sub(r'air\s*\d+.*', '', text)
    text = re.sub(r'\b\d+\s*scr\b.*', '', text)

    # -------------------------------------------------
    # 4️⃣ Remove procedural tails
    # -------------------------------------------------
    procedural_patterns = [
        r'criminal appeal.*',
        r'civil appeal.*',
        r'special leave.*',
        r'letters patent appeal.*',
        r'criminal misc.*',
        r'writ petition.*',
        r'suo moto.*',
        r'review petition.*',
        r'reported in.*'
    ]

    for pattern in procedural_patterns:
        text = re.sub(pattern, '', text)

    # -------------------------------------------------
    # 5️⃣ Remove titles (dr, mr, justice, etc.)
    # -------------------------------------------------
    text = re.sub(r'\b(dr|mr|mrs|ms|justice|honble|shri|smt)\b\.?', '', text)

    # -------------------------------------------------
    # 6️⃣ Normalize "others" / "ors" / "& others"
    # -------------------------------------------------
    text = re.sub(r'\b&?\s*others\b', '', text)
    text = re.sub(r'\bors\.?\b', '', text)

    # -------------------------------------------------
    # 7️⃣ Remove extra descriptors before "v"
    #    Keep only text around first " v "
    # -------------------------------------------------
    if " v " in text:
        parts = text.split(" v ", 1)
        left = parts[0].strip()
        right = parts[1].strip()

        # Remove trailing noise after right party
        right = re.split(r'\b(and|,)\b', right)[0].strip()

        text = f"{left} v {right}"

    # -------------------------------------------------
    # 8️⃣ Remove non-alphanumeric except spaces
    # -------------------------------------------------
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # -------------------------------------------------
    # 9️⃣ Normalize whitespace
    # -------------------------------------------------
    text = re.sub(r'\s+', ' ', text).strip()

    return text

TAG_PATTERN = r'<(.*?)>(.*?)</\1>'

device = "cpu"
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)

def embed_entity_list(entity_list):
    if len(entity_list) == 0:
        return torch.empty((0, model.get_sentence_embedding_dimension()))

    texts = [e[0] for e in entity_list]

    emb = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return emb.cpu()


PERSON_TYPES = {"PETITIONER", "RESPONDENT", "OTHER_PERSON"}

# TYPE_THRESHOLDS = {
#     "STATUTE":   {"semantic": 0.80, "fuzzy": 70},
#     "PRECEDENT": {"semantic": 0.75, "fuzzy": 65},
#     "PROVISION": {"semantic": 0.90, "fuzzy": 90},
#     "COURT":     {"semantic": 0.90, "fuzzy": 90},
#     "JUDGE":     {"semantic": 0.92, "fuzzy": 90},
#     "PETITIONER": {"semantic": 0.88, "fuzzy": 85},
#     "RESPONDENT": {"semantic": 0.88, "fuzzy": 85},
# }

def types_compatible(t1, t2):
    if t1 in PERSON_TYPES and t2 in PERSON_TYPES:
        return True
    return t1 == t2




def provision_similarity(p_text, t_text):
    return 1.0 if p_text == t_text else 0.0


# Lexical rescue stage removed for ablation study.
def build_similarity_matrix(pred_entities, target_entities, pred_emb, target_emb):

    n_pred = len(pred_entities)
    n_target = len(target_entities)

    S = np.zeros((n_pred, n_target))

    if n_pred == 0 or n_target == 0:
        return S

    semantic_matrix = (pred_emb @ target_emb.T).cpu().numpy()

    for i, (p_text, p_type) in enumerate(pred_entities):
        for j, (t_text, t_type) in enumerate(target_entities):

            if not types_compatible(p_type, t_type):
                continue

            sem = semantic_matrix[i, j]

            # -----------------------------
            # PROVISION
            # -----------------------------
            if p_type == "PROVISION":
                score = provision_similarity(p_text, t_text)

            # -----------------------------
            # STATUTE (FIXED)
            # -----------------------------
            elif p_type == "STATUTE":

                p_clean = canonicalize_statute(p_text)
                t_clean = canonicalize_statute(t_text)

                lex = lexical_similarity(p_clean, t_clean)
                overlap = token_overlap(p_clean, t_clean)

                score = max(
                    0.6 * sem + 0.4 * lex,
                    overlap
                )

            # -----------------------------
            # PRECEDENT (FIXED)
            # -----------------------------
            elif p_type == "PRECEDENT":

                p_clean = canonicalize_precedent_v2(p_text)

                t_clean = canonicalize_precedent_v2(t_text)

                if p_clean == t_clean:

                    score = 1.0

                else:

                    lex = lexical_similarity(p_clean, t_clean)

                    if lex < 0.90:

                        score = 0.0

                    else:

                        score = min(sem, lex)


            elif p_type in {"PETITIONER","RESPONDENT","OTHER_PERSON"}:

                lex = lexical_similarity(p_text, t_text)

                if lex >= 0.90:

                    score = lex

                else:

                    score = 0.0

            # -----------------------------
            # COURT
            # -----------------------------
            elif p_type == "COURT":

                p_clean = canonicalize_court(p_text)

                t_clean = canonicalize_court(t_text)

                score = 1.0 if p_clean == t_clean else 0.0

            # -----------------------------
            # JUDGE
            # -----------------------------
            elif p_type == "JUDGE":

                p_clean = canonicalize_judge(p_text)
                t_clean = canonicalize_judge(t_text)

                if p_clean == t_clean:
                    score = 1.0
                else:
                    score = 0.0

            else:
                score = sem

            S[i, j] = score

    return S

TYPE_THRESHOLDS = {
    "PROVISION": 1.00,
    "STATUTE": 0.90,
    "PRECEDENT": 0.90,
    "COURT": 1.00,
    "JUDGE": 1.00,
    "PETITIONER": 0.90,
    "RESPONDENT": 0.90,
    "OTHER_PERSON": 0.90,
}

# Lexical rescue stage removed for ablation study.

def hybrid_match_v3(
    pred_entities,
    target_entities,
    pred_emb,
    target_emb,
):

    if len(pred_entities) == 0 or len(target_entities) == 0:
        return set(), set()

    matched_pred = set()
    matched_target = set()

    # -----------------------------
    # Stage 1 Exact Match
    # -----------------------------
    target_lookup = {
        t: idx
        for idx, t in enumerate(target_entities)
    }

    for i, p in enumerate(pred_entities):
        if p in target_lookup:
            j = target_lookup[p]

            matched_pred.add(i)
            matched_target.add(j)

    # -----------------------------
    # Stage 2 Semantic + Structured Matching
    # -----------------------------
    S = build_similarity_matrix(
        pred_entities,
        target_entities,
        pred_emb,
        target_emb
    )

    for i, (p_text, p_type) in enumerate(pred_entities):

        if i in matched_pred:
            continue

        best_score = 0
        best_j = None

        for j, (t_text, t_type) in enumerate(target_entities):

            if j in matched_target:
                continue

            if p_type != t_type:
                continue

            score = S[i, j]

            if score > best_score:
                best_score = score
                best_j = j

        if best_j is None:
            continue

        threshold = TYPE_THRESHOLDS.get(
            p_type,
            0.75
        )

        if best_score >= threshold:

            matched_pred.add(i)
            matched_target.add(best_j)

    return matched_pred, matched_target

def lexical_similarity(a, b):

    a = normalize_legal_text(a)
    b = normalize_legal_text(b)

    s1 = fuzz.token_sort_ratio(a, b)
    s2 = fuzz.partial_ratio(a, b)
    s3 = fuzz.token_set_ratio(a, b)

    return max(s1, s2, s3) / 100.0
