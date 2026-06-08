"""

TANHA Ablation: No Canonicalization

All entity-specific canonicalization rules are disabled.

Entity matching relies on normalized surface forms,

lexical similarity, and semantic similarity.

"""
import re
import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
def canonicalize_provision(text):
    return normalize_legal_text(text)

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
    return normalize_legal_text(text)

def canonicalize_judge(text):
    return normalize_legal_text(text)

def canonicalize_statute(text):
    return normalize_legal_text(text)


def token_overlap(a, b):
    a_set = set(a.split())
    b_set = set(b.split())

    inter = len(a_set & b_set)
    union = len(a_set | b_set)

    if union == 0:
        return 0.0

    return inter / union

def canonicalize_precedent_v2(text):
    return normalize_legal_text(text)



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


def types_compatible(t1, t2):
    if t1 in PERSON_TYPES and t2 in PERSON_TYPES:
        return True
    return t1 == t2




def provision_similarity(p_text, t_text):
    return 1.0 if p_text == t_text else 0.0



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


def hybrid_match_v3(pred_entities, target_entities, pred_emb, target_emb):

    if len(pred_entities) == 0 or len(target_entities) == 0:
        return set(), set()

    matched_pred = set()
    matched_target = set()

    # ----------------------------------
    # Stage 1: Exact canonical match
    # ----------------------------------
    target_lookup = {t: idx for idx, t in enumerate(target_entities)}

    for i, p in enumerate(pred_entities):
        if p in target_lookup:
            j = target_lookup[p]
            matched_pred.add(i)
            matched_target.add(j)

    # ----------------------------------
    # Stage 2: Lexical rescue
    # ----------------------------------
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

            score = lexical_similarity(p_text, t_text)

            if score > best_score:
                best_score = score
                best_j = j

        if best_j is None:
            continue

        if p_type in ["COURT","JUDGE","PETITIONER","RESPONDENT","OTHER_PERSON"]:
            threshold = 0.90
        else:
            threshold = 0.80

        # strict match
        if best_score >= threshold:
            matched_pred.add(i)
            matched_target.add(best_j)

    # ----------------------------------
    # Stage 3: Semantic + structured similarity
    # ----------------------------------
    if len(matched_pred) < len(pred_entities):

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

            threshold = TYPE_THRESHOLDS.get(p_type, 0.75)

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
