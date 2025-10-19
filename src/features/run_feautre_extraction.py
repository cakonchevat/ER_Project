from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import jellyfish
from rapidfuzz.fuzz import ratio as rf_ratio
from src.utils.common_methods import tokenize


# Calculates the jaccard similarity between 2 rows: |A∩B| / |A∪B|
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


# Calculates longest common substring
def lcs_len(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = prev + 1 if ai == b[j - 1] else max(dp[j], dp[j - 1])
            prev = cur
    return dp[lb]


def lcs_ratio_from_norm(a_n: str, b_n: str) -> float:
    den = max(len(a_n), len(b_n))
    return (lcs_len(a_n, b_n) / den) if den else 1.0


# String metrics:
def edit_ratio_from_norm(a_n: str, b_n: str) -> float:
    return float(rf_ratio(a_n, b_n)) / 100.0


def jaro_winkler_from_norm(a_n: str, b_n: str) -> float:
    return float(jellyfish.jaro_winkler_similarity(a_n, b_n))


# Fonetic similarity
def dmetaphone_match_first_token(a_tokens: List[str], b_tokens: List[str]) -> int:
    if not a_tokens or not b_tokens:
        return 0
    return int(jellyfish.metaphone(a_tokens[0]) == jellyfish.metaphone(b_tokens[0]))


# Cosine similarity metrics:
def rowwise_cosine(A, B) -> np.ndarray:
    # safe row-wise cosine for sparse or dense matrices (paired rows)
    if hasattr(A, "multiply"):
        num = A.multiply(B).sum(axis=1).A1
        xnorm = np.sqrt(A.multiply(A).sum(axis=1)).A1
        ynorm = np.sqrt(B.multiply(B).sum(axis=1)).A1
    else:
        num = np.einsum("ij,ij->i", A, B)
        xnorm = np.linalg.norm(A, axis=1)
        ynorm = np.linalg.norm(B, axis=1)
    den = xnorm * ynorm
    den[den == 0.0] = 1.0
    return (num / den).astype(float)



def pair_token_cosine(src_tokens: List[List[str]], cand_tokens: List[List[str]]) -> np.ndarray:
    texts_union = [" ".join(t) for t in (src_tokens + cand_tokens)]
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
    M = cv.fit_transform(texts_union)
    n = len(src_tokens)
    return rowwise_cosine(M[:n], M[n:])


def pair_tfidf_cosine(src_texts: List[str], cand_texts: List[str], *,
                      analyzer: str, ngram_range: Tuple[int, int]) -> np.ndarray:
    corpus = src_texts + cand_texts
    tf = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        strip_accents="unicode",
        sublinear_tf=True,
        lowercase=False,  # already normalized
    )
    X = tf.fit_transform(corpus)
    n = len(src_texts)
    return rowwise_cosine(X[:n], X[n:])


# Building features:
SELECTED_FEATURES = [
    "edit_ratio",
    "jaro_winkler",
    "lcs_ratio",
    "token_jaccard",
    "token_cosine",
    "tfidf_word_cosine",
    "tfidf_char_cosine",
    "dmetaphone_match",
]


def build_matching_features(pairs: pd.DataFrame, src_col: str = "src_text",
                            cand_col: str = "cand_text") -> pd.DataFrame:

    if not {src_col, cand_col}.issubset(pairs.columns):
        missing = {src_col, cand_col} - set(pairs.columns)
        raise ValueError(f"Missing required text columns: {missing}")

    df = pairs.copy()

    # Prepare text and tokens
    src_raw = df[src_col].fillna("").astype(str).tolist()
    cand_raw = df[cand_col].fillna("").astype(str).tolist()

    src_tokens = [tokenize(s) for s in src_raw]
    cand_tokens = [tokenize(s) for s in cand_raw]
    src_norm = [" ".join(t) for t in src_tokens]
    cand_norm = [" ".join(t) for t in cand_tokens]

    # Tokens for Jaccard
    src_sets = [set(t) for t in src_tokens]
    cand_sets = [set(t) for t in cand_tokens]

    # Vector-based similarities
    token_cos = pair_token_cosine(src_tokens, cand_tokens)
    tfidf_word_cos = pair_tfidf_cosine(src_norm, cand_norm, analyzer="word", ngram_range=(1, 2))
    tfidf_char_cos = pair_tfidf_cosine(src_norm, cand_norm, analyzer="char", ngram_range=(3, 5))

    # String-based similarities
    edit_sim = np.array([edit_ratio_from_norm(a, b) for a, b in zip(src_norm, cand_norm)], dtype=float)
    jw = np.array([jaro_winkler_from_norm(a, b) for a, b in zip(src_norm, cand_norm)], dtype=float)
    lcs_sim = np.array([lcs_ratio_from_norm(a, b) for a, b in zip(src_norm, cand_norm)], dtype=float)
    jacc = np.array([jaccard(a, b) for a, b in zip(src_sets, cand_sets)], dtype=float)
    dm = np.array([dmetaphone_match_first_token(a, b) for a, b in zip(src_tokens, cand_tokens)], dtype=float)

    # Combine results
    feats = pd.DataFrame({
        "edit_ratio": edit_sim,
        "jaro_winkler": jw,
        "lcs_ratio": lcs_sim,
        "token_jaccard": jacc,
        "token_cosine": token_cos,
        "tfidf_word_cosine": tfidf_word_cos,
        "tfidf_char_cosine": tfidf_char_cos,
        "dmetaphone_match": dm,
    })

    base = [c for c in ["src_id", "cand_id", "cosine_sim", "src_text", "cand_text"] if c in df.columns]
    return pd.concat([df[base].reset_index(drop=True), feats], axis=1)


#Compute features for all candidate pairs stored in a CSV file
def build_features_for_csv(in_csv: Path, out_dir: Optional[Path] = None) -> Path:

    in_csv = Path(in_csv)
    if out_dir is None:
        out_dir = in_csv.parent.parent / "feature_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{in_csv.stem}_features.csv"

    df = pd.read_csv(in_csv)
    feats = build_matching_features(df)
    feats.to_csv(out_csv, index=False)

    print(f"[done] wrote {len(feats):,} rows to {out_csv}")
    return out_csv


if __name__ == "__main__":
    IN_CANDIDATES = Path("../../data/blocking/blocking_candidates_k40.csv")
    build_features_for_csv(IN_CANDIDATES)
