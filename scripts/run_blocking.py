import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

TEXT_COL = "affil1"
ID_COL = "id1"

# Embedding-based blocking with KNN
# Outputs candidate pairs and a mapped CSV with src_text and cand_text
def build_tfidf(texts: List[str], ngram_min: int = 1, ngram_max: int = 2,
                min_df: int = 2, max_df: float = 0.9):
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    return X, vec

def knn_indices(X, k: int):
    N = X.shape[0]
    k_eff = max(0, min(k, N - 1))
    n_neighbors = (k_eff + 1) if k_eff > 0 else 1

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    similarities = 1.0 - distances  # cosine similarity = 1 - cosine distance
    return indices, similarities


# Build candidate pairs (src_id, cand_id, cosine_sim), excluding self rows.
# If undirected=True, collapse (a,b) and (b,a) keeping max similarity.
def build_candidates(ids: List, neighbor_indices: np.ndarray, similarities: np.ndarray,
                     undirected: bool = False, min_sim: Optional[float] = None, ) -> pd.DataFrame:
    N = len(ids)
    rows = []
    for i in range(N):
        nbrs = neighbor_indices[i, 1:]
        sims = similarities[i, 1:]
        for j, s in zip(nbrs, sims):
            j = int(j)
            if j == i:
                continue
            if (min_sim is not None) and (s < min_sim):
                continue
            rows.append((ids[i], ids[j], float(s)))

    df = pd.DataFrame(rows, columns=["src_id", "cand_id", "cosine_sim"])

    if undirected and not df.empty:
        # collapse (a,b) and (b,a) keeping the higher similarity
        pair_key = df.apply(lambda r: tuple(sorted((r["src_id"], r["cand_id"]))), axis=1)
        df = (
            df.assign(pair_key=pair_key)
            .sort_values("cosine_sim", ascending=False)
            .groupby("pair_key", as_index=False)
            .first()[["src_id", "cand_id", "cosine_sim"]]
        )
    return df


def attach_original_texts(cand_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    id_to_text = dict(zip(df[ID_COL], df[TEXT_COL]))
    out = cand_df.copy()
    out["src_text"] = out["src_id"].map(id_to_text)
    out["cand_text"] = out["cand_id"].map(id_to_text)
    return out[["src_id", "cand_id", "cosine_sim", "src_text", "cand_text"]]


def run_blocking(csv_path: Path, k: int = 20, out_csv: Optional[Path] = None,
                 undirected: bool = False, min_sim: Optional[float] = None, ngram_min: int = 1,
                 ngram_max: int = 2, min_df: int = 2, max_df: float = 0.9, ) -> Path:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    ids = df[ID_COL].tolist()

    X, _ = build_tfidf(texts, ngram_min, ngram_max, min_df, max_df)
    neighbor_indices, similarities = knn_indices(X, k)

    cand_df = build_candidates(ids, neighbor_indices, similarities,
                               undirected=undirected, min_sim=min_sim)

    mapped = attach_original_texts(cand_df, df)
    out_csv = out_csv or Path("../data/blocking") / f"blocking_candidates_k{k}.csv"
    mapped.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv} with {len(mapped):,} rows")
    return out_csv


if __name__ == "__main__":
    CSV = Path("../data/tokenized_dataset/affiliationstrings_ids_with_tokens.csv")
    K = 40
    run_blocking(csv_path=CSV, k=K, undirected=True, min_sim=None)