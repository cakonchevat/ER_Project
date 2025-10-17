"""
K-sweep visualization for TF-IDF + cosine kNN blocking:
- Learns corpus stopwords via low-IDF and high-DF thresholds
- Builds "important token" sets per record (plus acronyms from raw text)
- Sweeps k and plots: overlap purity, optional Jaccard purity, AvgCos@k, reduction ratio
"""
from pathlib import Path
from typing import List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re


LOWER_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")


def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Heuristically detect text and id columns from a DataFrame."""
    text_candidates = [c for c in df.columns if c.lower() in
                       ("affil1", "affiliation", "affil", "text", "affil_clean", "affiliation_text", "name")]
    id_candidates = [c for c in df.columns if c.lower() in
                     ("id1", "id", "record_id", "row_id")]
    text_col = text_candidates[0] if text_candidates else df.columns[0]
    id_col = id_candidates[0] if id_candidates else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    return text_col, id_col


def tokenize(text: str) -> List[str]:
    """Lowercasing tokenization that preserves alnum words and dotted/hyphenated tokens."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [t.lower() for t in LOWER_TOKEN.findall(text)]


def find_acronyms(raw: str) -> Set[str]:
    """Extract all-caps alphanumeric acronyms from the raw string (returned lowercased)."""
    acr = set()
    for w in re.findall(r"[A-Z0-9&\-]{2,}", raw or ""):
        cleaned = re.sub(r"[^A-Za-z0-9]", "", w)
        if len(cleaned) >= 2 and cleaned.isupper():
            acr.add(cleaned.lower())
    return acr


def build_tfidf(
    texts: List[str],
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df_abs: int = 2,
    max_df_frac: float = 0.99,
):
    """Fit TF-IDF vectorizer and transform texts; returns (X, vectorizer)."""
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df_abs,
        max_df=max_df_frac,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    return X, vec


def derive_stopwords_auto(
    X,
    vec: TfidfVectorizer,
    low_idf_pct: float,
    high_df_percent: float,
    keep_acronyms: bool = True,  # kept for API symmetry; acronyms are re-added later from raw text
) -> Set[str]:
    """
    Learn stopwords from corpus:
      - tokens with IDF <= percentile(low_idf_pct)
      - tokens with DF > high_df_percent
    Returns a set of lowercased stopwords. Acronyms are not filtered here; they are re-added per-doc from raw text.
    """
    vocab = vec.vocabulary_
    inv_vocab = {j: i for i, j in vocab.items()}
    idf = vec.idf_
    N = X.shape[0]

    idf_cut = np.quantile(idf, low_idf_pct)
    low_idf_cols = set(np.where(idf <= idf_cut)[0])

    df_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    df_frac = df_counts / N if N > 0 else np.zeros_like(df_counts, dtype=float)
    high_df_cols = set(np.where(df_frac > high_df_percent)[0])

    stop_cols = low_idf_cols | high_df_cols
    return {inv_vocab[j] for j in stop_cols}


def important_token_sets(
    texts: List[str],
    X,
    vec: TfidfVectorizer,
    stopwords: Set[str],
) -> List[Set[str]]:
    """Per-document important token set: (vocab ∩ tokens) \\ stopwords ∪ acronyms(raw)."""
    vocab = set(vec.vocabulary_.keys())
    imp_sets: List[Set[str]] = []
    for raw in texts:
        toks = set(tokenize(raw))
        acrs = find_acronyms(raw)
        imp = (toks & vocab) - stopwords
        imp |= acrs
        imp_sets.append(imp)
    return imp_sets


def knn_indices(X, max_k: int):
    """Return (indices, similarities) for cosine kNN; each row includes self at [:,0]."""
    n_neighbors = (max_k + 1) if max_k > 0 else 1
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    sims = 1.0 - dists
    return idx, sims


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return 0.0 if inter == 0 else inter / float(len(a | b))


def visualize_k_values(
    csv_paths_try: List[str],
    k_values: List[int],
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df_abs: int = 2,
    max_df_frac_for_vocab: float = 0.99,
    low_idf_percentile: float = 0.20,
    high_df_percent: float = 0.20,
    keep_acronyms: bool = True,
    compute_jaccard: bool = True,
    jaccard_min: float = 0.30,
) -> None:
    """Compute purity/quality metrics across k and render plots to select an optimal k."""
    csv_path: Optional[Path] = None
    for p in csv_paths_try:
        pp = Path(p)
        if pp.exists():
            csv_path = pp
            break
    if csv_path is None:
        raise FileNotFoundError(f"CSV not found. Tried: {', '.join(csv_paths_try)}")
    print(f"[info] Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    text_col, id_col = detect_cols(df)
    texts = df[text_col].fillna("").astype(str).tolist()
    N = len(texts)
    print(f"[info] N={N}, text_col='{text_col}', id_col='{id_col}'")

    X, vec = build_tfidf(
        texts,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df_abs=min_df_abs,
        max_df_frac=max_df_frac_for_vocab,
    )

    stopwords = derive_stopwords_auto(
        X,
        vec,
        low_idf_pct=low_idf_percentile,
        high_df_percent=high_df_percent,
        keep_acronyms=keep_acronyms,
    )
    imp_sets = important_token_sets(texts, X, vec, stopwords)

    max_k = min(max(k_values), max(2, N - 1))
    neigh_idx, sims = knn_indices(X, max_k)

    ks = [k for k in k_values if k < N]
    overlap_purity, jacc_purity, avgcos, redratio = [], [], [], []

    for k in ks:
        top_idx = neigh_idx[:, 1:k + 1]
        top_sims = sims[:, 1:k + 1]

        overlap_hits = []
        jacc_hits = []
        for i in range(N):
            src_set = imp_sets[i]
            neighbors = top_idx[i]
            share = 0
            jac = 0
            if k > 0:
                for nb in neighbors:
                    nb_set = imp_sets[int(nb)]
                    if src_set & nb_set:
                        share += 1
                    if compute_jaccard and jaccard(src_set, nb_set) >= jaccard_min:
                        jac += 1
                overlap_hits.append(share / k)
                if compute_jaccard:
                    jacc_hits.append(jac / k)
            else:
                overlap_hits.append(0.0)
                if compute_jaccard:
                    jacc_hits.append(0.0)

        overlap_purity.append(float(np.mean(overlap_hits)) if overlap_hits else 0.0)
        if compute_jaccard:
            jacc_purity.append(float(np.mean(jacc_hits)) if jacc_hits else 0.0)
        avgcos.append(float(top_sims.mean()) if k > 0 else 0.0)

        total_pairs = N * k
        baseline = N * (N - 1)
        redratio.append(1.0 - (total_pairs / baseline) if baseline > 0 else 0.0)

    def plot_xy(x, y, xl, yl, title):
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

    plot_xy(ks, overlap_purity,
            "k (top-k neighbors)", "OverlapPurity@k (≥1 important token)",
            "Token-set Overlap Purity vs k")

    if compute_jaccard:
        plot_xy(ks, jacc_purity,
                "k (top-k neighbors)", f"JaccardPurity@k (J ≥ {jaccard_min})",
                "Jaccard Purity vs k")

    plot_xy(ks, avgcos,
            "k (top-k neighbors)", "AvgCosine@k",
            "Average Cosine Similarity vs k")

    plot_xy(ks, redratio,
            "k (top-k neighbors)", "Reduction Ratio (directed)",
            "Reduction Ratio vs k")

    plt.show()


if __name__ == "__main__":
    CSV_PATHS_TRY = ["../../data/affiliationstrings_ids_with_tokens.csv"]
    K_VALUES = [5, 10, 20, 30, 40, 50, 75, 100]

    visualize_k_values(
        csv_paths_try=CSV_PATHS_TRY,
        k_values=K_VALUES,
        ngram_min=1,
        ngram_max=2,
        min_df_abs=2,
        max_df_frac_for_vocab=0.99,
        low_idf_percentile=0.20,
        high_df_percent=0.20,
        keep_acronyms=True,
        compute_jaccard=True,
        jaccard_min=0.30,
    )
