import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple, Set
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.utils.common_methods import tokenize

# Computes K-sweep metrics for different k values using TF-IDF cosine kNN results
# Evaluates how well top-k neighbors share important tokens and how efficiently blocking reduces candidate pairs.
TEXT_COL = "affil1"

CSV_PATH = Path(r"../../data/tokenized_dataset/affiliationstrings_ids_with_tokens.csv")
K_VALUES = [5, 10, 20, 30, 40, 50, 75, 100]

NGRAM_MIN = 1
NGRAM_MAX = 2
MIN_DF_ABS = 2
MAX_DF_FRAC = 0.99

LOW_IDF_PERCENTILE = 0.20
HIGH_DF_PERCENT = 0.20

KEEP_ACRONYMS = True
COMPUTE_JACCARD = True
JACCARD_MIN = 0.30

ACRONYM_RE = re.compile(r"[A-Z0-9&\-]{2,}")
NON_ALNUMERIC = re.compile(r"[^A-Za-z0-9]")

#Finds acronyms and removes non alfa-numeric characters
def find_acronyms(raw: str) -> Set[str]:
    out: Set[str] = set()
    for w in ACRONYM_RE.findall(raw or ""):
        cleaned = NON_ALNUMERIC.sub("", w)
        if len(cleaned) >= 2 and cleaned.isupper():
            out.add(cleaned.lower())
    return out


def build_tfidf(texts: List[str]) -> Tuple:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(NGRAM_MIN, NGRAM_MAX),
        min_df=MIN_DF_ABS,
        max_df=MAX_DF_FRAC,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    return X, vec

# Learns the stopwords from the dataset based on the tf-idf vedctor
# Stopwords = low - IDF( <= percentile) ∪ high - DF( > percent)
def derive_stopwords_auto(X, vec: TfidfVectorizer) -> Set[str]:

    idf = vec.idf_
    N = X.shape[0]
    idf_cut = np.quantile(idf, LOW_IDF_PERCENTILE)
    low_idf_cols = set(np.where(idf <= idf_cut)[0])

    df_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    df_frac = df_counts / N if N > 0 else np.zeros_like(df_counts, dtype=float)
    high_df_cols = set(np.where(df_frac > HIGH_DF_PERCENT)[0])

    vocab = vec.vocabulary_
    inv_vocab = {j: i for i, j in vocab.items()}
    stop_cols = low_idf_cols | high_df_cols
    return {inv_vocab[j] for j in stop_cols}

# Finds the important token sets per row by removing stopwords ((tokens ∩ vocab) \ stopwords [+ acronyms])
def important_token_sets(texts: List[str], vec: TfidfVectorizer, stopwords: Set[str]) -> List[Set[str]]:

    vocab = set(vec.vocabulary_.keys())
    sigs: List[Set[str]] = []
    for raw in texts:
        toks = set(tokenize(raw))
        imp = (toks & vocab) - stopwords
        if KEEP_ACRONYMS:
            imp |= find_acronyms(raw)
        sigs.append(imp)
    return sigs

#Returns a matrix with ids and distances
def knn_indices(X, max_k: int):
    n_neighbors = (max_k + 1) if max_k > 0 else 1
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, idx = nn.kneighbors(X, return_distance=True)
    similarities = 1.0 - distances
    return idx, similarities


#Jaccard similarity between two sets
def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

def visualize_k_values():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    N = len(texts)

    X, vec = build_tfidf(texts)
    stopwords = derive_stopwords_auto(X, vec)
    important_sets = important_token_sets(texts, vec, stopwords)

    max_k = min(max(K_VALUES), max(2, N - 1))
    neigh_idx, similarities = knn_indices(X, max_k)

    ks = [k for k in K_VALUES if k < N]
    overlap_purity, jacc_purity, avgcos, redratio = [], [], [], []

    for k in ks:
        top_idx = neigh_idx[:, 1:k + 1]
        top_similarities = similarities[:, 1:k + 1]

        overlap_hits = []
        jacc_hits = [] if COMPUTE_JACCARD \
            else None

        for i in range(N):
            src_set = important_sets[i]
            share = 0
            jac = 0
            for nb in top_idx[i]:
                nb_set = important_sets[int(nb)]
                if src_set & nb_set:
                    share += 1
                if COMPUTE_JACCARD and jaccard(src_set, nb_set) >= JACCARD_MIN:
                    jac += 1
            overlap_hits.append(share / k)
            if COMPUTE_JACCARD:
                jacc_hits.append(jac / k)

        overlap_purity.append(float(np.mean(overlap_hits)) if overlap_hits else 0.0)
        if COMPUTE_JACCARD:
            jacc_purity.append(float(np.mean(jacc_hits)) if jacc_hits else 0.0)
        avgcos.append(float(top_similarities.mean()) if k > 0 else 0.0)

        #Reduction ratio:
        tota_pairs = N * (N - 1)
        redratio.append(1.0 - (N * k / tota_pairs) if tota_pairs > 0 else 0.0)

    def plot_xy(x, y, xl, yl, title):
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    plot_xy(ks, overlap_purity, "k", "OverlapPurity@k (≥1 important token)", "Token Overlap vs k")
    if COMPUTE_JACCARD:
        plot_xy(ks, jacc_purity, "k", f"JaccardPurity@k (J ≥ {JACCARD_MIN})", "Jaccard Purity vs k")
    plot_xy(ks, avgcos, "k", "AvgCosine@k", "Average Cosine Similarity vs k")
    plot_xy(ks, redratio, "k", "Reduction Ratio (directed)", "Reduction Ratio vs k")

    plt.show()

if __name__ == "__main__":
    visualize_k_values()
