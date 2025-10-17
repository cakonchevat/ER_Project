from collections import Counter
import pandas as pd
from src.common_methods import _tokenize

# ============================================================
# 1) Automatic stopwords (dataset-derived)
#    - build from token frequency across the dataset
# ============================================================

def build_stopwords(df: pd.DataFrame, text_col: str, freq_cutoff: float = 0.30) -> set[str]:
    """
    Very simple extractor: any token whose document frequency >= freq_cutoff becomes a stopword.
    """
    n = len(df)
    if n == 0:
        return set()
    # document frequency of tokens
    dfreq = Counter()
    for s in df[text_col].astype(str):
        toks = set(_tokenize(s))
        dfreq.update(toks)
    return {t for t, c in dfreq.items() if (c / n) >= freq_cutoff}
