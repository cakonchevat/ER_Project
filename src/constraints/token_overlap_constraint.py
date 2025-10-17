import re
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from src.common_methods import _tokenize
from typing import Dict, Tuple, Set, Iterable, List, Optional, Callable


# ============================================================
# 3) Token-overlap constraint (stopword-aware Jaccard)
# ============================================================

def jaccard_overlap(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

def token_overlap_pairs_to_prune(
    edges_df: pd.DataFrame,
    id2text: Dict[int, str],
    stopwords: Set[str],
    min_jaccard: float = 0.20,
    prob_col: str = "prob_match",
    threshold: Optional[float] = None,
) -> Dict[Tuple[int, int], str]:
    """
    Врати речник со {(src_id, cand_id): reason} за парови што ТРЕБА да се избришат
    според token-overlap правилото.

    - Ако threshold е зададен, прво филтрираме edges_df на prob_col >= threshold.
    - Ако некој текст е празен (или по стоп-ови станува празен сет), НЕ го бришеме парот
      (исто како твојата претходна логика: "don't prune on missing tokens").

    reason е кратка ознака, напр. "token_overlap<0.20"
    """
    # 0) ако има threshold – исечи ги кандидатите
    if threshold is not None:
        if prob_col not in edges_df.columns:
            raise ValueError(f"Column '{prob_col}' not found in edges_df.")
        work = edges_df.loc[edges_df[prob_col] >= threshold, ["src_id", "cand_id", prob_col]].copy()
    else:
        # работи врз сите редови (но очекуваме да има src_id/cand_id)
        work = edges_df[["src_id", "cand_id"]].copy()

    to_prune: Dict[Tuple[int, int], str] = {}
    reason = f"token_overlap<{min_jaccard:.2f}"

    # 1) итерација по редовите (доволно брзо за типични големини)
    for _, row in work.iterrows():
        src = int(row["src_id"])
        cand = int(row["cand_id"])

        a = id2text.get(src, "")
        b = id2text.get(cand, "")

        # стопворд-аван токенизација
        at = set(_tokenize(a)) - stopwords
        bt = set(_tokenize(b)) - stopwords

        # Ако не останале токени → НЕ бриши (конзервативно)
        if not at or not bt:
            continue

        jac = jaccard_overlap(at, bt)
        if jac < min_jaccard:
            to_prune[(src, cand)] = reason

    return to_prune