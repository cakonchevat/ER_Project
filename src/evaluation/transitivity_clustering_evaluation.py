from __future__ import annotations
import pandas as pd
from itertools import combinations
from typing import Tuple, Set, Dict, List

GOLD_CSV = "../../data/original/affiliationstrings_mapping.csv"
PRED_CLUSTERS_CSV = "../../data/transitivity_applied/clusters_transitivity_applied.csv"
ID_COL = "node_id"
CLUSTER_COL = "cluster_id"

Pair = Tuple[int, int]

def _norm_pair(a: int, b: int) -> Pair:
    return (a, b) if a <= b else (b, a)

def load_gold_pairs(path: str) -> Set[Pair]:
    df = pd.read_csv(path, header=None, names=["a", "b"])
    a = df["a"].astype(int).values
    b = df["b"].astype(int).values
    return {_norm_pair(int(x), int(y)) for x, y in zip(a, b)}


def load_pred_clusters(path: str, id_col: str = "id", cluster_col: str = "cluster_id") -> Dict[int, int | str]:
    df = pd.read_csv(path, usecols=[id_col, cluster_col]).dropna()
    df[id_col] = df[id_col].astype(int)
    mode_clusters = (
        df.groupby(id_col)[cluster_col]
        .agg(lambda s: s.value_counts().idxmax())
    )
    return mode_clusters.to_dict()


def evaluate_gold_recall(gold_pairs: Set[Pair], id2cluster: Dict[int, int | str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for u, v in gold_pairs:
        cu = id2cluster.get(u)
        cv = id2cluster.get(v)
        if cu is None or cv is None:
            rows.append({
                "u": u, "v": v,
                "same_cluster": False,
                "reason": "missing_id",
                "cluster_u": cu, "cluster_v": cv
            })
            continue
        same = (cu == cv)
        rows.append({
            "u": u, "v": v,
            "same_cluster": same,
            "reason": "ok" if same else "split",
            "cluster_u": cu, "cluster_v": cv
        })
    return pd.DataFrame(rows)


def predicted_pairs_from_clusters(id2cluster: Dict[int, int | str]) -> Set[Pair]:
    by_cluster: Dict[int | str, List[int]] = {}
    for _id, c in id2cluster.items():
        by_cluster.setdefault(c, []).append(_id)

    pairs: Set[Pair] = set()
    for ids in by_cluster.values():
        if len(ids) < 2:
            continue
        for a, b in combinations(sorted(ids), 2):
            pairs.add((a, b))
    return pairs


def main():
    gold_pairs = load_gold_pairs(GOLD_CSV)
    id2cluster = load_pred_clusters(PRED_CLUSTERS_CSV, ID_COL, CLUSTER_COL)

    res = evaluate_gold_recall(gold_pairs, id2cluster)
    total = len(res)
    correct = int(res["same_cluster"].sum())
    rate = correct / total if total else 0.0
    missing = int((res["reason"] == "missing_id").sum())
    split = int((res["reason"] == "split").sum())

    print("=== Cluster Evaluation ===")
    print(f"Gold pairs total: {total}")
    print(f"Pairs correctly co-clustered: {correct}")
    print(f"→ Match rate (gold pairs recovered): {rate:.3f}")
    if missing > 0:
        print(f"Pairs with missing ids: {missing}")
    if split > 0:
        print(f"Pairs split across clusters: {split}")

    pred_pairs = predicted_pairs_from_clusters(id2cluster)
    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\n=== Pairwise Metrics (Predicted vs Gold) ===")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Pairwise precision: {precision:.3f}")
    print(f"Pairwise recall:    {recall:.3f}")
    print(f"Pairwise F1:        {f1:.3f}")

if __name__ == "__main__":
    main()
