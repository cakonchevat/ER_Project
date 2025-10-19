from __future__ import annotations
from typing import Dict, Tuple, List, Iterable, Optional
from pathlib import Path
import pandas as pd

Pair = Tuple[int, int]

def union_pairs(constraints_dicts: Iterable[Dict[Pair, str]]) -> set[Pair]:
    pairs: set[Pair] = set()
    for d in constraints_dicts:
        pairs.update(d.keys())
    return pairs


def filter_edges_by_geo_constraints(
    edges_df: pd.DataFrame,
    geo_constraints: List[Dict[Pair, str]],
    src_col: str = "src_id",
    cand_col: str = "cand_id",
    prob_col: str = "prob_match",
    min_prob: float = 0.45,
    output_csv: Optional[str] = None,
    dropped_log_csv: Optional[str] = None
) -> tuple[pd.DataFrame, dict]:
    """
    Filters classifier-predicted edges by:
      • keeping only rows with prob >= min_prob
      • dropping rows that violate geographic constraints
    """

    for col in (src_col, cand_col, prob_col):
        if col not in edges_df.columns:
            raise ValueError(f"Column '{col}' is missing in edges_df.")

    stats = {
        "input_rows": len(edges_df)
    }

    # Step 1: keep only confident predictions
    after_threshold_df = (
        edges_df.loc[edges_df[prob_col] >= min_prob]
        .dropna(subset=[src_col, cand_col])
        .copy()
    )
    stats["after_threshold_rows"] = len(after_threshold_df)

    # Step 2: remove geo-violating pairs
    to_prune_pairs = union_pairs(geo_constraints)
    key_tuples = list(zip(after_threshold_df[src_col].astype(int),
                          after_threshold_df[cand_col].astype(int)))

    mask_prune = pd.Series(key_tuples, index=after_threshold_df.index).isin(to_prune_pairs)

    dropped_df = after_threshold_df.loc[mask_prune].copy()
    filtered_df = after_threshold_df.loc[~mask_prune].copy()

    stats["removed_rows"] = int(mask_prune.sum())
    stats["output_rows"] = len(filtered_df)
    stats["fraction_removed"] = round(mask_prune.mean(), 5)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_csv, index=False)
    if dropped_log_csv:
        Path(dropped_log_csv).parent.mkdir(parents=True, exist_ok=True)
        dropped_df.to_csv(dropped_log_csv, index=False)

    return filtered_df, stats

if __name__ == "__main__":
    from src.constraints.geo_constraints import geo_mismatch_pairs_to_prune
    from src.utils.common_methods import _id2text

    # Loading the xgb model as it was shown as the best performer
    edges = pd.read_csv("../data/classifier_predictions/classifier_predictions_xgb.csv")
    entities = pd.read_csv("../data/original/affiliationstrings_ids.csv")

    # Preparing ID→text mapping
    id2text = _id2text(entities, "id1", "affil1")

    # Applying Geo Constraints
    d_geo = geo_mismatch_pairs_to_prune(edges_df=edges, id2text=id2text)
    filtered_df, stats = filter_edges_by_geo_constraints(
        edges_df=edges,
        geo_constraints=[d_geo],
        src_col="src_id",
        cand_col="cand_id",
        prob_col="prob_match",
        min_prob=0.45,
        output_csv="../data/classifier_predictions/constraints/classifier_predictions_xgb_filtered.csv",
        dropped_log_csv="../data/classifier_predictions/constraints/classifier_predictions_xgb_dropped_geo.csv",
    )

    for key, value in stats.items():
        print(f"{key:25s}: {value}")

