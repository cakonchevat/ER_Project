from __future__ import annotations
from typing import Dict, Tuple, List, Iterable, Optional
from pathlib import Path
import pandas as pd

Pair = Tuple[int, int]  # (src_id, cand_id)

def _union_pairs(constraints_dicts: Iterable[Dict[Pair, str]]) -> set[Pair]:
    """Collect unique (src_id, cand_id) pairs to prune; ignore values/reasons."""
    pairs: set[Pair] = set()
    for d in constraints_dicts:
        pairs.update(d.keys())
    return pairs

def apply_constraints_filtered_only(
    edges_df: pd.DataFrame,
    constraints_dicts: List[Dict[Pair, str]],
    src_col: str = "src_id",
    cand_col: str = "cand_id",
    prob_col: str = "prob_match",
    min_prob: float = 0.45,
    output_csv: Optional[str] = None,
    dropped_log_csv: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    1) Keep rows with prob_col >= min_prob.
    2) Drop rows whose (src_id, cand_id) are in constraints (geo only, as provided).
    3) Return filtered_df + stats. Optionally write:
       - output_csv: kept rows
       - dropped_log_csv: ONLY the rows that were dropped by constraints (no reasons).
    """
    # column checks
    for col in (src_col, cand_col, prob_col):
        if col not in edges_df.columns:
            raise ValueError(f"Column '{col}' is missing in edges_df.")

    stats = {"input_rows": len(edges_df)}

    # 1) threshold filter
    after_threshold_df = edges_df.loc[edges_df[prob_col] >= min_prob].copy()
    stats["after_threshold_rows"] = len(after_threshold_df)

    # 2) constraints union (expecting GEO dict(s) only)
    to_prune_pairs = _union_pairs(constraints_dicts)
    stats["unique_pairs_to_prune"] = len(to_prune_pairs)

    # 3) build prune mask
    key_df = after_threshold_df[[src_col, cand_col]].astype(int)
    mask_prune = key_df.apply(lambda r: (r[src_col], r[cand_col]) in to_prune_pairs, axis=1)

    # split kept vs dropped
    dropped_df  = after_threshold_df.loc[mask_prune].copy()
    filtered_df = after_threshold_df.loc[~mask_prune].copy()

    stats["removed_rows"] = int(mask_prune.sum())
    stats["output_rows"]  = len(filtered_df)

    # 4) optional writes
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_csv, index=False)
    if dropped_log_csv:
        Path(dropped_log_csv).parent.mkdir(parents=True, exist_ok=True)
        # No reasoning/why column — per request, just the dropped rows themselves.
        dropped_df.to_csv(dropped_log_csv, index=False)

    return filtered_df, stats

# -------------------- Runner (single kept CSV + dropped log CSV) --------------------

def run_apply_constraints_from_csv_single(
    input_csv: str = "classifier_predictions_xgb.csv",
    output_csv: str = "classifier_predictions_xgb_filtered.csv",
    dropped_log_csv: str = "classifier_predictions_xgb_dropped_geo.csv",
    constraints_dicts: Optional[List[Dict[Pair, str]]] = None,
    src_col: str = "src_id",
    cand_col: str = "cand_id",
    prob_col: str = "prob_match",
    min_prob: float = 0.45,
) -> dict:
    """
    Reads input_csv, applies threshold + GEO constraints, and writes:
      - output_csv: kept rows
      - dropped_log_csv: ONLY the rows dropped by GEO (no reasons)
    """
    if not constraints_dicts:
        raise ValueError("constraints_dicts is empty — pass the GEO dict(s).")

    df = pd.read_csv(input_csv)
    _, stats = apply_constraints_filtered_only(
        edges_df=df,
        constraints_dicts=constraints_dicts,
        src_col=src_col,
        cand_col=cand_col,
        prob_col=prob_col,
        min_prob=min_prob,
        output_csv=output_csv,
        dropped_log_csv=dropped_log_csv,
    )

    print("=== APPLY CONSTRAINTS (SINGLE OUTPUT + DROPPED LOG) ===")
    print(f"Input rows:            {stats['input_rows']}")
    print(f"After threshold (≥{min_prob}): {stats['after_threshold_rows']}")
    print(f"Unique pairs to prune: {stats['unique_pairs_to_prune']}")
    print(f"Removed rows:          {stats['removed_rows']}")
    print(f"Output rows:           {stats['output_rows']}")
    print(f"Wrote kept rows to:    {output_csv}")
    print(f"Wrote dropped rows to: {dropped_log_csv}")

    return stats

if __name__ == "__main__":
    # GEO constraint only
    from src.constraints.geo_constraints import geo_mismatch_pairs_to_prune
    from src.common_methods import _id2text

    edges = pd.read_csv("../data/classifier_predictions/classifier_predictions_xgb.csv")
    entities = pd.read_csv("../data/original/affiliationstrings_ids.csv")

    # If your GEO constraint needs id->text mapping:
    id2text = _id2text(entities, "id1", "affil1")

    # Build ONLY the GEO dictionary
    d_geo = geo_mismatch_pairs_to_prune(edges_df=edges, id2text=id2text)

    # Pass only GEO dict
    constraints = [d_geo]

    run_apply_constraints_from_csv_single(
        input_csv="../data/classifier_predictions_xgb.csv",
        output_csv="../data/classifier_predictions_xgb_filtered.csv",
        dropped_log_csv="../data/classifier_predictions_xgb_dropped_geo.csv",
        constraints_dicts=constraints,
        src_col="src_id",
        cand_col="cand_id",
        prob_col="prob_match",
        min_prob=0.45,
    )
