from __future__ import annotations
import pandas as pd
import networkx as nx

def build_graph_from_predictions(
    predictions: pd.DataFrame,
    prob_column: str = "prob_match",
    source_column: str = "src_id",
    candidate_column: str = "cand_id",
    threshold: float = 0.45,
) -> nx.Graph:
    required = {prob_column, source_column, candidate_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    g = nx.Graph()
    rows = predictions[
        (predictions[prob_column] >= threshold)
        & (predictions[source_column] != predictions[candidate_column])
    ][[source_column, candidate_column, prob_column]]

    for src, cand, prob in rows.itertuples(index=False):
        src_i, cand_i = int(src), int(cand)
        w = float(prob)
        if g.has_edge(src_i, cand_i):
            prev = float(g[src_i][cand_i].get("weight", 0.0))
            w = max(prev, w)
        g.add_edge(src_i, cand_i, weight=w)

    return g
