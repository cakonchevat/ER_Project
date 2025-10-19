from pathlib import Path
import pandas as pd
from src.graph.build_graph_from_predictions import build_graph_from_predictions
from src.graph.visualize_graph_utils import (
    sample_subgraph, communities_louvain_or_cc, visualize_graph, export_for_gephi
)

def main():
    df_pred = pd.read_csv("../../data/classifier_predictions/constraints/classifier_predictions_xgb_filtered.csv")
    graph = build_graph_from_predictions(df_pred)
    sample_graph = sample_subgraph(graph, max_nodes=400)
    node2comm = communities_louvain_or_cc(sample_graph, use_louvain=True)

    visualize_graph(
        sample_graph,
        node2comm=node2comm,
        title="After geo-blocking / Before clustering (threshold=0.45, Louvain)",
        with_labels=False,
        out_path=Path("../../src/graph/images/er_graph_pred.png"),
    )
    export_for_gephi(graph, Path("../../src/graph/images/graph_from_predictions.gexf"))

if __name__ == "__main__":
    main()
