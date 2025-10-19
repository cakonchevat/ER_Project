#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from src.graph.build_graph_after_transitivity import build_graph_from_transitivity
from src.graph.visualize_graph_utils import sample_subgraph, visualize_graph, export_for_gephi

def main():
    df_trans = pd.read_csv("../../data/transitivity_applied/clusters_transitivity_applied.csv")
    G = build_graph_from_transitivity(df_trans, strategy="chain")  # or "star"
    G_small = sample_subgraph(G, max_nodes=400)

    visualize_graph(
        G_small,
        title="Post-Transitivity (cluster wiring: chain)",
        with_labels=False,
        out_path=Path("../../src/graph/images/er_graph_trans.png"),
    )
    export_for_gephi(G, Path("../../src/graph/images/er_graph_trans_full.gexf"))

if __name__ == "__main__":
    main()
