import networkx as nx
import pandas as pd
from src.graph.build_graph_from_predictions import build_graph_from_predictions

# load classifier_predictions predictions
df_pred = pd.read_csv("../data/classifier_predictions/classifier_predictions_xgb_filtered.csv")

# build graph (weighted by prob_match)
G = build_graph_from_predictions(df_pred, keep_threshold=0.45)

# run Connected Components clustering
clusters = list(nx.connected_components(G))

# flatten into dataframe
rows = []
for cid, comp in enumerate(clusters):
    for n in comp:
        rows.append((n, cid, len(comp)))
df_clusters = pd.DataFrame(rows, columns=["node_id", "cluster_id", "cluster_size"])

df_clusters.to_csv("../data/clusters_connected_components.csv", index=False)
print(f"[OK] {len(df_clusters)} nodes clustered into {len(clusters)} clusters")
