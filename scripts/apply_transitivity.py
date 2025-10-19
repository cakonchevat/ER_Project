from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List, Callable, Optional
import pandas as pd
import re

class DSU:
    def __init__(self):
        self.p: Dict[int, int] = {}
        self.r: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.p:
            self.p[x] = x
            self.r[x] = 0
            return x
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _text_to_tokens(s: str) -> Set[str]:
    # cheap, robust tokenization + lowercase
    if not isinstance(s, str):
        return set()
    return set(t.lower() for t in _WORD_RE.findall(s))

def _build_token_index(ents: pd.DataFrame) -> Dict[int, Set[str]]:
    tok: Dict[int, Set[str]] = {}
    for nid, txt in zip(ents["node_id"].astype(int).values, ents["text"].astype(str).values):
        tok[int(nid)] = _text_to_tokens(txt)
    return tok

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

def _apply_token_jaccard_filter(df: pd.DataFrame,
                                token_index: Dict[int, Set[str]],
                                j_min: Optional[float]) -> pd.DataFrame:
    """Keep edges whose endpoint token-sets have Jaccard >= j_min."""
    if not j_min or j_min <= 0.0 or df.empty:
        return df
    jj = []
    for u, v in zip(df["u"].values, df["v"].values):
        tu = token_index.get(int(u), set())
        tv = token_index.get(int(v), set())
        jj.append(_jaccard(tu, tv))
    out = df.copy()
    out["token_jaccard"] = jj
    out = out.loc[out["token_jaccard"] >= j_min].drop(columns=["token_jaccard"])
    return out

def _mutual_top_k_filter(df: pd.DataFrame, src_col: str, cand_col: str, prob_col: str, k: Optional[int]) -> pd.DataFrame:
    if not k or k <= 0:
        return df
    a = df.sort_values([src_col, prob_col], ascending=[True, False]).groupby(src_col).head(k)
    b = df.sort_values([cand_col, prob_col], ascending=[True, False]).groupby(cand_col).head(k)
    a = a.copy(); b = b.copy()
    a["u"] = a[[src_col, cand_col]].min(axis=1).astype(int)
    a["v"] = a[[src_col, cand_col]].max(axis=1).astype(int)
    b["u"] = b[[src_col, cand_col]].min(axis=1).astype(int)
    b["v"] = b[[src_col, cand_col]].max(axis=1).astype(int)
    ab = pd.merge(a[["u","v"]].drop_duplicates(), b[["u","v"]].drop_duplicates(), on=["u","v"], how="inner")
    out = pd.merge(df, ab, on=["u","v"], how="inner")
    return out

def _apply_degree_cap(df: pd.DataFrame, cap: Optional[int]) -> pd.DataFrame:
    if not cap or cap <= 0 or df.empty:
        return df
    deg = pd.concat([df["u"], df["v"]]).value_counts()
    deg_u = df["u"].map(deg).fillna(0).astype(int)
    deg_v = df["v"].map(deg).fillna(0).astype(int)
    mask = (deg_u < cap) & (deg_v < cap)
    return df.loc[mask]

def _refine_clusters_by_strong_edges(final_clusters: pd.DataFrame,
                                     edges_uvp: pd.DataFrame,
                                     bridge_prob: Optional[float]) -> pd.DataFrame:
    """
    Split clusters by removing weak bridges: within each cluster, keep only edges with prob >= bridge_prob,
    then take connected components to produce subclusters. This reduces over-merging from single-link chaining.
    """
    if not bridge_prob or bridge_prob <= 0.0 or edges_uvp.empty:
        return final_clusters

    # map node -> current cluster id
    node2cid = dict(zip(final_clusters["node_id"].astype(int).values,
                        final_clusters["cluster_id"].astype(int).values))

    # group edges by their current cluster (only edges fully inside the cluster)
    edges = edges_uvp.copy()
    edges["_cid_u"] = edges["u"].map(node2cid)
    edges["_cid_v"] = edges["v"].map(node2cid)
    inside = edges.loc[edges["_cid_u"].notna() & (edges["_cid_u"] == edges["_cid_v"])].copy()
    inside = inside.loc[inside["prob"] >= bridge_prob]  # keep only "strong" links for precision

    if inside.empty:
        # nothing to split by; return original
        return final_clusters

    new_cluster_id = 0
    new_rows: List[Tuple[int, int]] = []

    for cid, sub in inside.groupby("_cid_u"):
        dsu = DSU()
        nodes_in_cluster = final_clusters.loc[final_clusters["cluster_id"] == int(cid), "node_id"].astype(int).tolist()
        for n in nodes_in_cluster:
            dsu.find(int(n))
        for _, r in sub.iterrows():
            dsu.union(int(r["u"]), int(r["v"]))

        comp_map: Dict[int, List[int]] = {}
        for n in nodes_in_cluster:
            root = dsu.find(int(n))
            comp_map.setdefault(root, []).append(int(n))

        for comp_nodes in comp_map.values():
            for n in sorted(comp_nodes):
                new_rows.append((int(n), int(new_cluster_id)))
            new_cluster_id += 1

    used_nodes = set(n for n, _ in new_rows)
    remaining = final_clusters.loc[~final_clusters["node_id"].isin(used_nodes), "node_id"].astype(int).tolist()
    for n in sorted(remaining):
        new_rows.append((int(n), int(new_cluster_id)))
        new_cluster_id += 1

    refined = pd.DataFrame(new_rows, columns=["node_id", "cluster_id"])
    sizes = refined["cluster_id"].value_counts().rename("cluster_size")
    refined = refined.merge(sizes, left_on="cluster_id", right_index=True, how="left")
    refined = refined.sort_values(["cluster_id", "node_id"]).reset_index(drop=True)
    return refined

def run(
    input_csv: str,
    entities_csv: str,
    entities_id_col: str,
    entities_text_col: str,
    output_csv: str,
    src_col: str,
    cand_col: str,
    prob_col: str,
    min_prob: float,
    mutual_top_k: Optional[int] = None,
    degree_cap: Optional[int] = None,
    token_jaccard_min: Optional[float] = None,
    bridge_prob: Optional[float] = None,
) -> pd.DataFrame:
    edges = pd.read_csv(input_csv)
    ents = pd.read_csv(entities_csv)[[entities_id_col, entities_text_col]].rename(
        columns={entities_id_col: "node_id", entities_text_col: "text"}
    )

    for c in (src_col, cand_col, prob_col):
        if c not in edges.columns:
            raise ValueError(f"Missing column in edges: {c}")

    edges[src_col] = edges[src_col].astype(int)
    edges[cand_col] = edges[cand_col].astype(int)

    strong = edges.loc[edges[prob_col] >= min_prob, [src_col, cand_col, prob_col]].copy()
    if strong.empty:
        all_nodes = ents["node_id"].astype(int).unique()
        final = pd.DataFrame({"node_id": all_nodes})
        final["cluster_id"] = final["node_id"]
        cid_map = {cid: i for i, cid in enumerate(sorted(pd.Series(final["cluster_id"]).unique()))}
        final["cluster_id"] = final["cluster_id"].map(cid_map).astype(int)
        sizes = final["cluster_id"].value_counts().rename("cluster_size")
        final = final.merge(sizes, left_on="cluster_id", right_index=True, how="left")
        final = final.sort_values(["cluster_id", "node_id"]).reset_index(drop=True)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_csv, index=False)
        print(f"[OK] {len(final)} rows → {output_csv} (no edges ≥ {min_prob})")
        return final

    strong["u"] = strong[[src_col, cand_col]].min(axis=1).astype(int)
    strong["v"] = strong[[src_col, cand_col]].max(axis=1).astype(int)
    strong = strong.drop_duplicates(subset=["u", "v"], keep="first").rename(columns={prob_col: "prob"})

    strong = _mutual_top_k_filter(strong, src_col, cand_col, "prob", mutual_top_k)

    token_index = _build_token_index(ents)
    strong = _apply_token_jaccard_filter(strong, token_index, token_jaccard_min)

    strong = _apply_degree_cap(strong, degree_cap)

    if strong.empty:
        all_nodes = ents["node_id"].astype(int).unique()
        final = pd.DataFrame({"node_id": all_nodes})
        final["cluster_id"] = final["node_id"]
        cid_map = {cid: i for i, cid in enumerate(sorted(pd.Series(final["cluster_id"]).unique()))}
        final["cluster_id"] = final["cluster_id"].map(cid_map).astype(int)
        sizes = final["cluster_id"].value_counts().rename("cluster_size")
        final = final.merge(sizes, left_on="cluster_id", right_index=True, how="left")
        final = final.sort_values(["cluster_id", "node_id"]).reset_index(drop=True)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_csv, index=False)
        print(f"[OK] {len(final)} rows → {output_csv} (filters removed all edges)")
        return final

    dsu = DSU()
    for _, r in strong.iterrows():
        dsu.union(int(r["u"]), int(r["v"]))

    all_nodes = ents["node_id"].astype(int).unique()
    final = pd.DataFrame({"node_id": all_nodes})
    final["cluster_id"] = final["node_id"].apply(lambda n: dsu.find(int(n)))
    cid_map = {cid: i for i, cid in enumerate(sorted(pd.Series(final["cluster_id"]).unique()))}
    final["cluster_id"] = final["cluster_id"].map(cid_map).astype(int)

    sizes = final["cluster_id"].value_counts().rename("cluster_size")
    final = final.merge(sizes, left_on="cluster_id", right_index=True, how="left")
    final = final.sort_values(["cluster_id", "node_id"]).reset_index(drop=True)

    refined = _refine_clusters_by_strong_edges(final, strong[["u","v","prob"]], bridge_prob)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    refined.to_csv(output_csv, index=False)
    print(f"[OK] {len(refined)} rows → {output_csv} "
          f"(transitive clusters ≥ {min_prob}, topK={mutual_top_k}, degree_cap={degree_cap}, "
          f"token_jaccard_min={token_jaccard_min}, bridge_prob={bridge_prob})")
    return refined


if __name__ == "__main__":
     run(
        input_csv="../data/classifier_predictions/constraints/classifier_predictions_xgb_filtered.csv",
        entities_csv="../data/original/affiliationstrings_ids.csv",
        entities_id_col="id1",
        entities_text_col="affil1",
        output_csv="../data/transitivity_applied/clusters_transitivity_applied.csv",
        src_col="src_id",
        cand_col="cand_id",
        prob_col="prob_match",
        min_prob=0.60,
        mutual_top_k=5,
        degree_cap=25,
        token_jaccard_min=0.25,
        bridge_prob=0.65,
    )
