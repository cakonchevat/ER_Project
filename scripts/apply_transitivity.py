from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List, Callable
import re
import pandas as pd

from src.constraints.geo_constraints import (
    GEO_COUNTRIES_WHITE_LIST,
    ACRONYM_MAP_ORDERED,
    undot_acronyms,
    country_normalization_rules,
    country_name_normalizer,
    compile_country_patterns,
)

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


def _countries_from_text(
    text: str,
    normalizer: Callable[[str], str],
    patterns: Dict[str, re.Pattern],
) -> Set[str]:
    if not isinstance(text, str) or not text:
        return set()
    t1 = undot_acronyms(text)
    t2 = normalizer(t1)
    out: Set[str] = set()
    for cname, pat in patterns.items():
        if pat.search(t2):
            out.add(cname)  # cname is canonical
    return out


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
) -> pd.DataFrame:
    edges = pd.read_csv(input_csv)
    ents = (
        pd.read_csv(entities_csv)[[entities_id_col, entities_text_col]]
        .rename(columns={entities_id_col: "node_id", entities_text_col: "text"})
    )

    for c in (src_col, cand_col, prob_col):
        if c not in edges.columns:
            raise ValueError(f"Missing column in edges: {c}")

    edges[src_col] = edges[src_col].astype(int)
    edges[cand_col] = edges[cand_col].astype(int)

    subs = country_normalization_rules(ACRONYM_MAP_ORDERED, GEO_COUNTRIES_WHITE_LIST)
    normalizer = country_name_normalizer(subs)
    patterns = compile_country_patterns(GEO_COUNTRIES_WHITE_LIST)

    # Extract countries per node (canonical lower)
    node_text = ents.set_index("node_id")["text"].to_dict()
    nodes_all = pd.unique(pd.concat([edges[src_col], edges[cand_col]], ignore_index=True))
    node2countries: Dict[int, Set[str]] = {}
    for nid in nodes_all:
        node2countries[int(nid)] = _countries_from_text(
            node_text.get(int(nid), ""), normalizer, patterns
        )

    # Keep only strong edges that share at least one common country
    strong = edges.loc[edges[prob_col] >= min_prob, [src_col, cand_col, prob_col]].copy()
    rows: List[Tuple[int, int, str]] = []  # (u, v, cohort_country_lower)

    for _, r in strong.iterrows():
        u = int(r[src_col])
        v = int(r[cand_col])
        cu = node2countries.get(u, set())
        cv = node2countries.get(v, set())
        inter = cu.intersection(cv)
        for country in sorted(inter):
            rows.append((u, v, country))

    if not rows:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["node_id", "cluster_id", "cluster_size"])
        empty.to_csv(output_csv, index=False)
        print(f"[OK] 0 rows → {output_csv} (no strong same-country edges ≥ {min_prob})")
        return empty

    seed_df = pd.DataFrame(rows, columns=["u", "v", "cohort"])  # cohort = canonical lower

    # 5) transitivity per cohort, then global renumbering
    partials: List[pd.DataFrame] = []

    for coh, sub in seed_df.groupby("cohort", dropna=False):
        dsu = DSU()
        for a, b in zip(sub["u"], sub["v"]):
            dsu.union(int(a), int(b))

        nodes = pd.unique(pd.concat([sub["u"], sub["v"]], ignore_index=True)).astype(int)
        roots = {int(n): dsu.find(int(n)) for n in nodes}

        comp = pd.DataFrame({"node_id": list(roots.keys()), "cluster_id": list(roots.values())})

        # local normalize cluster IDs within cohort
        local_map = {cid: i for i, cid in enumerate(sorted(comp["cluster_id"].unique()))}
        comp["cluster_id"] = comp["cluster_id"].map(local_map).astype(int)

        comp["cohort"] = coh
        partials.append(comp)

    final = pd.concat(partials, ignore_index=True)

    # global renumbering: (cohort, local_cluster_id) → global id 0..K-1
    final["_pair"] = list(zip(final["cohort"], final["cluster_id"]))
    unique_pairs = sorted(final["_pair"].unique())
    global_map = {pair: i for i, pair in enumerate(unique_pairs)}
    final["cluster_id"] = final["_pair"].map(global_map).astype(int)

    # compute cluster sizes
    final = final.drop(columns=["_pair", "cohort"])
    sizes = final["cluster_id"].value_counts().rename("cluster_size")
    final = final.merge(sizes, left_on="cluster_id", right_index=True, how="left")

    final = final.sort_values(["cluster_id", "node_id"]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False)
    print(f"[OK] {len(final)} rows → {output_csv} (transitive clusters, strong edges ≥ {min_prob})")
    return final


if __name__ == "__main__":
    run(
        input_csv="../data/classifier_predictions/constraints/classifier_predictions_xgb_filtered.csv",
        entities_csv="../data/original/affiliationstrings_ids.csv",
        entities_id_col="id1",
        entities_text_col="affil1",
        output_csv="../data/transitivity_applied/clusters_transitivity_applied_0.3.csv",
        src_col="src_id",
        cand_col="cand_id",
        prob_col="prob_match",
        min_prob=0.60,
    )
