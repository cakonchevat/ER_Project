# scripts/apply_transitivity_cohort_from_text.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Tuple, List
import re
import pandas as pd

# користиме истите правила што ги имаш во geo модулот
from src.constraints.geo_constraints import (
    GEO_COUNTRIES_WHITE_LIST,
    ACRONYM_MAP_ORDERED,
    _undot_acronyms,
    _build_country_subs,
    _make_text_normalizer,
    _compile_country_patterns,
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
    normalizer,
    patterns: Dict[str, "re.Pattern"]
) -> Set[str]:
    """
    Extract set of canonical-lower country names found in text using the
    same acronym normalization and regex patterns as the geo module.
    """
    if not isinstance(text, str) or not text:
        return set()
    t1 = _undot_acronyms(text)
    t2 = normalizer(t1)
    out: Set[str] = set()
    for cname, pat in patterns.items():
        if pat.search(t2):
            out.add(cname)  # cname е lower canonical: "japan", "canada", ...
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
    min_prob: float
) -> pd.DataFrame:
    # 1) Вчитај
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

    # 2) Подготви нормализатор и шаблони
    subs = _build_country_subs(ACRONYM_MAP_ORDERED, GEO_COUNTRIES_WHITE_LIST)
    normalizer = _make_text_normalizer(subs)
    patterns = _compile_country_patterns(GEO_COUNTRIES_WHITE_LIST)

    # 3) Извади земји per node (канонски lower)
    node_text = ents.set_index("node_id")["text"].to_dict()
    nodes_all = pd.unique(pd.concat([edges[src_col], edges[cand_col]], ignore_index=True))
    node2countries: Dict[int, Set[str]] = {}
    for nid in nodes_all:
        node2countries[int(nid)] = _countries_from_text(
            node_text.get(int(nid), ""), normalizer, patterns
        )

    # 4) Земаме САМО силни ребра И со барем една заедничка земја (intersection ≠ ∅)
    strong = edges.loc[edges[prob_col] >= min_prob, [src_col, cand_col, prob_col]].copy()
    rows: List[Tuple[int, int, str]] = []  # (u, v, cohort_country_lower)

    for _, r in strong.iterrows():
        u = int(r[src_col])
        v = int(r[cand_col])
        cu = node2countries.get(u, set())
        cv = node2countries.get(v, set())
        inter = cu.intersection(cv)
        # само ребра со јасна заедничка земја влегуваат како „семе“
        for country in sorted(inter):
            rows.append((u, v, country))

    # Ако нема семе-ребра со заедничка земја: празен излез
    if not rows:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["node_id", "cluster_id", "cluster_size"])
        empty.to_csv(output_csv, index=False)
        print(f"[OK] 0 rows → {output_csv} (no strong same-country edges ≥ {min_prob})")
        return empty

    seed_df = pd.DataFrame(rows, columns=["u", "v", "cohort"])  # cohort = canonical lower (e.g., "japan")

    # 5) Транзитивност ОДДЕЛНО по cohort (земја од текст),
    #    но на КРАЈ глобално пренумерираме cluster_id како единствен број (0..N-1)
    partials: List[pd.DataFrame] = []

    for coh, sub in seed_df.groupby("cohort", dropna=False):
        dsu = DSU()
        for a, b in zip(sub["u"], sub["v"]):
            dsu.union(int(a), int(b))

        # јазлите во оваа кохорта се сите што се појавиле во семето за дадената земја
        nodes = pd.unique(pd.concat([sub["u"], sub["v"]], ignore_index=True)).astype(int)
        roots = {int(n): dsu.find(int(n)) for n in nodes}

        comp = pd.DataFrame(
            {"node_id": list(roots.keys()), "cluster_id": list(roots.values())}
        )

        # локално нормализирај кластер ID-и (конзистентност во рамки на кохортата)
        local_map = {cid: i for i, cid in enumerate(sorted(comp["cluster_id"].unique()))}
        comp["cluster_id"] = comp["cluster_id"].map(local_map).astype(int)

        # привремено задржи ја кохортата за глобално пренумерирање
        comp["cohort"] = coh

        partials.append(comp)

    final = pd.concat(partials, ignore_index=True)

    # Глобално пренумерирање: уникатни (cohort, local_cluster_id) → глобален број 0..K-1
    final["_pair"] = list(zip(final["cohort"], final["cluster_id"]))
    unique_pairs = sorted(final["_pair"].unique())
    global_map = {pair: i for i, pair in enumerate(unique_pairs)}
    final["cluster_id"] = final["_pair"].map(global_map).astype(int)

    # Сега можеме да го исчистиме и да пресметаме cluster_size по глобален ID
    final = final.drop(columns=["_pair", "cohort"])
    sizes = final["cluster_id"].value_counts().rename("cluster_size")
    final = final.merge(sizes, left_on="cluster_id", right_index=True, how="left")

    # Сортирање и запишување
    final = final.sort_values(["cluster_id", "node_id"]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False)
    print(f"[OK] {len(final)} rows → {output_csv} (transitive clusters, strong edges ≥ {min_prob})")
    return final


if __name__ == "__main__":
    run(
        input_csv="../data/classifier_predictions/classifier_predictions_xgb_filtered.csv",
        entities_csv="../data/original/affiliationstrings_ids.csv",
        entities_id_col="id1",        # колона со ID во entities CSV
        entities_text_col="affil1",   # колона со текст (affiliation)
        output_csv="../data/transitivity_applied/er_clusters_transitive.csv",
        src_col="src_id",
        cand_col="cand_id",
        prob_col="prob_match",
        min_prob=0.60,
    )
