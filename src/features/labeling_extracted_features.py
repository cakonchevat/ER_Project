from pathlib import Path

import numpy as np
import pandas as pd

# In order to pass a dataset to a classifier_predictions, we need that dataset to be labeled
# However, up until now we only have a dataset where each row corresponds to a possible match of entities (src_id with cand_id)
# and a vector of features (like edit_ratio, token_jaccard and so on), that quantify the textual (or phonetic) similarity between the two entities.

# Before a classifier_predictions can predict whether a match from a possible-match entry, it needs ground truth labels.
# That is why this function exists -> to label the feature_extraction_k40.csv dataset
def attach_labels(features_csv: Path, mapping_csv: Path, labeled_csv: Path | None = None) -> Path:
    df = pd.read_csv(features_csv)
    gold_mapping = pd.read_csv(mapping_csv, header=None, names=["src_id","cand_id"])

    # Normalizing order (making pair (a,b) equivalent to (b,a))
    gold_mapping["pair_key"] = list(map(tuple, np.sort(gold_mapping[["src_id", "cand_id"]].values, axis=1)))
    df["pair_key"] = list(map(tuple, np.sort(df[["src_id", "cand_id"]].values, axis=1)))
    # 1	7007	(1, 7007)
    # 7007	1	(1, 7007)

    # Attaches binary labels 1 if a match, 0 if not
    gold_set = set(gold_mapping["pair_key"])
    df["label"] = df["pair_key"].isin(gold_set).astype(int)

    df.drop(columns=["pair_key"], inplace=True)
    if labeled_csv is None:
        labeled_csv = features_csv.with_name(features_csv.stem + "_labeled.csv")
    df.to_csv(labeled_csv, index=False)

    print(f"[labels] {labeled_csv} written with {df['label'].sum()} positives out of {len(df)} rows")
    return labeled_csv


attach_labels(
    features_csv=Path("../../data/feature_extraction/blocking_candidates_k40_features.csv"),
    mapping_csv=Path("../../data/original/affiliationstrings_mapping.csv")
)
