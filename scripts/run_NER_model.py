import sys
from pathlib import Path
import pandas as pd
sys.path.append(str((Path(__file__).parent.parent / "src").resolve()))
from models.ner.ner_extractor import NERExtractor
from models.ner.token_processor import TokenProcessor


def process_csv_and_write_tokens(
    csv_path: str,
    model_type: str = "spacy",
    model_name: str = "en_core_web_trf",
    backoff_model: str = "dslim/bert-base-NER",
    affiliation_column: str = "affil1",
    id_column: str = "id1",
    batch_size: int = 100,
    entity_types=None,
):
    csv_path = Path(csv_path).resolve()

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Dataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    sample = df[affiliation_column].iloc[0] if len(df) > 0 and affiliation_column in df.columns else "None"
    print(f"  Sample affiliation: {sample}")

    extractor = NERExtractor(
        model_type=model_type,
        model_name=model_name,
        transformers_backoff_model=backoff_model,
    )

    results = extractor.process_dataset(
        data=df,
        affiliation_column=affiliation_column,
        id_column=id_column,
        batch_size=batch_size,
        entity_types=entity_types,
    )

    tokens_df = TokenProcessor.results_to_tokens_df(results, id_column=id_column)
    merged = TokenProcessor.merge_tokens_into_original_csv(df, tokens_df, id_column=id_column)

    output_path = csv_path.with_name(csv_path.stem + "_with_tokens.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Updated CSV written: {output_path}")


if __name__ == "__main__":
    process_csv_and_write_tokens("../data/original/affiliationstrings_ids.csv")
