import pandas as pd
from models.ner.ner_extractor import ExtractedEntity
import logging
from typing import List, Dict, Any
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Converting the outputs from NERExtractor into small DataFrame with two token columns:
# affil_tokens_labeled and affil_tokens
class TokenProcessor:

    @staticmethod
    def _get_text(e: ExtractedEntity | Dict[str, Any]) -> str:
        return (getattr(e, "text", None) or e.get("text") or "").strip()

    @staticmethod
    def _get_label(e: ExtractedEntity | Dict[str, Any]) -> str:
        return (getattr(e, "label", None) or e.get("label") or "").strip()

    # Returns each entry of the dataset divided by its entities with symbol ;
    # and label based on the type of entity extracted
    @staticmethod
    def extracted_tokens_with_labels(entities: List[ExtractedEntity]) -> str:
        seen = set()
        out = []
        for entity in entities:
            text = TokenProcessor._get_text(entity).rstrip(";:,")
            label = TokenProcessor._get_label(entity)
            if not text:
                continue
            key = (text.lower(), label)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{text}<{label}>")
        return "; ".join(out)

    # Same as previous function, but doesn't label the entities
    @staticmethod
    def extracted_tokens(entities: List[ExtractedEntity]) -> str:
        seen = set()
        out = []
        for e in entities:
            text = TokenProcessor._get_text(e).rstrip(";:,")
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return "; ".join(out)

    # Putting together a DataFrame of columns with extracted tokens
    @staticmethod
    def results_to_tokens_df(results: List[Dict[str, Any]], id_column: str = "id1") -> pd.DataFrame:
        rows = []
        for item in results:
            entities = item.get("entities", []) or []
            tokens = TokenProcessor.extracted_tokens(entities)
            tokens_labeled = TokenProcessor.extracted_tokens_with_labels(entities)

            # Fallback: if no entities are found, leave the original string
            if not tokens:
                raw = (item.get("affiliation") or "").strip()
                tokens = raw
                tokens_labeled = f"{raw}<RAW>" if raw else ""

            rows.append({
                id_column: item["id"],
                "affil_tokens": tokens,
                "affil_tokens_labeled": tokens_labeled
            })
        return pd.DataFrame(rows)

    # Merging the columns with the original DF
    @staticmethod
    def merge_tokens_into_original_csv(original_df: pd.DataFrame, tokens_df: pd.DataFrame, id_column: str = "id1") -> pd.DataFrame:
        if id_column in original_df.columns and id_column in tokens_df.columns:
            merged = original_df.merge(tokens_df, on=id_column, how="left")
        else:
            tokens_df = tokens_df.rename(columns={id_column: "index"})
            merged = (original_df.reset_index().merge(tokens_df, on="index", how="left").drop(columns=["index"]))
        return merged
