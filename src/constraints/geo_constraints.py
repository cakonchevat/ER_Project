import re
import pandas as pd
from typing import Dict, Tuple, Set, Iterable, List, Optional, Callable

# Whitelisted canonical country names we care about
GEO_COUNTRIES_WHITE_LIST: Set[str] = {
    "United States", "United Kingdom", "Taiwan", "China", "United Arab Emirates",
    "Switzerland", "Greece", "Singapore", "Germany", "Hong Kong", "Canada",
    "Italy", "France", "Australia", "India", "Netherlands", "Israel",
    "Japan", "Brazil", "Denmark",
}

# Ordered regex mappings: acronyms or aliases -> canonical country names
ACRONYM_MAP_ORDERED: List[Tuple[str, str]] = [
    (r"\bUSA\b", "United States"),
    (r"\bUS\b", "United States"),
    (r"\bUK\b", "United Kingdom"),
    (r"\bROC\b", "Taiwan"),
    (r"\bP\.?\s*R\.?\s*China\b", "China"),
    (r"\bPeople's Republic of China\b", "China"),
    (r"\bUAE\b", "United Arab Emirates"),
    (r"\bCH\b", "Switzerland"),
    (r"\bGR(?=[\W_]|$)", "Greece"),
    (r"\bS\'?pore(?=[\W_]|$)", "Singapore"),
    (r"\bSingapor(?=[\W_]|$)", "Singapore"),
    (r"\bHong\s*Kong\b", "Hong Kong"),
]

# Detect dotted acronyms like U.S.A., U.S. or E.U.
PATTERNDOTTED = re.compile(r"(?<![A-Za-z])(?:[A-Z]\.){2,}[A-Z]?(?=\W|$)")


# Removes dots and spaces from acronyms (U.S.A. -> USA)
def undot_acronyms(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    def _repl(m: re.Match) -> str:
        return m.group(0).replace(".", "").replace(" ", "")

    return PATTERNDOTTED.sub(_repl, text)


# Builds regex rules to normalize country acronyms into full canonical names
def country_normalization_rules(
    acronym_map: Iterable[Tuple[str, str]],
    allowed: Set[str]
) -> List[Tuple[re.Pattern, str]]:
    subs: List[Tuple[re.Pattern, str]] = []
    for pat, repl in acronym_map:
        if repl in allowed:
            subs.append((re.compile(pat, re.IGNORECASE), repl))
    return subs


# Creates a function that applies the regex rules above to normalize text
def country_name_normalizer(
    subs: List[Tuple[re.Pattern, str]]
) -> Callable[[str], str]:
    def normalize(text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        out = text
        for regex, replacement in subs:
            out = regex.sub(replacement, out)
        return out
    return normalize


# Compiles regex patterns for each canonical country name
def compile_country_patterns(countries: Set[str]) -> Dict[str, re.Pattern]:
    patterns: Dict[str, re.Pattern] = {}
    for country in countries:
        patterns[country.lower()] = re.compile(
            rf"(?<!\w){re.escape(country)}(?!\w)", re.IGNORECASE
        )
    return patterns


# Extracts all mentioned countries from text using regex patterns
def extract_countries_from_text(
    text: str,
    country_patterns: Dict[str, re.Pattern],
    restrict_to: Optional[Set[str]] = None
) -> Set[str]:
    found: Set[str] = set()
    if not isinstance(text, str) or not text:
        return found

    for cname, pattern in country_patterns.items():
        if pattern.search(text):
            found.add(cname)

    if restrict_to:
        found &= {c.lower() for c in restrict_to}

    return found


# Runs the full normalization pipeline: undot → normalize → extract countries
def extract_canonical_countries(
    text: str,
    normalizer: Callable[[str], str],
    country_patterns: Dict[str, re.Pattern],
    restrict_to: Optional[Set[str]] = None
) -> Set[str]:
    undotted = undot_acronyms(text)
    normalized = normalizer(undotted)
    return extract_countries_from_text(normalized, country_patterns, restrict_to)


# Remove (src_id, cand_id) if both sides mention at least one detected country and
#their country sets are disjoint (no overlapping)
def geo_mismatch_pairs_to_prune(
    edges_df: pd.DataFrame,
    id2text: Dict[int, str],
    restrict_to_countries: Optional[Iterable[str]] = None,
    skip_if_country_unknown: bool = True,
) -> Dict[Tuple[int, int], str]:

    # Prepare the normalizer and regex patterns
    subs = country_normalization_rules(ACRONYM_MAP_ORDERED, GEO_COUNTRIES_WHITE_LIST)
    normalizer = country_name_normalizer(subs)
    patterns = compile_country_patterns(GEO_COUNTRIES_WHITE_LIST)
    restrict_set = {c.lower().strip() for c in restrict_to_countries} if restrict_to_countries else None

    pairs = edges_df[["src_id", "cand_id"]].drop_duplicates()
    to_prune: Dict[Tuple[int, int], str] = {}

    # Compare each pair
    for _, row in pairs.iterrows():
        a_id, b_id = int(row["src_id"]), int(row["cand_id"])
        a_text, b_text = id2text.get(a_id, ""), id2text.get(b_id, "")

        a_countries = extract_canonical_countries(a_text, normalizer, patterns, restrict_set)
        b_countries = extract_canonical_countries(b_text, normalizer, patterns, restrict_set)

        # Skip if one side has no detected country (conservative mode)
        if skip_if_country_unknown and (not a_countries or not b_countries):
            continue

        # Prune only if both have countries and they do not overlap
        if a_countries and b_countries and a_countries.isdisjoint(b_countries):
            left = ";".join(sorted(a_countries))
            right = ";".join(sorted(b_countries))
            to_prune[(a_id, b_id)] = f"geo_mismatch:{left}|{right}"

    return to_prune

