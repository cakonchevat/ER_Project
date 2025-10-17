import re
import pandas as pd
from typing import Dict, Tuple, Set, Iterable, List, Optional, Callable

GEO_COUNTRIES_WHITE_LIST: Set[str] = {
    "United States", "United Kingdom", "Taiwan", "China", "United Arab Emirates",
    "Switzerland", "Greece", "Singapore", "Germany", "Hong Kong", "Canada",
    "Italy", "France", "Australia", "India", "Netherlands", "Israel",
    "Japan", "Brazil", "Denmark",
}

# transformations of acronyms -> full meaning
ACRONYM_MAP_ORDERED: List[Tuple[str, str]] = [
    (r"\bUSA\b",                 "United States"),
    (r"\bUS\b",                  "United States"),
    (r"\bUK\b",                  "United Kingdom"),
    (r"\bROC\b",                 "Taiwan"),
    (r"\bP\.?\s*R\.?\s*China\b", "China"),
    (r"\bPeople's Republic of China\b", "China"),
    (r"\bUAE\b",                 "United Arab Emirates"),
    (r"\bCH\b",                  "Switzerland"),
    (r"\bGR(?=[\W_]|$)",         "Greece"),
    (r"\bS\'?pore(?=[\W_]|$)",   "Singapore"),
    (r"\bSingapor(?=[\W_]|$)",   "Singapore"),
    (r"\bHong\s*Kong\b", "Hong Kong"),
]

PATTERNDOTTED = re.compile(
    r"(?<![A-Za-z])[A-Z](?:\.[A-Z]){1,}.?(?=(?:\W||$))"
)

def _undot_acronyms(text: str) -> str:
    """
    1) Ги наоѓа 'точкасти' акроними (U.S.A., U.S., E.U.)
    2) Ги претвора во форма без точки/празнини (USA, US, EU)
    """
    if not isinstance(text, str) or not text:
        return "" if text is None else text

    def _repl(m: re.Match) -> str:
        token = m.group(0)
        token = token.replace(".", "").replace(" ", "")
        return token

    return PATTERNDOTTED.sub(_repl, text)


def _build_country_subs(acronym_map: Iterable[Tuple[str, str]], allowed: Set[str]) -> List[Tuple[re.Pattern, str]]:
    """Земи само правила што нормализираат во земји и компилирај ги (редоследно)."""
    subs = []
    for pat, repl in acronym_map:
        if repl in allowed:
            subs.append((re.compile(pat, re.IGNORECASE), repl))
    return subs

def _make_text_normalizer(subs: List[Tuple[re.Pattern, str]]) -> Callable[[str], str]:
    """Функција што ќе нормализира текст со дадените правила"""
    def _normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""
        out = text
        for preg, repl in subs:
            out = preg.sub(repl, out)
        return out
    return _normalize

def _compile_country_patterns(countries: Set[str]) -> Dict[str, re.Pattern]:
    """
    За секоја земја прави word-boundary regex за КАНОНСКОТО име.
    (Нема алијаси – тие веќе се решени преку одточкување + нормализација.)
    """
    patterns = {}
    for c in countries:
        patterns[c.lower()] = re.compile(rf"(?<!\w){re.escape(c)}(?!\w)", re.IGNORECASE)
    return patterns

def _extract_countries_from_text(text: str, country_patterns: Dict[str, re.Pattern],
                                 restrict_to: Optional[Set[str]]=None) -> Set[str]:
    """Врати сет од земји (канонски lower) што се спомнуваат во текстот."""
    out = set()
    if not isinstance(text, str) or not text:
        return out
    for cname, pat in country_patterns.items():
        if pat.search(text):
            out.add(cname)
    if restrict_to:
        out &= {c.lower() for c in restrict_to}
    return out

def geo_mismatch_pairs_to_prune(
    edges_df: pd.DataFrame,
    id2text: Dict[int, str],
    restrict_to_countries: Optional[Iterable[str]] = None,
    conservative_when_unknown: bool = True,
) -> Dict[Tuple[int, int], str]:
    """
    Сече парови (src_id, cand_id) само ако:
      - по одточкување + нормализација двете страни имаат земја, и
      - земјите се различни (дисјунктни множества).
    """
    # 0) Прет-чекор: изгради нормализатор за земји
    subs = _build_country_subs(ACRONYM_MAP_ORDERED, GEO_COUNTRIES_WHITE_LIST)
    normalizer = _make_text_normalizer(subs)
    patterns = _compile_country_patterns(GEO_COUNTRIES_WHITE_LIST)
    rset = {c.lower().strip() for c in restrict_to_countries} if restrict_to_countries else None

    # 1) Итерација низ уникатни парови
    work = edges_df[["src_id", "cand_id"]].drop_duplicates().astype({"src_id": int, "cand_id": int})
    to_prune: Dict[Tuple[int, int], str] = {}

    for _, row in work.iterrows():
        a_raw = id2text.get(row["src_id"], "") or ""
        b_raw = id2text.get(row["cand_id"], "") or ""

        # a) прво: ОДТОЧКУВАЊЕ (U.S.A. -> USA; U.S. -> US; E.U. -> EU)
        a_step1 = _undot_acronyms(a_raw)
        b_step1 = _undot_acronyms(b_raw)

        # b) второ: НОРМАЛИЗАЦИЈА во КАНОНСКИ ЗЕМЈИ (USA -> United States, UK -> United Kingdom, CH -> Switzerland, ...)
        a_txt = normalizer(a_step1)
        b_txt = normalizer(b_step1)

        # c) трето: ДЕТЕКЦИЈА на земји (само канонски имиња)
        a_countries = _extract_countries_from_text(a_txt, patterns, rset)
        b_countries = _extract_countries_from_text(b_txt, patterns, rset)

        # d) правило за сечење
        if conservative_when_unknown and (not a_countries or not b_countries):
            continue
        if a_countries and b_countries and a_countries.isdisjoint(b_countries):
            left = ";".join(sorted(a_countries))
            right = ";".join(sorted(b_countries))
            to_prune[(row["src_id"], row["cand_id"])] = f"geo_mismatch:{left}|{right}"

    return to_prune
