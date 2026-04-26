import os
from typing import Iterable, Optional

import pandas as pd


DEFAULT_POSTCODE_PREFIXES = ("W8", "SW7", "SW5", "SW3")


def get_postcode_prefix_filters() -> list[str]:
    raw = os.getenv("POSTCODE_PREFIXES", ",".join(DEFAULT_POSTCODE_PREFIXES))
    if raw.strip().upper() == "ALL":
        return []
    return [part.strip().upper().replace(" ", "") for part in raw.split(",") if part.strip()]


def matches_postcode_prefix(postcode_clean: Optional[str], prefixes: Iterable[str]) -> bool:
    if not postcode_clean:
        return False

    normalized = str(postcode_clean).upper().replace(" ", "").strip()
    prefix_list = list(prefixes)

    if not prefix_list:
        return True

    return any(normalized.startswith(prefix) for prefix in prefix_list)


def filter_dataframe_by_postcode_prefixes(
    df: pd.DataFrame,
    postcode_column: str,
    prefixes: list[str],
) -> pd.DataFrame:
    if not prefixes:
        return df

    return df[
        df[postcode_column]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(" ", "", regex=False)
        .apply(lambda value: matches_postcode_prefix(value, prefixes))
    ].copy()
