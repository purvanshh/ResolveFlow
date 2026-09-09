"""Data loading and light validation for ResolveFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

EXPECTED_COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
)

DEFAULT_RAW_PATH = Path("data/raw/twcs.csv")


def resolve_raw_path(path: str | Path | None = None) -> Path:
    """Return a readable path to the raw CSV."""
    candidate = Path(path) if path is not None else DEFAULT_RAW_PATH
    if not candidate.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {candidate}. "
            "Place twcs.csv under data/raw/ (see README)."
        )
    return candidate


def load_raw_tweets(
    path: str | Path | None = None,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load the Customer Support on Twitter CSV.

    Empty / missing text becomes an empty string. Timestamps are parsed when
    possible; unparseable values become NaT without raising.
    """
    csv_path = resolve_raw_path(path)
    df = pd.read_csv(
        csv_path,
        nrows=nrows,
        dtype={"author_id": str},
        keep_default_na=True,
        na_values=["", "NA", "NaN", "null", "None"],
    )

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected schema; missing columns: {missing}")

    df = df.copy()
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)
    df["author_id"] = df["author_id"].fillna("").astype(str)
    df["inbound"] = _normalize_inbound(df["inbound"])
    # Twitter export format used by this dataset; fall back for fixtures / odd rows.
    parsed = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )
    need_fallback = parsed.isna() & df["created_at"].notna()
    if bool(need_fallback.any()):
        fallback = pd.to_datetime(
            df.loc[need_fallback, "created_at"],
            errors="coerce",
            utc=True,
            format="mixed",
        )
        parsed.loc[need_fallback] = fallback
    df["created_at"] = parsed

    # response_tweet_id can be a comma-separated list of IDs
    df["response_tweet_id"] = df["response_tweet_id"].where(
        df["response_tweet_id"].notna(), None
    )
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    )
    return df


def _normalize_inbound(series: pd.Series) -> pd.Series:
    """Coerce inbound flags to nullable boolean."""
    if series.dtype == bool:
        return series
    mapped = series.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            1: True,
            0: False,
            "1": True,
            "0": False,
        }
    )
    return mapped.astype("boolean")


def find_duplicate_tweet_ids(df: pd.DataFrame) -> pd.Series:
    """Return tweet_id values that appear more than once."""
    counts = df["tweet_id"].value_counts(dropna=True)
    return counts[counts > 1]


def parse_response_ids(value: object) -> list[int]:
    """Parse a response_tweet_id cell into a list of integer IDs."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(float(part)))
        except ValueError:
            continue
    return ids


def brand_authors(df: pd.DataFrame) -> set[str]:
    """Brand accounts are authors of outbound (inbound=False) tweets."""
    mask = df["inbound"] == False  # noqa: E712
    return set(df.loc[mask, "author_id"].dropna().unique())


def is_brand_author(author_id: str, brands: Iterable[str]) -> bool:
    return str(author_id) in set(brands)
