"""Tests for raw tweet ingestion."""

from pathlib import Path

import pandas as pd
import pytest

from resolveflow.data.ingest import (
    find_duplicate_tweet_ids,
    load_raw_tweets,
    parse_response_ids,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_tweets.csv"


def test_load_parses_timestamps_and_empty_text():
    df = load_raw_tweets(FIXTURE)
    assert "created_at" in df.columns
    # Malformed timestamp becomes NaT, does not crash
    bad = df.loc[df["tweet_id"] == 50, "created_at"].iloc[0]
    assert pd.isna(bad)
    # Empty brand text becomes empty string
    empty = df.loc[df["tweet_id"] == 31, "text"].iloc[0]
    assert empty == ""


def test_duplicate_tweet_ids_detected():
    df = load_raw_tweets(FIXTURE)
    dups = find_duplicate_tweet_ids(df)
    assert 40 in set(dups.index.astype(int))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_raw_tweets("/tmp/does-not-exist-resolveflow.csv")


def test_parse_response_ids_handles_lists_and_junk():
    assert parse_response_ids(None) == []
    assert parse_response_ids(float("nan")) == []
    assert parse_response_ids("11,12") == [11, 12]
    assert parse_response_ids("nope,3") == [3]


def test_malformed_inbound_still_loads():
    # Ensure pipeline tolerates odd inbound values via a tiny frame rewrite
    raw = FIXTURE.read_text()
    # Smoke: loader returns expected columns
    df = load_raw_tweets(FIXTURE)
    assert set(
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ]
    ).issubset(df.columns)
    assert len(df) == len(raw.strip().splitlines()) - 1
