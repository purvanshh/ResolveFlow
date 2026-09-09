"""Tests for profiling helpers."""

from pathlib import Path

from resolveflow.data.ingest import load_raw_tweets
from resolveflow.data.profile import (
    compute_brand_table,
    format_profile_report,
    profile_dataframe,
    select_brand,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_tweets.csv"


def test_profile_includes_core_fields():
    df = load_raw_tweets(FIXTURE)
    profile = profile_dataframe(df)
    assert profile["rows"] == len(df)
    assert "tweet_id" in profile["column_names"]
    assert profile["unique_brands"] >= 1
    assert "inbound_distribution" in profile
    assert profile["duplicate_tweet_ids"] >= 1


def test_brand_table_and_selection():
    df = load_raw_tweets(FIXTURE)
    metrics = compute_brand_table(df, top_n=10, min_brand_replies=1)
    assert metrics
    brands = {m.brand for m in metrics}
    assert "AcmeHelp" in brands
    selected = select_brand(metrics)
    assert selected is not None
    report = format_profile_report(
        profile_dataframe(df), metrics, selected, top_k=5
    )
    assert "ResolveFlow Data Profile" in report
    assert selected.brand in report
