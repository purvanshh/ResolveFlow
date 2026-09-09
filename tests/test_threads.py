"""Tests for conversation reconstruction."""

from pathlib import Path

from resolveflow.data.ingest import load_raw_tweets
from resolveflow.data.threads import (
    assign_conversation_ids,
    conversation_stats,
    reconstruct_conversations,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_tweets.csv"


def test_conversation_ids_are_deterministic():
    df = load_raw_tweets(FIXTURE)
    a = assign_conversation_ids(df)
    b = assign_conversation_ids(df)
    assert a.equals(b)
    # Payment thread rooted at tweet 14
    row = df[df["tweet_id"] == 10].index[0]
    assert int(a.loc[row]) == 14


def test_messages_ordered_by_timestamp():
    df = load_raw_tweets(FIXTURE)
    convs = reconstruct_conversations(df, brand="AcmeHelp")
    payment = next(c for c in convs if c["conversation_id"] == "14")
    times = [m["timestamp"] for m in payment["messages"] if m["timestamp"]]
    assert times == sorted(times)
    roles = [m["role"] for m in payment["messages"]]
    assert "customer" in roles and "brand" in roles


def test_empty_text_does_not_crash_stats():
    df = load_raw_tweets(FIXTURE)
    convs = reconstruct_conversations(df, brand="AcmeHelp")
    stats = conversation_stats(convs)
    assert stats["conversations"] >= 1
    assert "response_available" in stats
    assert stats["usable_conversations"] >= 1


def test_cycle_does_not_crash():
    import pandas as pd

    cyclic = pd.DataFrame(
        {
            "tweet_id": [1, 2],
            "author_id": ["A", "B"],
            "inbound": [False, True],
            "created_at": pd.to_datetime(
                ["2017-01-01", "2017-01-02"], utc=True
            ),
            "text": ["x", "y"],
            "response_tweet_id": [2, 1],
            "in_response_to_tweet_id": [2.0, 1.0],
        }
    )
    ids = assign_conversation_ids(cyclic)
    assert ids.notna().all()
