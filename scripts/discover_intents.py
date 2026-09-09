#!/usr/bin/env python3
"""Discover candidate intents via embeddings + KMeans (exploration only)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.data.brand_messages import (  # noqa: E402
    build_brand_customer_messages,
    sample_intent_discovery,
)
from resolveflow.data.ingest import load_raw_tweets  # noqa: E402


def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers required. pip install sentence-transformers"
        ) from exc
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)


def cluster_keywords(texts: list[str], labels: np.ndarray, k: int, top_n: int = 12):
    """Per-cluster TF-IDF keywords for human inspection."""
    out: dict[int, list[str]] = {}
    for c in range(k):
        members = [t for t, lab in zip(texts, labels, strict=False) if lab == c]
        if not members:
            out[c] = []
            continue
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2, stop_words="english")
        try:
            X = vec.fit_transform(members)
        except ValueError:
            out[c] = []
            continue
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        top = terms[scores.argsort()[::-1][:top_n]]
        out[c] = top.tolist()
    return out


def nearest_to_centroid(emb: np.ndarray, labels: np.ndarray, k: int, n: int = 8):
    reps: dict[int, list[int]] = {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[c] = []
            continue
        centroid = emb[idx].mean(axis=0)
        dists = np.linalg.norm(emb[idx] - centroid, axis=1)
        order = dists.argsort()[:n]
        reps[c] = idx[order].tolist()
    return reps


def run_kmeans(emb: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(emb)


def format_cluster_report(
    sample: pd.DataFrame,
    labels: np.ndarray,
    keywords: dict[int, list[str]],
    reps: dict[int, list[int]],
    k: int,
) -> str:
    lines = [f"KMeans k={k}", "=" * 40, ""]
    for c in range(k):
        members = np.where(labels == c)[0]
        lengths = [len(sample.iloc[i]["normalized_text"]) for i in members]
        lines.append(f"Cluster {c}")
        lines.append("-" * 9)
        lines.append(f"Size: {len(members)}")
        lines.append(
            f"Avg length: {np.mean(lengths):.1f}" if lengths else "Avg length: n/a"
        )
        lines.append(f"Keywords: {', '.join(keywords.get(c, []))}")
        lines.append("Representative examples:")
        for i in reps.get(c, [])[:6]:
            text = sample.iloc[i]["normalized_text"][:220].replace("\n", " ")
            lines.append(f'  - "{text}"')
        lines.append("Likely topic: <human: fill in>")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default="AmazonHelp")
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", default="8,10,12,15")
    parser.add_argument(
        "--messages-cache",
        type=Path,
        default=ROOT / "data" / "interim" / "amazonhelp_customer_messages.parquet",
    )
    parser.add_argument(
        "--sample-out",
        type=Path,
        default=ROOT / "data" / "interim" / "intent_discovery_sample.parquet",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "data" / "interim" / "clusters",
    )
    args = parser.parse_args()

    if args.messages_cache.exists():
        print(f"Loading cached messages: {args.messages_cache}")
        messages = pd.read_parquet(args.messages_cache)
    else:
        print("Loading raw tweets and building brand customer messages...")
        raw = load_raw_tweets(ROOT / "data" / "raw" / "twcs.csv")
        messages = build_brand_customer_messages(raw, args.brand)
        args.messages_cache.parent.mkdir(parents=True, exist_ok=True)
        messages.to_parquet(args.messages_cache, index=False)
        print(f"Wrote {args.messages_cache} ({len(messages):,} rows)")

    sample = sample_intent_discovery(messages, n=args.sample_size, seed=args.seed)
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.sample_out, index=False)
    print(f"Discovery sample: {len(sample):,} → {args.sample_out}")

    texts = sample["normalized_text"].astype(str).tolist()
    print("Embedding messages...")
    emb = np.asarray(embed_texts(texts))
    emb_path = ROOT / "data" / "interim" / "intent_discovery_embeddings.npy"
    np.save(emb_path, emb)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    summary = {"brand": args.brand, "sample_size": len(sample), "ks": {}}

    for k in ks:
        print(f"Clustering k={k}...")
        labels = run_kmeans(emb, k=k, seed=args.seed)
        keywords = cluster_keywords(texts, labels, k)
        reps = nearest_to_centroid(emb, labels, k)
        report = format_cluster_report(sample, labels, keywords, reps, k)
        out = args.report_dir / f"kmeans_k{k}.txt"
        out.write_text(report)
        counts = Counter(int(x) for x in labels)
        summary["ks"][str(k)] = {
            "sizes": {str(c): counts[c] for c in range(k)},
            "report": str(out),
        }
        print(f"  wrote {out}")

    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Done. Inspect reports under {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
