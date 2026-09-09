"""Build and load the embedding retrieval index."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from resolveflow.cases import cases_checksum


def fingerprint(
    *,
    source_checksum: str,
    embedding_model: str,
    preprocessing_version: str,
    num_cases: int,
) -> str:
    raw = f"{source_checksum}|{embedding_model}|{preprocessing_version}|{num_cases}"
    return hashlib.sha256(raw.encode()).hexdigest()


def query_text(customer_message: str, customer_context: str = "") -> str:
    msg = (customer_message or "").strip()
    ctx = (customer_context or "").strip()
    if not ctx:
        return msg
    return f"{msg}\n{ctx}"


def embed_texts(
    texts: list[str],
    model_name: str,
    *,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


class RetrievalIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        case_ids: list[str],
        cases: pd.DataFrame,
        metadata: dict[str, Any],
    ):
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if len(case_ids) != embeddings.shape[0]:
            raise ValueError("case_ids length must match embeddings rows")
        self.embeddings = embeddings.astype(np.float32)
        # Ensure L2-normalized for cosine via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms
        self.case_ids = list(case_ids)
        self.cases = cases.set_index("case_id", drop=False)
        self.metadata = metadata
        self._id_to_row = {cid: i for i, cid in enumerate(self.case_ids)}

    @property
    def conversation_ids(self) -> set[str]:
        return set(self.cases["conversation_id"].astype(str))

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        sims = self.embeddings @ q
        if top_k <= 0:
            return []
        # partial top-k
        if top_k >= len(sims):
            idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, top_k)[:top_k]
            idx = part[np.argsort(-sims[part])]
        out: list[tuple[str, float]] = []
        for i in idx:
            score = float(sims[i])
            if score < similarity_threshold:
                continue
            out.append((self.case_ids[int(i)], score))
        return out

    def save(
        self,
        *,
        embeddings_path: Path,
        index_path: Path,
        metadata_path: Path,
        case_ids_path: Path,
    ) -> None:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.embeddings)
        # Compact index sidecar (matrix already in npy; store dims + ids)
        np.savez_compressed(
            index_path,
            dim=np.array([self.embeddings.shape[1]], dtype=np.int32),
            n=np.array([self.embeddings.shape[0]], dtype=np.int32),
        )
        case_ids_path.write_text(json.dumps(self.case_ids) + "\n")
        metadata_path.write_text(json.dumps(self.metadata, indent=2) + "\n")

    @classmethod
    def load(
        cls,
        *,
        embeddings_path: Path,
        metadata_path: Path,
        case_ids_path: Path,
        cases_path: Path,
    ) -> "RetrievalIndex":
        emb = np.load(embeddings_path)
        case_ids = json.loads(case_ids_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        cases = pd.read_parquet(cases_path)
        # Keep only indexed cases, ordered
        cases = cases.set_index("case_id").loc[case_ids].reset_index()
        return cls(emb, case_ids, cases, metadata)


def build_index_from_cases(
    cases: pd.DataFrame,
    *,
    embedding_model: str,
    brand: str,
    preprocessing_version: str,
    force_recompute: bool = False,
    cache_embeddings_path: Path | None = None,
    cache_fingerprint: str | None = None,
) -> RetrievalIndex:
    source_checksum = cases_checksum(cases)
    texts = [
        query_text(r.customer_message, r.customer_context)
        for r in cases.itertuples(index=False)
    ]
    fp = fingerprint(
        source_checksum=source_checksum,
        embedding_model=embedding_model,
        preprocessing_version=preprocessing_version,
        num_cases=len(cases),
    )

    emb: np.ndarray | None = None
    if (
        not force_recompute
        and cache_embeddings_path
        and cache_embeddings_path.exists()
        and cache_fingerprint
        and cache_fingerprint == fp
    ):
        emb = np.load(cache_embeddings_path)
        if emb.shape[0] != len(cases):
            emb = None

    if emb is None:
        emb = embed_texts(texts, embedding_model)

    metadata = {
        "brand": brand,
        "embedding_model": embedding_model,
        "embedding_dimension": int(emb.shape[1]),
        "source_checksum": source_checksum,
        "fingerprint": fp,
        "preprocessing_version": preprocessing_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_cases": int(len(cases)),
    }
    return RetrievalIndex(emb, cases["case_id"].astype(str).tolist(), cases, metadata)
