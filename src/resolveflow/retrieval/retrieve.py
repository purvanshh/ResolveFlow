"""Retrieve similar historical cases for a query."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from resolveflow.retrieval.index import RetrievalIndex, embed_texts, query_text
from resolveflow.retrieval import RetrievedCase


class Retriever:
    def __init__(self, index: RetrievalIndex, embedding_model: str):
        self.index = index
        self.embedding_model = embedding_model
        self._embed_one = _make_embedder(embedding_model)

    def retrieve(
        self,
        query: str,
        *,
        context: str = "",
        top_k: int = 5,
        similarity_threshold: float = 0.45,
    ) -> list[RetrievedCase]:
        text = query_text(query, context)
        q = self._embed_one(text)
        hits = self.index.search(
            q, top_k=top_k, similarity_threshold=similarity_threshold
        )
        out: list[RetrievedCase] = []
        for case_id, score in hits:
            row = self.index.cases.loc[case_id]
            out.append(
                RetrievedCase(
                    case_id=str(case_id),
                    similarity=float(score),
                    customer_message=str(row["customer_message"]),
                    brand_response=str(row["brand_response"]),
                    intent=str(row.get("intent") or ""),
                    resolution_summary=str(row.get("resolution_summary") or ""),
                    customer_context=str(row.get("customer_context") or ""),
                    conversation_id=str(row.get("conversation_id") or ""),
                )
            )
        return out


def _make_embedder(model_name: str) -> Callable[[str], np.ndarray]:
    # Lazy singleton per model name
    @lru_cache(maxsize=1)
    def _model():
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    def embed_one(text: str) -> np.ndarray:
        model = _model()
        vec = model.encode([text], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)

    return embed_one


def load_retriever(
    *,
    embeddings_path: Path,
    metadata_path: Path,
    case_ids_path: Path,
    cases_path: Path,
) -> Retriever:
    index = RetrievalIndex.load(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        case_ids_path=case_ids_path,
        cases_path=cases_path,
    )
    model = index.metadata["embedding_model"]
    return Retriever(index, model)
