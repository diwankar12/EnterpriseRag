"""
Hybrid Retriever for Dense + Sparse Retrieval with Re-ranking

- Dense retrieval: Vector DB (ChromaDB / AlloyDB / pgvector / etc.)
- Sparse retrieval: BM25 (rank_bm25)
- Fusion: Reciprocal Rank Fusion (RRF)
- Optional Re-ranking: Cross-Encoder (Sentence Transformers)

Usage:
    retriever = HybridRetriever(...)
    results = retriever.retrieve(query)
"""

from typing import List, Dict, Any, Optional
import os
import time
import gc
import logging
import importlib
import numpy as np

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional BM25 dependency handling
# ---------------------------------------------------------------------
bm25_spec = importlib.util.find_spec("rank_bm25")
if bm25_spec is not None:
    from rank_bm25 import BM25Okapi  # type: ignore
else:
    class BM25Okapi:  # type: ignore
        def __init__(self, corpus):
            self._size = len(corpus)

        def get_scores(self, tokens):
            return np.zeros(self._size)

    _logger.warning(
        "rank_bm25 not installed; BM25 scores will be zero (sparse retrieval degraded)."
    )


# ---------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------
class HybridRetriever:
    def __init__(
        self,
        vector_search,
        all_chunks: List[Dict],
        cross_encoder_model_path: Optional[str] = None,
        use_metadata: bool = True,
    ):
        """
        Args:
            vector_search: function(query: str, top_k: int) -> List[Document]
            all_chunks: List of dicts with 'text', 'id', and optional 'meta'
            cross_encoder_model_path: Path to cross-encoder model (optional)
            use_metadata: Include metadata during re-ranking
        """
        self.logger = logging.getLogger(__name__)

        self.vector_search = vector_search
        self.all_chunks = all_chunks
        self.use_metadata = use_metadata

        # BM25 initialization
        if all_chunks and len(all_chunks) > 0:
            self.bm25 = BM25Okapi([chunk["text"].split() for chunk in all_chunks])
        else:
            self.logger.warning(
                "No chunks provided to HybridRetriever; BM25 will be disabled."
            )

            class DummyBM25:
                def get_scores(self, tokens):
                    return np.zeros(0)

            self.bm25 = DummyBM25()

        # Cross-encoder (lazy loaded)
        self.cross_encoder_model_path = cross_encoder_model_path
        self.cross_encoder = None
        self.cross_encoder_tokenizer = None
        self.cross_encoder_loaded = False

        if not cross_encoder_model_path:
            self.logger.info(
                "No cross_encoder_model_path provided. Re-ranking will be disabled."
            )

    # -----------------------------------------------------------------
    # Main retrieval pipeline
    # -----------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 600,
        rrf_k: int = 5000,
        rrf_const: int = 60,
    ) -> List[Any]:
        """
        Hybrid pipeline:
        Dense → BM25 → RRF → optional cross-encoder re-ranking
        """

        self.logger.debug(
            f"[Pipeline] HybridRetriever start for query='{query}' (top_k={top_k})"
        )
        t0 = time.time()

        # --------------------------------------------------------------
        # FAST MODE controls
        # --------------------------------------------------------------
        fast_mode = os.getenv("FAST_MODE", "false").lower() in ("1", "true", "yes")

        if fast_mode:
            orig_top_k, orig_rrf_k = top_k, rrf_k
            rrf_k = min(rrf_k, max(200, top_k * 5))
            top_k = min(top_k, 60)
            self.logger.debug(
                f"[FAST_MODE] Adjusted params: top_k {orig_top_k}->{top_k}, "
                f"rrf_k {orig_rrf_k}->{rrf_k}"
            )

        # --------------------------------------------------------------
        # Cap dense fan-out
        # --------------------------------------------------------------
        try:
            max_dense = int(os.getenv("HYBRID_MAX_DENSE", str(rrf_k)))
            dense_k = min(max_dense, rrf_k)
        except Exception:
            dense_k = rrf_k

        # --------------------------------------------------------------
        # 1. Dense retrieval
        # --------------------------------------------------------------
        dense_results = self.vector_search(query, top_k=dense_k)
        self.logger.debug(
            f"[Pipeline] Dense retrieval returned {len(dense_results)} results"
        )

        # Early exit (FAST MODE)
        if fast_mode and dense_results:
            early_thresh = float(os.getenv("EARLY_EXIT_SCORE", "0.0"))
            scores = [getattr(r, "score", None) for r in dense_results if hasattr(r, "score")]
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score >= early_thresh and len(dense_results) >= top_k:
                    self.logger.debug("[FAST_MODE] Early exit after dense stage")
                    return dense_results[:top_k]

        # --------------------------------------------------------------
        # 2. Sparse retrieval (BM25)
        # --------------------------------------------------------------
        bm25_scores = self.bm25.get_scores(query.split())
        sparse_indices = np.argsort(bm25_scores)[::-1][:rrf_k]

        from src.vector_store_interface import Document

        sparse_results = [
            Document(
                id=chunk.get("id", f"bm25_{i}"),
                text=chunk.get("text", ""),
                metadata=chunk.get("meta", {}),
                score=None,
            )
            for i, chunk in enumerate(self.all_chunks)
            if i in sparse_indices
        ]

        self.logger.debug(
            f"[Pipeline] Sparse retrieval returned {len(sparse_results)} results"
        )

        if not dense_results and not sparse_results:
            self.logger.warning("Both dense and sparse retrieval returned no results.")
            return []

        # --------------------------------------------------------------
        # 3. Reciprocal Rank Fusion (RRF)
        # --------------------------------------------------------------
        def rrf_rank(results):
            return {doc.id: rank for rank, doc in enumerate(results)}

        dense_rank = rrf_rank(dense_results)
        sparse_rank = rrf_rank(sparse_results)

        all_ids = set(dense_rank) | set(sparse_rank)
        rrf_scores = {}

        for doc_id in all_ids:
            r1 = dense_rank.get(doc_id, rrf_k)
            r2 = sparse_rank.get(doc_id, rrf_k)
            rrf_scores[doc_id] = (1 / (rrf_const + r1)) + (1 / (rrf_const + r2))

        candidates = sorted(
            [doc for doc in dense_results + sparse_results if doc.id in all_ids],
            key=lambda d: rrf_scores[d.id],
            reverse=True,
        )

        # Deduplicate
        seen = set()
        unique_candidates = []
        for doc in candidates:
            if doc.id not in seen:
                unique_candidates.append(doc)
                seen.add(doc.id)

        self.logger.debug(
            f"[Pipeline] RRF produced {len(unique_candidates)} unique candidates"
        )

        # --------------------------------------------------------------
        # 4. Optional cross-encoder re-ranking
        # --------------------------------------------------------------
        if fast_mode and os.getenv("CROSS_ENCODER_ENABLED", "false").lower() not in (
            "1",
            "true",
            "yes",
        ):
            return unique_candidates[:top_k]

        if not self.cross_encoder_model_path:
            return unique_candidates[:top_k]

        if not self.cross_encoder_loaded:
            try:
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )

                self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(
                    self.cross_encoder_model_path
                )
                self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(
                    self.cross_encoder_model_path
                )
                self.cross_encoder.eval()
                self.cross_encoder_loaded = True
            except Exception as e:
                self.logger.error(f"Failed to load cross-encoder: {e}")
                return unique_candidates[:top_k]

        max_rerank = min(max(top_k * 4, 80), 200)
        rerank_candidates = unique_candidates[:max_rerank]

        pairs = []
        for c in rerank_candidates:
            if self.use_metadata and c.metadata:
                meta = " ".join(f"{k}:{v}" for k, v in c.metadata.items())
                pairs.append((query, f"{c.text} [META] {meta}"))
            else:
                pairs.append((query, c.text))

        if importlib.util.find_spec("torch") is None:
            self.logger.warning("torch not available; skipping cross-encoder")
            return unique_candidates[:top_k]

        import torch

        scores = []
        batch_size = 4

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            inputs = self.cross_encoder_tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            with torch.no_grad():
                logits = self.cross_encoder(**inputs).logits
                batch_scores = logits.squeeze().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)

            del inputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        reranked = sorted(
            zip(rerank_candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        final_results = [doc for doc, _ in reranked[:top_k]]

        self.logger.debug(
            f"[Timing] Full pipeline took {time.time() - t0:.3f}s"
        )

        return final_results

    # -----------------------------------------------------------------
    def embed_query(self, query: str):
        raise NotImplementedError(
            "embed_query must be implemented for your embedding model"
        )
