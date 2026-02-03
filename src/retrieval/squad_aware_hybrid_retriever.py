from typing import List, Dict, Any, Optional
import logging

from .hybrid_retriever import HybridRetriever


class SquadAwareHybridRetriever:
    """
    Squad-aware Hybrid Retriever.

    Supports:
    - Squad-level retrieval
    - Tribe-level retrieval (no squads)
    - Metadata-scan fallback
    - All-documents fallback
    - Extensive diagnostics for runtime debugging
    """

    def __init__(
        self,
        vector_store,
        doc_type: str,
        cross_encoder_model_path: Optional[str] = None,
        all_chunks: Optional[List[Dict[str, Any]]] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.vector_store = vector_store
        self.doc_type = doc_type
        self.cross_encoder_model_path = cross_encoder_model_path

        self.all_chunks_seed = all_chunks or []

        self._squad_retrievers: Dict[str, HybridRetriever] = {}
        self._all_documents_retriever: Optional[HybridRetriever] = None
        self._tribe_level_retriever: Optional[HybridRetriever] = None

    # ============================================================
    # Tribe-level Retriever
    # ============================================================

    def _get_tribe_level_retriever(self) -> Optional[HybridRetriever]:
        """
        Build a HybridRetriever that searches the tribe-level collection
        (no squad filtering).
        """
        if getattr(self, "_tribe_level_retriever", None):
            return self._tribe_level_retriever

        self.logger.info("Building tribe-level HybridRetriever (no squad filtering)")

        all_chunks = list(self.all_chunks_seed) if self.all_chunks_seed else []

        try:
            tribe_qr = None
            if hasattr(self.vector_store, "query_documents"):
                try:
                    tribe_qr = self.vector_store.query_documents(
                        doc_type=self.doc_type,
                        query="",
                        k=2000,
                    )
                except Exception as qerr:
                    self.logger.debug(
                        f"tribe-level query_documents probe failed: {qerr}"
                    )

            if not all_chunks:
                if tribe_qr and getattr(tribe_qr, "documents", None):
                    docs = tribe_qr.documents
                else:
                    try:
                        docs = self.vector_store.get_documents(
                            doc_type=self.doc_type,
                            limit=2000,
                        )
                    except Exception as gerr:
                        self.logger.debug(
                            f"tribe-level get_documents probe failed: {gerr}"
                        )
                        docs = []

                for d in docs or []:
                    try:
                        text = d.text if hasattr(d, "text") else getattr(
                            d, "page_content", ""
                        )
                        meta = d.metadata if getattr(d, "metadata", None) else {}
                        all_chunks.append(
                            {
                                "id": getattr(d, "id", None)
                                or f"tribe_doc_{len(all_chunks)}",
                                "text": text,
                                "meta": meta,
                            }
                        )
                    except Exception:
                        continue

            if not all_chunks:
                self.logger.warning(
                    "No tribe-level documents found when building tribe-level retriever"
                )
                return None

            def tribe_vector_search(query: str, top_k: int):
                try:
                    qr = self.vector_store.query_documents(
                        doc_type=self.doc_type,
                        query=query,
                        k=top_k,
                    )
                    return getattr(qr, "documents", []) or []
                except Exception as e:
                    self.logger.debug(f"tribe_vector_search failed: {e}")
                    return []

            self._tribe_level_retriever = HybridRetriever(
                vector_search=tribe_vector_search,
                all_chunks=all_chunks,
                cross_encoder_model_path=self.cross_encoder_model_path,
            )

            self.logger.info(
                f"Built tribe-level HybridRetriever with {len(all_chunks)} chunks"
            )
            return self._tribe_level_retriever

        except Exception as e:
            self.logger.error(f"Failed to build tribe-level retriever: {e}")
            return None

    # ============================================================
    # Metadata-scan fallback retriever
    # ============================================================

    def _create_fallback_retriever_from_metadata(
        self, squads: List[str]
    ) -> Optional[HybridRetriever]:
        """
        Create minimal fallback retriever by scanning all documents
        and filtering by target_squad metadata.
        """
        self.logger.warning(
            f"Attempting metadata-scan fallback retriever for squads: {squads}"
        )

        try:
            all_docs = self.vector_store.get_documents(
                doc_type=self.doc_type, limit=10000
            )
        except Exception as e:
            self.logger.error(f"Metadata-scan fallback failed to get documents: {e}")
            return None

        squad_chunks = []

        for doc in all_docs:
            try:
                meta_val = (
                    doc.metadata.get("target_squad")
                    if getattr(doc, "metadata", None)
                    else None
                )
            except Exception:
                meta_val = None

            if meta_val and str(meta_val).lower() in {s.lower() for s in squads}:
                squad_chunks.append(
                    {
                        "id": doc.id,
                        "text": doc.text
                        if hasattr(doc, "text")
                        else doc.page_content,
                        "meta": doc.metadata if hasattr(doc, "metadata") else {},
                    }
                )

        if not squad_chunks:
            self.logger.debug(
                "Metadata-scan fallback found no matching squad documents"
            )
            return None

        def squad_vector_search(query: str, top_k: int):
            all_results = []
            for squad in squads:
                try:
                    if hasattr(self.vector_store, "query_squad_documents"):
                        res = self.vector_store.query_squad_documents(
                            doc_type=self.doc_type,
                            query=query,
                            squad=squad,
                            k=max(1, top_k // len(squads)),
                        )
                        all_results.extend(getattr(res, "documents", []))
                except Exception:
                    continue
            return all_results[:top_k]

        retriever = HybridRetriever(
            vector_search=squad_vector_search,
            all_chunks=squad_chunks,
            cross_encoder_model_path=self.cross_encoder_model_path,
        )

        self.logger.warning(
            f"Created HybridRetriever via metadata-scan fallback with {len(squad_chunks)} chunks"
        )
        return retriever

    # ============================================================
    # Squad-specific Retriever
    # ============================================================

    def _get_squad_retriever(self, squads: List[str]) -> Optional[HybridRetriever]:
        squad_key = "::".join(sorted(squads))
        if squad_key in self._squad_retrievers:
            return self._squad_retrievers[squad_key]

        squad_chunks = []

        for squad in squads:
            try:
                if hasattr(self.vector_store, "query_squad_documents"):
                    res = self.vector_store.query_squad_documents(
                        doc_type=self.doc_type,
                        query="",
                        squad=squad,
                        k=100,
                    )
                    for d in getattr(res, "documents", []) or []:
                        squad_chunks.append(
                            {
                                "id": d.id,
                                "text": d.text
                                if hasattr(d, "text")
                                else d.page_content,
                                "meta": d.metadata if hasattr(d, "metadata") else {},
                            }
                        )
            except Exception:
                continue

        if not squad_chunks and self.all_chunks_seed:
            squad_chunks = list(self.all_chunks_seed)

        def squad_vector_search(query: str, top_k: int):
            all_results = []
            per_squad_k = max(1, top_k // len(squads))
            for squad in squads:
                try:
                    if hasattr(self.vector_store, "query_squad_documents"):
                        r = self.vector_store.query_squad_documents(
                            doc_type=self.doc_type,
                            query=query,
                            squad=squad,
                            k=per_squad_k,
                        )
                        all_results.extend(getattr(r, "documents", []))
                except Exception:
                    continue
            return all_results[:top_k]

        retriever = HybridRetriever(
            vector_search=squad_vector_search,
            all_chunks=squad_chunks,
            cross_encoder_model_path=self.cross_encoder_model_path,
        )

        self._squad_retrievers[squad_key] = retriever
        return retriever

    # ============================================================
    # Public Retrieve API
    # ============================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        selected_squads: Optional[List[str]] = None,
    ):
        self.logger.debug("SquadAwareHybridRetriever.retrieve called")
        self.logger.debug("selected_squads=%s", selected_squads)

        try:
            from ..app_config import tribe_supports_squads
        except Exception:
            try:
                from src.app_config import tribe_supports_squads
            except Exception:
                tribe_supports_squads = None

        try:
            tribe_name = getattr(self.vector_store, "tribe_name", None) or ""
        except Exception:
            tribe_name = ""

        supports = False
        try:
            supports = (
                tribe_supports_squads(tribe_name)
                if tribe_supports_squads
                else False
            )
        except Exception:
            supports = False

        if selected_squads and not supports:
            self.logger.warning(
                f"Selected squads requested for tribe '{tribe_name}' "
                f"which does not support squads; ignoring squads"
            )
            selected_squads = None

        retriever = None

        if selected_squads:
            try:
                retriever = self._get_squad_retriever(selected_squads)
            except Exception as e:
                self.logger.error(
                    f"Failed to create squad retriever for {selected_squads}: {e}"
                )

        if retriever is None and selected_squads:
            fb = self._create_fallback_retriever_from_metadata(selected_squads)
            if fb:
                retriever = fb

        if retriever is None:
            self.logger.debug("Attempting tribe-level retrieval")
            retriever = self._get_tribe_level_retriever()

        if retriever is None:
            self.logger.error("No retriever available; returning empty results")
            return []

        self.logger.debug(
            "Calling retriever.retrieve(query=%s, top_k=%s, rrf_k=%s)",
            query,
            top_k,
            rrf_k,
        )
        results = retriever.retrieve(query, top_k=top_k, rrf_k=rrf_k)
        self.logger.debug(
            "Retriever returned %s results", len(results) if results else 0
        )
        return results

    # ============================================================
    # Embed Query Delegation
    # ============================================================

    def embed_query(self, query: str):
        """
        Delegate embed_query to an underlying retriever
        """
        if self._all_documents_retriever:
            return self._all_documents_retriever.embed_query(query)
        if self._squad_retrievers:
            return next(iter(self._squad_retrievers.values())).embed_query(query)

        temp = self._get_tribe_level_retriever()
        if temp:
            return temp.embed_query(query)

        raise RuntimeError("No retriever available for embed_query")
    

    def reseed_from_vector_store(self, limit: int = 5000) -> int:
        """
        Rebuild the internal BM25 seed (`all_chunks_seed`) from the vector store.

        This method is safe to call at runtime after upserts. It will:
        - fetch up to `limit` documents from the underlying vector store for this doc_type
        - replace self.all_chunks_seed with the newly built chunk list
        - clear cached squad/tribe/all-doc retrievers so they will be rebuilt with fresh data

        Returns the number of chunks seeded.
        """
        try:
            self.logger.info(
                f"Reseeding all_chunks_seed from vector_store for doc_type={self.doc_type} "
                f"(limit={limit})"
            )

            docs = []
            try:
                docs = (
                    self.vector_store.get_documents(
                        doc_type=self.doc_type, limit=limit
                    )
                    or []
                )
            except Exception as e:
                self.logger.warning(f"get_documents failed during reseed: {e}")
                try:
                    qr = self.vector_store.query_documents(
                        doc_type=self.doc_type, query="", k=limit
                    )
                    docs = getattr(qr, "documents", []) or []
                except Exception as e2:
                    self.logger.error(
                        f"query_documents also failed during reseed: {e2}"
                    )
                    docs = []

            new_chunks = []
            for d in (docs or []):
                try:
                    text = (
                        d.text
                        if hasattr(d, "text")
                        else getattr(d, "page_content", "")
                    )
                    meta = d.metadata if getattr(d, "metadata", None) else {}
                    new_chunks.append(
                        {
                            "id": getattr(d, "id", None)
                            or f"reseeder_doc_{len(new_chunks)}",
                            "text": text,
                            "meta": meta,
                        }
                    )
                except Exception:
                    continue

            self.all_chunks_seed = new_chunks

            # Invalidate cached retrievers so they rebuild on next request
            self._squad_retrievers = {}
            self._all_documents_retriever = None
            if hasattr(self, "_tribe_level_retriever"):
                try:
                    delattr(self, "_tribe_level_retriever")
                except Exception:
                    self._tribe_level_retriever = None

            self.logger.info(
                f"Reseed completed: seeded {len(new_chunks)} chunks"
            )
            return len(new_chunks)

        except Exception as exc:
            self.logger.exception(
                f"Unexpected error during reseed: {exc}"
            )
            return 0
