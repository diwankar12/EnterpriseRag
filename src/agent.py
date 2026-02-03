import json
import logging
import gc
import time
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, validator

from .llm import LLM
from .vector_store_interface import VectorStoreInterface, DocumentType
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.squad_aware_hybrid_retriever import SquadAwareHybridRetriever

logger = logging.getLogger(__name__)

SYSTEM_JSON_SPEC = """
You are an expert Product Owner & Tech Lead. Your primary goal is to create production-ready Jira stories.
You must always output valid JSON with keys:
- title
- description
- acceptance_criteria
- subtasks
No extra keys.
"""


# ------------------------------------------------------------
# Story Model
# ------------------------------------------------------------

class StoryDraft(BaseModel):
    title: str = Field(..., min_length=3)
    description: str = Field(..., min_length=10)
    acceptance_criteria: List[str] = Field(default_factory=list)
    subtasks: List[str] = Field(default_factory=list)

    @validator("acceptance_criteria", "subtasks", each_item=True)
    def non_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("list item must not be empty")
        return v


# ------------------------------------------------------------
# Agentic RAG
# ------------------------------------------------------------

class AgenticRAG:

    def __init__(
        self,
        llm: LLM,
        store: VectorStoreInterface,
        config,
        all_knowledge_chunks=None,
        all_code_chunks=None,
        embed_query_fn=None,
        cross_encoder_model_path=None,
    ):
        self.llm = llm
        self.store = store
        self.logger = logging.getLogger(__name__)

        # Cache
        self._query_cache = {}
        self._cache_access_order = []
        self._cache_max_size = 8
        self._memory_cleanup_counter = 0

        if all_knowledge_chunks is None or embed_query_fn is None:
            raise ValueError("Knowledge retriever requires chunks and embed_query_fn")

        self.knowledge_retriever = SquadAwareHybridRetriever(
            vector_store=store,
            doc_type=DocumentType.KNOWLEDGE,
            cross_encoder_model_path=cross_encoder_model_path,
            all_chunks=all_knowledge_chunks,
        )
        self.knowledge_retriever.embed_query = embed_query_fn

        if all_code_chunks:
            self.code_retriever = SquadAwareHybridRetriever(
                vector_store=store,
                doc_type=DocumentType.CODE,
                cross_encoder_model_path=cross_encoder_model_path,
                all_chunks=all_code_chunks,
            )
            self.code_retriever.embed_query = embed_query_fn
        else:
            self.code_retriever = None

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    @staticmethod
    def text_truncate(text: str, max_length: int = 20000) -> str:
        return text[:max_length] + "..." if len(text) > max_length else text

    def context_to_text(self, ctx: Dict[str, Any], include_doc_refs: bool = False) -> str:
        if not ctx:
            return "(No context available)"

        parts = []

        if ctx.get("docs"):
            parts.append("## Knowledge Documents\n")
            for i, doc in enumerate(ctx["docs"]):
                content = doc.get("text", "")
                if include_doc_refs:
                    parts.append(f"### Doc {i+1}\n{self.text_truncate(content)}\n")
                else:
                    parts.append(f"{self.text_truncate(content)}\n")

        if ctx.get("code"):
            parts.append("## Code Documents\n")
            for i, doc in enumerate(ctx["code"]):
                content = doc.get("text", "")
                if include_doc_refs:
                    parts.append(f"### Code {i+1}\n{self.text_truncate(content)}\n")
                else:
                    parts.append(f"{self.text_truncate(content)}\n")

        return "\n".join(parts)

    # ------------------------------------------------------------
    # Query preprocessing & intent
    # ------------------------------------------------------------

    def _preprocess_query(self, query: str, tribe_context: str = "") -> str:
        query = query.strip()
        noise = {"please", "can you", "could you", "help me", "i need"}
        words = [w for w in query.lower().split() if w not in noise]
        processed = " ".join(words)

        domain_hints = {
            "mortgage": ["loan", "underwriting"],
            "altfi": ["fintech", "lending"],
            "property_gateway": ["real estate", "gateway"]
        }

        if tribe_context in domain_hints and len(processed.split()) <= 2:
            processed += " " + " ".join(domain_hints[tribe_context])

        return processed

    def _identify_query_intent(self, query: str) -> Dict[str, Any]:
        q = query.lower()

        types = []
        if any(w in q for w in ["how", "what is", "explain"]):
            types.append("explanation")
        if any(w in q for w in ["fix", "error", "issue", "debug"]):
            types.append("troubleshooting")
        if any(w in q for w in ["build", "implement", "create"]):
            types.append("implementation")

        complexity = "simple"
        if len(query.split()) > 10:
            complexity = "high"
        elif len(query.split()) > 5:
            complexity = "medium"

        technical_areas = []
        if "api" in q:
            technical_areas.append("api")
        if "database" in q:
            technical_areas.append("database")

        return {
            "types": types or ["general"],
            "complexity": complexity,
            "technical_areas": technical_areas,
        }

    # ------------------------------------------------------------
    # Query Expansion
    # ------------------------------------------------------------

    def _expand_query(self, query: str, tribe_context: str = "") -> List[str]:
        processed = self._preprocess_query(query, tribe_context)

        fast = self._fast_query_expansion(processed, tribe_context)
        if fast:
            return fast

        try:
            prompt = f"""
Generate 2 related search queries for:
"{processed}"
Context: {tribe_context or "software development"}
Return JSON list.
"""
            response = self.llm.generate(prompt=prompt, temperature=0.3, max_output_tokens=150)
            cleaned = "\n".join(l for l in response.splitlines() if not l.strip().startswith("```"))
            parsed = json.loads(cleaned)
            return [query] + parsed if isinstance(parsed, list) else [query]
        except Exception:
            return [query]

    def _fast_query_expansion(self, query: str, tribe_context: str = "") -> Optional[List[str]]:
        q = query.lower()
        expansions = []

        if "what is" in q or "explain" in q:
            expansions.extend([
                f"{query} definition",
                f"{query} examples"
            ])

        elif "how to" in q:
            base = q.replace("how to", "").strip()
            expansions.extend([
                f"{base} step by step",
                f"{base} best practices"
            ])

        if expansions:
            return [query] + expansions[:2]

        return None

    # ------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------

    def _manage_cache(self, key: str, value: Any = None) -> Any:
        self._memory_cleanup_counter += 1
        if self._memory_cleanup_counter % 10 == 0:
            self._force_memory_cleanup()

        if value is not None:
            if key in self._query_cache:
                self._cache_access_order.remove(key)
            self._query_cache[key] = value
            self._cache_access_order.append(key)

            if len(self._query_cache) > self._cache_max_size:
                oldest = self._cache_access_order.pop(0)
                del self._query_cache[oldest]
            return value

        if key in self._query_cache:
            self._cache_access_order.remove(key)
            self._cache_access_order.append(key)
            return self._query_cache[key]

        return None

    def clear_cache(self):
        self.logger.info("Clearing cache")
        self._query_cache.clear()
        self._cache_access_order.clear()
        gc.collect()

    def _force_memory_cleanup(self):
        if len(self._query_cache) > self._cache_max_size // 2:
            for _ in range(len(self._query_cache) // 2):
                if self._cache_access_order:
                    k = self._cache_access_order.pop(0)
                    self._query_cache.pop(k, None)
        gc.collect()

    # ------------------------------------------------------------
    # Answer Evaluation
    # ------------------------------------------------------------

    def _analyze_answer_completeness(
        self, query: str, answer: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            prompt = f"""
Analyze if the answer addresses the query.

Query: {query}
Answer: {answer}

Return JSON with:
completeness_score, is_complete, missing_aspects, suggested_followups, confidence
"""
            response = self.llm.generate(prompt=prompt, temperature=0.3)
            cleaned = "\n".join(l for l in response.splitlines() if not l.strip().startswith("```"))
            analysis = json.loads(cleaned)

            defaults = {
                "completeness_score": 0.5,
                "is_complete": False,
                "missing_aspects": [],
                "suggested_followups": [],
                "confidence": 0.5,
            }

            for k, v in defaults.items():
                analysis.setdefault(k, v)

            return analysis

        except Exception as e:
            self.logger.error(f"Completeness analysis failed: {e}")
            return {
                "completeness_score": 0.5,
                "is_complete": False,
                "missing_aspects": [],
                "suggested_followups": [],
                "confidence": 0.3,
            }

    def _generate_followup_questions(
        self, query: str, answer: str, missing_aspects: List[str]
    ) -> List[str]:
        if not missing_aspects:
            return []

        try:
            prompt = f"""
Original Query: {query}
Answer: {answer}
Missing Aspects: {', '.join(missing_aspects)}

Generate 2-3 follow-up questions in JSON array.
"""
            response = self.llm.generate(prompt=prompt, temperature=0.4)
            cleaned = "\n".join(l for l in response.splitlines() if not l.strip().startswith("```"))
            followups = json.loads(cleaned)
            return followups[:3] if isinstance(followups, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------
    # Multi-step decision
    # ------------------------------------------------------------

    def _should_attempt_multi_step(self, query: str, intent: Dict[str, Any]) -> bool:
        indicators = ["how to", "step by step", "process", "compare", "pros and cons"]
        q = query.lower()

        return (
            intent.get("complexity") == "high"
            or any(i in q for i in indicators)
            or len(intent.get("technical_areas", [])) > 1
            or len(query.split()) > 10
        )

    # ------------------------------------------------------------
    # Deduplication & Ranking
    # ------------------------------------------------------------

    def _calculate_content_similarity(self, t1: str, t2: str) -> float:
        try:
            w1, w2 = set(t1.lower().split()), set(t2.lower().split())
            if not w1 or not w2:
                return 0.0
            jaccard = len(w1 & w2) / len(w1 | w2)
            len_ratio = min(len(t1), len(t2)) / max(len(t1), len(t2))
            return jaccard * 0.7 + len_ratio * 0.3
        except Exception:
            return 0.0

    def _deduplicate_context(
        self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        deduped = []

        for doc in documents:
            duplicate = False
            for existing in deduped:
                if (
                    self._calculate_content_similarity(
                        doc.get("text", ""), existing.get("text", "")
                    )
                    > similarity_threshold
                ):
                    duplicate = True
                    if doc.get("score", 0) > existing.get("score", 0):
                        existing.update(doc)
                    break
            if not duplicate:
                deduped.append(doc)

        self.logger.info(
            f"Deduplication reduced documents from {len(documents)} to {len(deduped)}"
        )
        return deduped

    def _rank_context_by_priority(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        query_intent: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not documents:
            return documents

        for doc in documents:
            score = doc.get("score", 0.5)
            text = doc.get("text", "").lower()

            for qt in query_intent.get("types", []):
                if qt in ["implementation", "troubleshooting"]:
                    if any(k in text for k in ["function", "class", "error", "exception"]):
                        score += 0.1
                if qt == "explanation":
                    if any(k in text for k in ["overview", "what is", "purpose"]):
                        score += 0.1

            for area in query_intent.get("technical_areas", []):
                if area in text:
                    score += 0.15

            doc["priority_score"] = score

        return sorted(documents, key=lambda d: d["priority_score"], reverse=True)

    def _optimize_context_window(
    self,
    docs: List[Dict[str, Any]],
    code: List[Dict[str, Any]],
    query: str,
    max_tokens: int = 80000
) -> Dict[str, Any]:
        """Optimize context window usage with intelligent selection and ranking"""

        logger.info(
            f"Optimizing context window for {len(docs)} docs and {len(code)} code items"
        )

        try:
            # Analyze query intent
            query_intent = self._identify_query_intent(query)

            # Deduplicate
            deduplicated_docs = self._deduplicate_context(docs, similarity_threshold=0.7)
            deduplicated_code = self._deduplicate_context(code, similarity_threshold=0.8)

            # Rank by priority
            ranked_docs = self._rank_context_by_priority(
                deduplicated_docs, query, query_intent
            )
            ranked_code = self._rank_context_by_priority(
                deduplicated_code, query, query_intent
            )

            selected_docs = []
            selected_code = []

            # Cap tokens for memory efficiency
            max_tokens = min(max_tokens, 15000)
            max_doc_tokens = int(max_tokens * 0.6)
            max_code_tokens = int(max_tokens * 0.4)

            # Select documents
            doc_tokens = 0
            for doc in ranked_docs:
                doc_text = doc.get("text", "")
                doc_token_estimate = len(doc_text) // 4

                if doc_tokens + doc_token_estimate <= max_doc_tokens:
                    selected_docs.append(doc)
                    doc_tokens += doc_token_estimate
                else:
                    if doc.get("priority_score", 0.0) > 0.8 and doc_tokens < max_doc_tokens * 0.8:
                        remaining_tokens = max_doc_tokens - doc_tokens
                        truncated_text = doc_text[: remaining_tokens * 4]
                        truncated_doc = doc.copy()
                        truncated_doc["text"] = truncated_text + "... [truncated]"
                        selected_docs.append(truncated_doc)
                    break

            # Select code
            code_tokens = 0
            for code_item in ranked_code:
                code_text = code_item.get("text", "")
                code_token_estimate = len(code_text) // 4

                if code_tokens + code_token_estimate <= max_code_tokens:
                    selected_code.append(code_item)
                    code_tokens += code_token_estimate
                else:
                    break

            optimized_context = {
                "docs": selected_docs,
                "code": selected_code,
            }

            logger.info(
                f"Context optimization: {len(docs)}->{len(selected_docs)} docs, "
                f"{len(code)}->{len(selected_code)} code items"
            )
            logger.info(
                f"Estimated tokens: {(doc_tokens + code_tokens)}/{max_tokens}"
            )

            return optimized_context

        except Exception as e:
            logger.error(f"Error optimizing context window: {str(e)}")
            return {"docs": docs, "code": code}

    def _retrieve_squad_aware(
    self,
    query: str,
    include_code: bool = False,
    tribe_suffix: str = "",
    selected_squads: List[str] = None,
    k_docs: int = 5,
    k_code: int = 3,
    max_context_tokens: int = 20000,
    status_callback=None,
) -> Dict[str, Any]:
        """
        Squad-aware retrieval for mortgage tribe with multiple squad support.
        """

        from .app_config import tribe_supports_squads

        logger.debug(
            f"[DEBUG] _retrieve_squad_aware called with selected_squads: {selected_squads}"
        )

        current_tribe = tribe_suffix.replace("_", "") if tribe_suffix else "mortgage"
        current_tribe = current_tribe.lower()

        logger.debug(
            f"[DEBUG] Checking tribe support: current_tribe={current_tribe}, "
            f"supports_squads={tribe_supports_squads(current_tribe)}"
        )

        if not tribe_supports_squads(current_tribe):
            return self._retrieve(
                query,
                include_code,
                tribe_suffix,
                k_docs,
                k_code,
                max_context_tokens,
                status_callback,
            )

        if not selected_squads:
            logger.error(
                "No squads selected for squad-aware retrieval - explicit selection required"
            )
            raise ValueError(
                "No squads selected: please choose one or more squads before retrieval"
            )

        logger.info(f"Squad-aware retrieval for squads: {selected_squads}")

        if status_callback:
            status_callback(f"Retrieving from {len(selected_squads)} squads...")

        # Cache key
        squad_key = "-".join(sorted(selected_squads))
        cache_key = hashlib.md5(
            f"{query}_{include_code}_{tribe_suffix}_{squad_key}_{k_docs}_{k_code}".encode()
        ).hexdigest()

        cached_result = self._manage_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached squad-aware result for query: '{query}'")
            if status_callback:
                status_callback("Using cached results...")
            return cached_result

        aggregated_ctx = {"docs": [], "code": []}
        retrieved_doc_ids = set()

        k_docs_per_squad = max(1, k_docs // len(selected_squads))
        k_code_per_squad = max(1, k_code // len(selected_squads))

        for i, squad in enumerate(selected_squads):
            if status_callback:
                status_callback(
                    f"Retrieving from squad {squad} ({i+1}/{len(selected_squads)})..."
                )

            logger.info(f"Retrieving from squad: {squad}")

            try:
                knowledge_docs = self.knowledge_retriever.retrieve(
                    query=query,
                    top_k=k_docs_per_squad,
                    selected_squads=[squad],
                )

                code_docs = []
                if include_code and self.code_retriever:
                    code_docs = self.code_retriever.retrieve(
                        query=query,
                        top_k=k_code_per_squad,
                        selected_squads=[squad],
                    )

                squad_docs = []
                for doc in knowledge_docs:
                    doc_dict = {
                        "text": getattr(doc, "page_content", getattr(doc, "text", str(doc))),
                        "meta": getattr(doc, "metadata", {}),
                        "score": getattr(doc, "score", 0.0),
                    }
                    squad_docs.append(doc_dict)

                squad_code = []
                for doc in code_docs:
                    doc_dict = {
                        "text": getattr(doc, "page_content", getattr(doc, "text", str(doc))),
                        "meta": getattr(doc, "metadata", {}),
                        "score": getattr(doc, "score", 0.0),
                    }
                    squad_code.append(doc_dict)

                for doc in squad_docs:
                    if doc["text"] and doc["text"] not in [
                        d.get("text", "") for d in aggregated_ctx["docs"]
                    ]:
                        aggregated_ctx["docs"].append(doc)

                for doc in squad_code:
                    if doc["text"] and doc["text"] not in [
                        c.get("text", "") for c in aggregated_ctx["code"]
                    ]:
                        aggregated_ctx["code"].append(doc)

            except Exception as e:
                logger.error(f"Failed to retrieve from squad {squad}: {e}")
                logger.info(
                    f"Skipping squad {squad} due to retrieval failure - maintaining isolation"
                )
                continue

        logger.info(
            f"Squad-aware retrieval complete: "
            f"{len(aggregated_ctx['docs'])} docs, "
            f"{len(aggregated_ctx['code'])} code items "
            f"from {len(selected_squads)} squads"
        )

        if status_callback:
            status_callback(
                f"Retrieved {len(aggregated_ctx['docs'])} docs, "
                f"{len(aggregated_ctx['code'])} code items"
            )

        return aggregated_ctx
    
    def _retrieve(
    self,
    query: str,
    include_code: bool = False,
    tribe_suffix: str = "",
    k_docs: int = 5,
    k_code: int = 3,
    max_context_tokens: int = 20_000,
    status_callback=None,
    selected_squads=None,
) -> dict:



        logger.info(f"[DEBUG] retrieve_merged ENTERED with query: {query}")
        logger.info(
            f"[DEBUG] parameters include_code={include_code}, "
            f"k_docs={k_docs}, k_code={k_code}, selected_squads={selected_squads}"
        )

        # -------------------------------
        # Cache key (cache disabled later)
        # -------------------------------
        squads_str = ",".join(sorted(selected_squads)) if selected_squads else "no_squads"
        cache_key_input = f"{query}_{include_code}_{tribe_suffix}_{k_docs}_{k_code}_{squads_str}"
        cache_key = hashlib.md5(cache_key_input.encode()).hexdigest()
        logger.info(f"[DEBUG] Cache key: {cache_key}")

        logger.info("Cache disabled – performing fresh retrieval")

        if status_callback:
            status_callback("Expanding query...")

        # -------------------------------
        # Query expansion
        # -------------------------------
        t0 = time.time()
        try:
            expanded_queries = self._expand_query(query, tribe_suffix)
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            expanded_queries = [query]

        if not expanded_queries:
            expanded_queries = [query]

        logger.info(f"Query expansion took {time.time() - t0:.2f}s")
        logger.info(f"[DEBUG] Expanded queries: {expanded_queries}")

        queries_to_use = expanded_queries[:3]
        if not queries_to_use:
            return {"docs": [], "code": []}

        ctx = {"docs": [], "code": []}
        retrieved_doc_ids = set()
        retrieved_code_ids = set()
        current_tokens = 0

        # -------------------------------
        # Knowledge retrieval
        # -------------------------------
        for i, q in enumerate(queries_to_use):
            logger.info(f"[DEBUG] Knowledge retrieval iteration {i+1}: {q}")

            if status_callback:
                status_callback(f"Retrieving knowledge {i+1}/{len(queries_to_use)}")

            top_k = 40 if i == 0 else 25

            try:
                results = self.knowledge_retriever.retrieve(
                    q,
                    top_k=top_k,
                    rrf_k=80,
                    selected_squads=selected_squads,
                ) or []
            except Exception as e:
                logger.error(f"Knowledge retrieval failed: {e}")
                logger.error(traceback.format_exc())
                results = []

            for doc in results:
                if doc.id in retrieved_doc_ids:
                    continue

                tokens = len(doc.text) // 4
                if current_tokens + tokens > max_context_tokens * 0.7:
                    break

                ctx["docs"].append(
                    {
                        "text": doc.text,
                        "meta": getattr(doc, "metadata", {}),
                        "score": getattr(doc, "score", None),
                    }
                )
                current_tokens += tokens
                retrieved_doc_ids.add(doc.id)

                if len(ctx["docs"]) >= k_docs:
                    break

            if len(ctx["docs"]) >= k_docs:
                break

        # -------------------------------
        # Fallback search
        # -------------------------------
        if not ctx["docs"]:
            logger.warning("Primary retrieval empty – attempting fallback")
            try:
                fb = self._attempt_fallback_search(
                    query, tribe_suffix, k_docs, k_code, include_code, selected_squads
                )
                if fb:
                    ctx = fb
            except Exception as e:
                logger.error(f"Fallback failed: {e}")

        # -------------------------------
        # Code retrieval
        # -------------------------------
        if include_code and self.code_retriever:
            for i, q in enumerate(queries_to_use):
                logger.info(f"[DEBUG] Code retrieval iteration {i+1}: {q}")

                top_k = 15 if i == 0 else 10
                try:
                    code_results = self.code_retriever.retrieve(
                        q,
                        top_k=top_k,
                        rrf_k=30,
                        selected_squads=selected_squads,
                    ) or []
                except Exception as e:
                    logger.error(f"Code retrieval failed: {e}")
                    continue

                for doc in code_results:
                    if doc.id in retrieved_code_ids:
                        continue

                    tokens = len(doc.text) // 4
                    if current_tokens + tokens > max_context_tokens:
                        break

                    ctx["code"].append(
                        {
                            "text": doc.text,
                            "meta": getattr(doc, "metadata", {}),
                            "score": getattr(doc, "score", None),
                        }
                    )
                    current_tokens += tokens
                    retrieved_code_ids.add(doc.id)

                    if len(ctx["code"]) >= k_code:
                        break

                if len(ctx["code"]) >= k_code:
                    break

        # -------------------------------
        # Context window optimization
        # -------------------------------
        if status_callback:
            status_callback("Optimizing context window...")

        try:
            optimized_ctx = self._optimize_context_window(
                ctx.get("docs", []),
                ctx.get("code", []),
                query,
                max_context_tokens,
            )
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            optimized_ctx = ctx

        if not optimized_ctx:
            optimized_ctx = ctx

        # -------------------------------
        # Cleanup
        # -------------------------------
        try:
            del ctx
            del retrieved_doc_ids
            del retrieved_code_ids
        except Exception:
            pass

        logger.info(
            f"[DEBUG] retrieve_merged RETURNING "
            f"{len(optimized_ctx.get('docs', []))} docs, "
            f"{len(optimized_ctx.get('code', []))} code items"
        )

        return optimized_ctx
