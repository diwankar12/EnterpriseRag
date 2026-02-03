"""
Application configuration and initialization module.

This module centralizes all app configuration, service initialization,
and dependency setup for better maintainability.
"""

import logging
import os
import time
import threading
from typing import Dict, Any, Optional, List

import streamlit as st

import config
from .llm import LLM
from .agent import AgenticRAG
from .jira_client import JiraClient
from .vector_store_factory import create_tribe_vector_store
from .confluence_client import ConfluenceClient
from .pdf_processor import PDFProcessor
from .chunker import Chunker
from .feedback_agent import FeedbackAgent
from .vector_store_interface import DocumentType
from .tribe_config import TRIBE_CONFIG
from .app_config import tribe_supports_squads


# ---------------------------------------------------------
# App-level configuration
# ---------------------------------------------------------

app_config = config.AppConfig()
default_vector_store_provider = getattr(
    app_config, "VECTOR_STORE_PROVIDER", "chromadb"
)


# ---------------------------------------------------------
# Application Services Container
# ---------------------------------------------------------

class AppServices:
    """
    Container for all application services and dependencies.
    Implements singleton pattern for efficient resource management.
    """

    def __init__(self):
        self.config = None
        self.llm = None
        self.vector_store = None
        self.agentic_rag = None
        self.jira_client = None
        self.confluence = None
        self.pdf_processor = None
        self.chunker = None
        self.feedback_agent = None
        self.current_tribe = None
        self._initialized = False

        # Reseed scheduling
        self._reseed_lock = threading.Lock()
        self._reseed_timer: Optional[threading.Timer] = None
        self._reseed_pending = False

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    def initialize(self, tribe_name: str) -> bool:
        """
        Initialize all services for the specified tribe.
        """
        try:
            # Initialize config
            self.config = config.AppConfig()

            # Initialize LLM
            self.llm = LLM(
                project=self.config.PROJECT_ID,
                location=self.config.REGION,
                model_name=self.config.GEMINI_MODEL_NAME,
                embed_model=self.config.GEMINI_EMBED_MODEL_NAME,
            )

            # Detect squad selection from Streamlit session (lazy creation)
            selected_squads = None
            try:
                sel = None
                for key in ("squad_selector", "selected_squad", "selected_squads"):
                    sel = st.session_state.get(key)
                    if sel:
                        break

                if sel:
                    if isinstance(sel, (list, tuple)):
                        selected_squads = [str(s).upper() for s in sel]
                    else:
                        selected_squads = [str(sel).upper()]

                    logging.info(
                        f"Detected selected squads in session: {selected_squads}"
                    )
            except Exception:
                selected_squads = None

            # Initialize vector store
            self.vector_store = self._create_vector_store(
                tribe_name, squads=selected_squads
            )
            if not self.vector_store:
                return False

            # -------------------------------------------------
            # Load documents for BM25 & hybrid retriever
            # -------------------------------------------------

            logging.info(
                "Loading chunks from vector store for hybrid retriever..."
            )

            all_knowledge_docs = []
            all_code_docs = []

            def _load_docs_with_optional_squads(doc_type: DocumentType):
                try:
                    max_bm25_docs = int(os.getenv("MAX_BM25_DOCS", "0"))
                    bm25_limit = max_bm25_docs if max_bm25_docs > 0 else None
                except Exception:
                    bm25_limit = None

                docs = []

                if tribe_supports_squads(tribe_name) and selected_squads:
                    for squad in selected_squads:
                        try:
                            filters = {
                                "target_squad_normalized": squad.lower()
                            }
                            batch = self.vector_store.get_documents(
                                doc_type=doc_type,
                                limit=bm25_limit,
                                filters=filters,
                            )
                            if batch:
                                docs.extend(batch)
                                logging.info(
                                    f"Loaded {len(batch)} documents from squad '{squad}'"
                                )
                        except Exception as e:
                            logging.warning(
                                f"Failed loading docs for squad {squad}: {e}"
                            )
                else:
                    try:
                        docs = self.vector_store.get_documents(
                            doc_type=doc_type, limit=bm25_limit
                        )
                        logging.info(
                            f"Loaded {len(docs)} documents for doc_type={doc_type.value}"
                        )
                    except Exception as e:
                        logging.warning(
                            f"Failed loading all docs for {doc_type}: {e}"
                        )

                return docs

            all_knowledge_docs = _load_docs_with_optional_squads(
                DocumentType.KNOWLEDGE
            )
            all_code_docs = _load_docs_with_optional_squads(
                DocumentType.CODE
            )

            all_knowledge_chunks = [
                {"id": d.id, "text": d.text, "meta": d.metadata}
                for d in all_knowledge_docs
            ]
            all_code_chunks = [
                {"id": d.id, "text": d.text, "meta": d.metadata}
                for d in all_code_docs
            ]

            logging.info(
                f"Loaded {len(all_knowledge_chunks)} knowledge chunks "
                f"and {len(all_code_chunks)} code chunks"
            )

            # Cross-encoder path
            cross_encoder_model_path = os.getenv(
                "CROSS_ENCODER_MODEL_PATH",
                "models/cross-encoder-msmarco-MiniLM-L6-v2",
            )

            # Initialize AgenticRAG
            self.agentic_rag = AgenticRAG(
                llm=self.llm,
                store=self.vector_store,
                config=self.config,
                all_knowledge_chunks=all_knowledge_chunks,
                all_code_chunks=all_code_chunks,
                embed_query_fn=self.llm.embed_texts,
                cross_encoder_model_path=cross_encoder_model_path,
            )

            # Initialize other services
            self.jira_client = JiraClient(
                jira_url=self.config.JIRA_API_URL,
                email=self.config.JIRA_EMAIL,
                api_token=self.config.JIRA_API_TOKEN,
                cert_path=self.config.SSL_CERT_PATH,
                config=self.config,
            )

            self.confluence = ConfluenceClient(self.config)
            self.pdf_processor = PDFProcessor(self.config)
            self.chunker = Chunker(llm=self.llm)

            self.feedback_agent = FeedbackAgent(
                model=self.config.GEMINI_MODEL_NAME
            )

            self.current_tribe = tribe_name
            self._initialized = True

            logging.info(
                f"Successfully initialized all services for tribe: {tribe_name}"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to initialize services: {e}")
            return False

    # -----------------------------------------------------
    # Vector Store Creation
    # -----------------------------------------------------

    def _create_vector_store(
        self, tribe_name: str, squads: Optional[List[str]] = None
    ):
        try:
            tribe_config = TRIBE_CONFIG.get(
                tribe_name, TRIBE_CONFIG["mortgage"]
            )
            provider = tribe_config["vector_store_provider"]

            vector_store = create_tribe_vector_store(
                tribe_name=tribe_name,
                store_type_str=provider,
                embedder=self.llm.embed_texts,
                config_dict={
                    "PROJECT_ID": self.config.PROJECT_ID,
                    "REGION": self.config.REGION,
                    "CHROMA_DB_PATH": self.config.CHROMA_DB_PATH,
                    "ALLOYDB_HOST": self.config.ALLOYDB_HOST,
                    "ALLOYDB_PORT": self.config.ALLOYDB_PORT,
                    "ALLOYDB_USER": self.config.ALLOYDB_USER,
                    "ALLOYDB_PASS": self.config.ALLOYDB_PASS,
                },
                llm_instance=self.llm,
            )

            # Verify squad collections if supported
            if vector_store and tribe_supports_squads(tribe_name):
                logging.info(
                    f"Verifying squad collections for tribe: {tribe_name}"
                )
                try:
                    if hasattr(vector_store, "create_squad_collections"):
                        try:
                            created = vector_store.create_squad_collections(
                                squads=squads
                            )
                        except TypeError:
                            created = vector_store.create_squad_collections()

                        if created is False:
                            logging.warning(
                                "Some squad collections may not have been created"
                            )
                        else:
                            logging.info(
                                "All squad collections verified/created successfully"
                            )
                    else:
                        self._verify_squad_collections(
                            vector_store, tribe_name
                        )
                except Exception as e:
                    logging.error(
                        f"Exception while verifying/creating squad collections: {e}",
                        exc_info=True,
                    )

            return vector_store

        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            return None

    # -----------------------------------------------------
    # Squad Verification (Legacy)
    # -----------------------------------------------------

    def _verify_squad_collections(self, vector_store, tribe_name: str):
        try:
            logging.info(
                f"Verifying squad collections for tribe: {tribe_name}"
            )
            try:
                health = vector_store.health_check()
            except Exception as e:
                logging.warning(
                    f"Transient failure calling health_check(): {e}"
                )
                health = {}

            if not isinstance(health, dict):
                logging.debug(
                    f"Unexpected health_check result: {type(health)}"
                )
                health = {}

            squads = health.get("squad_collections")
            if isinstance(squads, dict):
                logging.info(
                    f"Squad collections status: {len(squads)} squads found"
                )
                for squad, info in squads.items():
                    logging.info(f"  Squad {squad}: {info}")
            else:
                logging.info("No squad collections found in health check")

        except Exception as e:
            logging.warning(
                f"Error verifying squad collections (non-fatal): {e}"
            )
            logging.debug(
                "Full exception while verifying squad collections",
                exc_info=True,
            )

    # -----------------------------------------------------
    # Tribe Config Access
    # -----------------------------------------------------

    def get_tribe_config(self, tribe_name: str) -> Dict[str, Any]:
        return TRIBE_CONFIG.get(tribe_name, TRIBE_CONFIG["mortgage"])

    # -----------------------------------------------------
    # BM25 Reseed Scheduling
    # -----------------------------------------------------

    def schedule_reseed(
        self, debounce_seconds: Optional[int] = None
    ):
        if debounce_seconds is None:
            try:
                debounce_seconds = int(
                    os.getenv("RESEED_DEBOUNCE_SECONDS", "10")
                )
            except Exception:
                debounce_seconds = 10

        if os.getenv("RESEED_ON_UPSERT", "true").lower() not in (
            "1",
            "true",
            "yes",
        ):
            logging.debug("RESEED_ON_UPSERT disabled; skipping reseed")
            return

        with self._reseed_lock:
            try:
                if self._reseed_timer is not None:
                    self._reseed_timer.cancel()
            except Exception:
                pass

            self._reseed_timer = threading.Timer(
                debounce_seconds, self._execute_reseed
            )
            self._reseed_timer.daemon = True
            self._reseed_timer.start()

    def _execute_reseed(self):
        try:
            if self.agentic_rag:
                self.agentic_rag.reseed_bm25()
        except Exception as e:
            logging.warning(f"BM25 reseed failed: {e}")

    import logging
    import os
    import time
    import threading
    from typing import Dict, Any, Optional, List

    import streamlit as st

    from .tribe_config import TRIBE_CONFIG
    from .vector_store_factory import create_tribe_vector_store
    from .vector_store_interface import DocumentType


# -------------------------------------------------------------------
# Reseed helpers
# -------------------------------------------------------------------

    def run_reseed(state):
        """
        Internal runner invoked by the timer.
        Calls immediate_reseed and clears pending flag.
        """
        with state["_reseed_lock"]:
            state["_reseed_pending"] = False
            state["_reseed_timer"] = None

        try:
            logging.info("Background reseed triggered")
            immediate_reseed(state)
        except Exception as e:
            logging.error(f"Background reseed failed: {e}")


    def immediate_reseed(state):
        """
        Perform an immediate reseed of both knowledge and code retrievers.
        """
        try:
            agentic_rag = state.get("agentic_rag")
            if not agentic_rag:
                logging.debug("No agentic_rag available to reseed")
                return

            try:
                kr = getattr(agentic_rag, "knowledge_retriever", None)
                if kr:
                    logging.info("Reseeding knowledge retriever from vector store")
                    seeded = kr.reseed_from_vector_store(
                        limit=int(os.getenv("RESEED_LIMIT", "5000"))
                    )
                    logging.info(
                        f"Knowledge retriever reseed inserted {seeded} chunks"
                    )
            except Exception as e:
                logging.error(f"Failed to reseed knowledge retriever: {e}")

            try:
                cr = getattr(agentic_rag, "code_retriever", None)
                if cr:
                    logging.info("Reseeding code retriever from vector store")
                    seeded_code = cr.reseed_from_vector_store(
                        limit=int(os.getenv("RESEED_LIMIT", "5000"))
                    )
                    logging.info(
                        f"Code retriever reseed inserted {seeded_code} chunks"
                    )
            except Exception as e:
                logging.error(f"Failed to reseed code retriever: {e}")

        except Exception as exc:
            logging.exception(
                f"Unexpected error during immediate_reseed: {exc}"
            )


    def stop_reseed(state):
        """Cancel any scheduled reseed timer."""
        with state["_reseed_lock"]:
            try:
                if state.get("_reseed_timer") is not None:
                    state["_reseed_timer"].cancel()
                    state["_reseed_timer"] = None
                    state["_reseed_pending"] = False
                    logging.info("Cancelled scheduled reseed")
            except Exception:
                logging.debug("Failed to cancel reseed timer")


    def schedule_reseed(state, debounce_seconds: Optional[int] = None):
        """
        Schedule a debounced reseed of BM25.
        """
        if debounce_seconds is None:
            try:
                debounce_seconds = int(
                    os.getenv("RESEED_DEBOUNCE_SECONDS", "10")
                )
            except Exception:
                debounce_seconds = 10

        if os.getenv("RESEED_ON_UPSERT", "true").lower() not in (
            "1",
            "true",
            "yes",
        ):
            logging.debug("RESEED_ON_UPSERT disabled; skipping reseed")
            return

        with state["_reseed_lock"]:
            try:
                if state.get("_reseed_timer") is not None:
                    state["_reseed_timer"].cancel()
            except Exception:
                pass

            timer = threading.Timer(debounce_seconds, run_reseed, args=(state,))
            timer.daemon = True
            state["_reseed_timer"] = timer
            state["_reseed_pending"] = True
            timer.start()


    # -------------------------------------------------------------------
    # Health & state helpers
    # -------------------------------------------------------------------

    def is_initialized(state) -> bool:
        """Check if services are initialized."""
        return state.get("_initialized") and state.get("current_tribe") is not None


    def health_check(state) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_status = {
            "overall": "healthy",
            "services": {},
            "tribe": state.get("current_tribe"),
        }

        try:
            if state.get("vector_store"):
                health_status["services"]["vector_store"] = (
                    state["vector_store"].health_check()
                )

            if state.get("llm"):
                health_status["services"]["llm"] = {
                    "status": "healthy",
                    "model": state["llm"].model_name,
                }

            health_status["services"]["jira"] = {
                "status": "healthy"
                if state.get("jira_client")
                else "not_initialized"
            }
            health_status["services"]["confluence"] = {
                "status": "healthy"
                if state.get("confluence")
                else "not_initialized"
            }
            health_status["services"]["feedback_agent"] = {
                "status": "healthy"
                if state.get("feedback_agent")
                else "not_initialized"
            }

        except Exception as e:
            health_status["overall"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status


    # -------------------------------------------------------------------
    # Tribe / squad helpers
    # -------------------------------------------------------------------

    def tribe_supports_squads(tribe_name: str) -> bool:
        """Check if a tribe supports squad-level selection."""
        tribe_config = TRIBE_CONFIG.get(tribe_name, {})
        return tribe_config.get("supports_squads", False)


    def get_available_squads(tribe_name: str) -> List[str]:
        """Get list of available squads for a tribe."""
        tribe_config = TRIBE_CONFIG.get(tribe_name, {})
        return tribe_config.get("squads", [])


    def get_squad_collection_suffix(
        tribe_name: str, squad: str, collection_type: str
    ) -> str:
        """
        Generate collection suffix for squad-specific collections.
        """
        if not tribe_supports_squads(tribe_name):
            tribe_config = TRIBE_CONFIG.get(tribe_name, {})
            return tribe_config.get("collection_suffix", f"_{tribe_name}")

        return f"_{squad.lower()}"


    def get_squad_collections(
        tribe_name: str, squads: List[str], collection_type: str
    ) -> List[str]:
        """
        Get list of collection names for multiple squads.
        """
        if not tribe_supports_squads(tribe_name) or not squads:
            tribe_config = TRIBE_CONFIG.get(tribe_name, {})
            base_suffix = tribe_config.get("collection_suffix", f"_{tribe_name}")
            return [f"{collection_type}{base_suffix}"]

        collections = []
        for squad in squads:
            collections.append(
                f"{tribe_name.lower()}_{collection_type}_{squad.lower()}"
            )

        return collections


    # -------------------------------------------------------------------
    # Streamlit helpers
    # -------------------------------------------------------------------

    def setup_streamlit_config():
        """Configure Streamlit settings."""
        st.set_page_config(
            page_title="Idea Flow: AI-Powered Story Generation",
            layout="wide",
            initial_sidebar_state="expanded",
        )


    # -------------------------------------------------------------------
    # Services cache (module-level)
    # -------------------------------------------------------------------

    _services_cache = {}


    def get_services(tribe_name: str = None):
        """
        Get the services instance for the specified tribe with isolation.
        """
        if tribe_name is None:
            try:
                tribe_name = (
                    st.session_state.get("tribe_selector_key")
                    or st.session_state.get("tribe_selector")
                )

                if (
                    tribe_name
                    and isinstance(tribe_name, str)
                    and tribe_name.title()
                    in [
                        cfg.get("display_name", "").title()
                        for cfg in TRIBE_CONFIG.values()
                    ]
                ):
                    for key, cfg in TRIBE_CONFIG.items():
                        if (
                            cfg.get("display_name", "").title()
                            == tribe_name.title()
                        ):
                            tribe_name = key
                            break

                if not tribe_name:
                    tribe_name = "mortgage"

            except Exception:
                tribe_name = "mortgage"

        if tribe_name not in _services_cache:
            logging.info(f"Initializing services for tribe: {tribe_name}")
            services = {}
            services["_initialized"] = False
            services["current_tribe"] = tribe_name
            _services_cache[tribe_name] = services

        return _services_cache[tribe_name]


    def initialize_app_services(tribe_name: str):
        """Initialize application services for the specified tribe."""
        return get_services(tribe_name)
