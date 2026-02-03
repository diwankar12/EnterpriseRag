"""
ChromaDB implementation of the vector store interface.

This module provides a ChromaDB-based implementation that supports
tribe-specific collections for knowledge and code documents.
"""

import hashlib
import time
import logging
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

from .vector_store_interface import (
    VectorStoreInterface,
    VectorStoreConfig,
    DocumentType,
    Document,
    QueryResult,
)


# -------------------------------------------------------------------
# Chroma Embedder Wrapper
# -------------------------------------------------------------------

class ChromaDBEmbedder(embedding_functions.EmbeddingFunction):
    """Wrapper for external embedding function to work with ChromaDB"""

    def __init__(
        self,
        embedder_fn: Callable[[List[str]], List[List[float]]],
        llm_instance=None,
    ):
        self.embedder_fn = embedder_fn
        self.llm_instance = llm_instance

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        return self.embedder_fn(inputs)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using LLM if available, otherwise estimate"""
        if self.llm_instance and hasattr(self.llm_instance, "count_tokens"):
            return self.llm_instance.count_tokens(text)
        # Conservative estimate
        return max(1, len(text) // 2)


# -------------------------------------------------------------------
# Chroma Vector Store
# -------------------------------------------------------------------

class ChromaDBVectorStore(VectorStoreInterface):
    """
    ChromaDB implementation of the vector store interface.
    Supports tribe-specific collections with persistent storage.
    """

    @staticmethod
    def sanitize_metadata(meta):
        """Replace None values with empty strings for ChromaDB compatibility"""

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif obj is None:
                return ""
            return obj

        return sanitize(meta)

    def __init__(
        self,
        config: VectorStoreConfig,
        embedder: Callable[[List[str]], List[List[float]]],
        llm_instance=None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            config: Vector store configuration
            embedder: Function to generate embeddings from text
            llm_instance: Optional LLM instance for token counting
        """
        super().__init__(config, embedder)

        self.logger = logging.getLogger(__name__)
        self.llm_instance = llm_instance

        self._collections_cache: Dict[str, Any] = {}
        self.chroma_embedder = ChromaDBEmbedder(
            embedder_fn=embedder, llm_instance=llm_instance
        )

        self.persist_path = (
            config.persist_path or f"./chroma_data_{config.tribe_name}"
        )
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Ensure collections exist
            try:
                from .app_config import tribe_supports_squads
            except Exception:
                from src.app_config import tribe_supports_squads

            if tribe_supports_squads(self.tribe_name):
                self.create_squad_collections()
            else:
                if not self.create_collection(DocumentType.KNOWLEDGE):
                    self.logger.error(
                        "Failed to create knowledge collection during initialization."
                    )
                if not self.create_collection(DocumentType.CODE):
                    self.logger.error(
                        "Failed to create code collection during initialization."
                    )

            self.logger.info(
                f"ChromaDB initialized with tribe collections for tribe '{self.tribe_name}'"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to initialize ChromaDB: {e}", exc_info=True
            )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _safe_get_or_create_collection(
        self, name: str, metadata: Dict[str, Any]
    ):
        """
        Safely get or create a collection.
        Handles Chroma configuration parse errors by backing up and resetting.
        """
        try:
            return self.client.get_or_create_collection(
                name=name,
                embedding_function=self.chroma_embedder,
                metadata=metadata,
            )

        except Exception as e:
            err_str = str(e)

            if isinstance(e, KeyError) or (
                "_type" in err_str
                or ("CollectionConfiguration" in err_str)
            ):
                import traceback

                tb = traceback.format_exc()
                self.logger.error(
                    f"Chroma config parse error for collection '{name}': {e}\n{tb}"
                )

                # Backup persist directory
                try:
                    backup_path = (
                        f"{self.persist_path}.backup.{int(time.time())}"
                    )
                    self.logger.warning(
                        f"Backing up Chroma persist directory '{self.persist_path}' "
                        f"to '{backup_path}' before reset"
                    )

                    try:
                        shutil.copytree(
                            self.persist_path,
                            backup_path,
                            dirs_exist_ok=False,
                        )
                    except TypeError:
                        if not os.path.exists(backup_path):
                            os.mkdir(backup_path)
                        for root, _, files in os.walk(self.persist_path):
                            rel = os.path.relpath(
                                root, self.persist_path
                            )
                            target_root = (
                                os.path.join(backup_path, rel)
                                if rel != "."
                                else backup_path
                            )
                            os.makedirs(
                                target_root, exist_ok=True
                            )
                            for f in files:
                                shutil.copy2(
                                    os.path.join(root, f),
                                    os.path.join(target_root, f),
                                )

                except Exception as backup_err:
                    self.logger.error(
                        f"Failed to backup Chroma persist directory before reset: {backup_err}"
                    )

                # Reset or recreate client
                try:
                    if hasattr(self.client, "reset"):
                        self.logger.warning(
                            "Calling client.reset() to recover from bad configuration"
                        )
                        self.client.reset()
                    else:
                        self.logger.warning(
                            "Recreating PersistentClient to recover from bad configuration"
                        )
                        self.client = chromadb.PersistentClient(
                            path=self.persist_path,
                            settings=Settings(
                                anonymized_telemetry=False,
                                allow_reset=True,
                            ),
                        )
                except Exception as reset_err:
                    self.logger.error(
                        f"Failed to reset/recreate Chroma client: {reset_err}"
                    )
                    raise

                # Retry once
                try:
                    collection = self.client.get_or_create_collection(
                        name=name,
                        embedding_function=self.chroma_embedder,
                        metadata=metadata,
                    )
                    self._collections_cache[name] = collection
                    self.logger.info(
                        f"Successfully recreated collection '{name}' after reset"
                    )
                    return collection
                except Exception as retry_exc:
                    self.logger.error(
                        f"Retry after client reset failed for collection '{name}': {retry_exc}",
                        exc_info=True,
                    )
                    raise

            raise

    # -------------------------------------------------------------------
    # Collection creation
    # -------------------------------------------------------------------

    def create_collection(self, doc_type: DocumentType) -> bool:
        """
        Create a tribe-specific collection with HNSW parameters.
        """
        try:
            collection_name = self.get_collection_name(doc_type)

            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.chroma_embedder,
                metadata={
                    "tribe": self.tribe_name,
                    "doc_type": doc_type.value,
                    "created_at": time.time(),
                    "description": f"{doc_type.value} collection for {self.tribe_name} tribe",
                },
            )

            self._collections_cache[collection_name] = collection
            self.logger.info(
                f"Created/Retrieved ChromaDB collection: {collection_name}"
            )
            return True

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            self.logger.error(
                f"Failed to create collection {doc_type.value}: {e}\n{tb}"
            )

            # Defensive reset and retry once
            try:
                if hasattr(self.client, "reset"):
                    self.client.reset()
                else:
                    self.client = chromadb.PersistentClient(
                        path=self.persist_path,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True,
                        ),
                    )
            except Exception as reset_err:
                self.logger.error(
                    f"Failed to reset/recreate Chroma client: {reset_err}"
                )
                return False

            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.chroma_embedder,
                    metadata={
                        "tribe": self.tribe_name,
                        "doc_type": doc_type.value,
                        "created_at": time.time(),
                        "description": f"{doc_type.value} collection for {self.tribe_name} tribe",
                    },
                )
                self._collections_cache[collection_name] = collection
                self.logger.info(
                    f"Successfully recreated ChromaDB collection after reset: {collection_name}"
                )
                return True
            except Exception as retry_exc:
                self.logger.error(
                    f"Retry after reset failed for collection {collection_name}: {retry_exc}",
                    exc_info=True,
                )
                return False

        return False

"""
ChromaDB implementation of the vector store interface.

This module provides a ChromaDB-based implementation that supports
tribe-specific collections for knowledge and code documents.
"""

import hashlib
import time
import logging
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

from .vector_store_interface import (
    VectorStoreInterface,
    VectorStoreConfig,
    DocumentType,
    Document,
    QueryResult,
)


# -------------------------------------------------------------------
# Chroma Embedder Wrapper
# -------------------------------------------------------------------

class ChromaDBEmbedder(embedding_functions.EmbeddingFunction):
    """Wrapper for external embedding function to work with ChromaDB"""

    def __init__(
        self,
        embedder_fn: Callable[[List[str]], List[List[float]]],
        llm_instance=None,
    ):
        self.embedder_fn = embedder_fn
        self.llm_instance = llm_instance

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        return self.embedder_fn(inputs)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using LLM if available, otherwise estimate"""
        if self.llm_instance and hasattr(self.llm_instance, "count_tokens"):
            return self.llm_instance.count_tokens(text)
        # Conservative estimate
        return max(1, len(text) // 2)


# -------------------------------------------------------------------
# Chroma Vector Store
# -------------------------------------------------------------------

class ChromaDBVectorStore(VectorStoreInterface):
    """
    ChromaDB implementation of the vector store interface.
    Supports tribe-specific collections with persistent storage.
    """

    @staticmethod
    def sanitize_metadata(meta):
        """Replace None values with empty strings for ChromaDB compatibility"""

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif obj is None:
                return ""
            return obj

        return sanitize(meta)

    def __init__(
        self,
        config: VectorStoreConfig,
        embedder: Callable[[List[str]], List[List[float]]],
        llm_instance=None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            config: Vector store configuration
            embedder: Function to generate embeddings from text
            llm_instance: Optional LLM instance for token counting
        """
        super().__init__(config, embedder)

        self.logger = logging.getLogger(__name__)
        self.llm_instance = llm_instance

        self._collections_cache: Dict[str, Any] = {}
        self.chroma_embedder = ChromaDBEmbedder(
            embedder_fn=embedder, llm_instance=llm_instance
        )

        self.persist_path = (
            config.persist_path or f"./chroma_data_{config.tribe_name}"
        )
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Ensure collections exist
            try:
                from .app_config import tribe_supports_squads
            except Exception:
                from src.app_config import tribe_supports_squads

            if tribe_supports_squads(self.tribe_name):
                self.create_squad_collections()
            else:
                if not self.create_collection(DocumentType.KNOWLEDGE):
                    self.logger.error(
                        "Failed to create knowledge collection during initialization."
                    )
                if not self.create_collection(DocumentType.CODE):
                    self.logger.error(
                        "Failed to create code collection during initialization."
                    )

            self.logger.info(
                f"ChromaDB initialized with tribe collections for tribe '{self.tribe_name}'"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to initialize ChromaDB: {e}", exc_info=True
            )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _safe_get_or_create_collection(
        self, name: str, metadata: Dict[str, Any]
    ):
        """
        Safely get or create a collection.
        Handles Chroma configuration parse errors by backing up and resetting.
        """
        try:
            return self.client.get_or_create_collection(
                name=name,
                embedding_function=self.chroma_embedder,
                metadata=metadata,
            )

        except Exception as e:
            err_str = str(e)

            if isinstance(e, KeyError) or (
                "_type" in err_str
                or ("CollectionConfiguration" in err_str)
            ):
                import traceback

                tb = traceback.format_exc()
                self.logger.error(
                    f"Chroma config parse error for collection '{name}': {e}\n{tb}"
                )

                # Backup persist directory
                try:
                    backup_path = (
                        f"{self.persist_path}.backup.{int(time.time())}"
                    )
                    self.logger.warning(
                        f"Backing up Chroma persist directory '{self.persist_path}' "
                        f"to '{backup_path}' before reset"
                    )

                    try:
                        shutil.copytree(
                            self.persist_path,
                            backup_path,
                            dirs_exist_ok=False,
                        )
                    except TypeError:
                        if not os.path.exists(backup_path):
                            os.mkdir(backup_path)
                        for root, _, files in os.walk(self.persist_path):
                            rel = os.path.relpath(
                                root, self.persist_path
                            )
                            target_root = (
                                os.path.join(backup_path, rel)
                                if rel != "."
                                else backup_path
                            )
                            os.makedirs(
                                target_root, exist_ok=True
                            )
                            for f in files:
                                shutil.copy2(
                                    os.path.join(root, f),
                                    os.path.join(target_root, f),
                                )

                except Exception as backup_err:
                    self.logger.error(
                        f"Failed to backup Chroma persist directory before reset: {backup_err}"
                    )

                # Reset or recreate client
                try:
                    if hasattr(self.client, "reset"):
                        self.logger.warning(
                            "Calling client.reset() to recover from bad configuration"
                        )
                        self.client.reset()
                    else:
                        self.logger.warning(
                            "Recreating PersistentClient to recover from bad configuration"
                        )
                        self.client = chromadb.PersistentClient(
                            path=self.persist_path,
                            settings=Settings(
                                anonymized_telemetry=False,
                                allow_reset=True,
                            ),
                        )
                except Exception as reset_err:
                    self.logger.error(
                        f"Failed to reset/recreate Chroma client: {reset_err}"
                    )
                    raise

                # Retry once
                try:
                    collection = self.client.get_or_create_collection(
                        name=name,
                        embedding_function=self.chroma_embedder,
                        metadata=metadata,
                    )
                    self._collections_cache[name] = collection
                    self.logger.info(
                        f"Successfully recreated collection '{name}' after reset"
                    )
                    return collection
                except Exception as retry_exc:
                    self.logger.error(
                        f"Retry after client reset failed for collection '{name}': {retry_exc}",
                        exc_info=True,
                    )
                    raise

            raise

    # -------------------------------------------------------------------
    # Collection creation
    # -------------------------------------------------------------------

    def create_collection(self, doc_type: DocumentType) -> bool:
        """
        Create a tribe-specific collection with HNSW parameters.
        """
        try:
            collection_name = self.get_collection_name(doc_type)

            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.chroma_embedder,
                metadata={
                    "tribe": self.tribe_name,
                    "doc_type": doc_type.value,
                    "created_at": time.time(),
                    "description": f"{doc_type.value} collection for {self.tribe_name} tribe",
                },
            )

            self._collections_cache[collection_name] = collection
            self.logger.info(
                f"Created/Retrieved ChromaDB collection: {collection_name}"
            )
            return True

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            self.logger.error(
                f"Failed to create collection {doc_type.value}: {e}\n{tb}"
            )

            # Defensive reset and retry once
            try:
                if hasattr(self.client, "reset"):
                    self.client.reset()
                else:
                    self.client = chromadb.PersistentClient(
                        path=self.persist_path,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True,
                        ),
                    )
            except Exception as reset_err:
                self.logger.error(
                    f"Failed to reset/recreate Chroma client: {reset_err}"
                )
                return False

            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.chroma_embedder,
                    metadata={
                        "tribe": self.tribe_name,
                        "doc_type": doc_type.value,
                        "created_at": time.time(),
                        "description": f"{doc_type.value} collection for {self.tribe_name} tribe",
                    },
                )
                self._collections_cache[collection_name] = collection
                self.logger.info(
                    f"Successfully recreated ChromaDB collection after reset: {collection_name}"
                )
                return True
            except Exception as retry_exc:
                self.logger.error(
                    f"Retry after reset failed for collection {collection_name}: {retry_exc}",
                    exc_info=True,
                )
                return False

        return False
