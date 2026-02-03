"""
Storage Factory - Factory pattern for creating storage backend instances.
"""

import logging
import os
from typing import Optional

from src.services.storage_interface import StorageInterface
from src.services.chromadb_metrics_storage import ChromaDBMetricsStorage
from src.services.alloydb_metrics_storage import AlloyDBMetricsStorage
from config import AppConfig

logger = logging.getLogger(__name__)


class StorageFactory:
    """
    Factory for creating storage backend instances.
    """

    @staticmethod
    def create_storage(
        storage_type: str = "chromadb",
        **kwargs
    ) -> Optional[StorageInterface]:
        """
        Create a storage backend instance.

        Args:
            storage_type: Type of storage ("chromadb" or "alloydb")
            **kwargs: Storage-specific configuration

        Returns:
            StorageInterface instance or None if creation fails
        """
        try:
            storage_type = storage_type.lower()

            # -------------------------
            # ChromaDB Storage
            # -------------------------
            if storage_type == "chromadb":
                persist_path = kwargs.get("persist_path", "./chroma_usage_metrics")
                storage = ChromaDBMetricsStorage(persist_path=persist_path)

                if storage.is_available():
                    logger.info(f"Created ChromaDB storage at {persist_path}")
                    return storage
                else:
                    logger.warning(
                        "ChromaDB storage not available, but returning instance anyway"
                    )
                    # Return instance even if not immediately available
                    return storage

            # -------------------------
            # AlloyDB Storage
            # -------------------------
            elif storage_type == "alloydb":
                config = kwargs.get("config", AppConfig())
                storage = AlloyDBMetricsStorage(config=config)

                if storage.is_available():
                    logger.info("Created AlloyDB storage")
                    return storage
                else:
                    logger.warning(
                        "AlloyDB storage not available, but returning instance anyway"
                    )
                    # Return instance even if not immediately available
                    return storage

            # -------------------------
            # Unknown Storage
            # -------------------------
            else:
                logger.error(f"Unknown storage type: {storage_type}")
                return None

        except Exception as e:
            logger.error(
                f"Failed to create storage backend: {e}",
                exc_info=True
            )
            return None

    @staticmethod
    def create_default_storage() -> Optional[StorageInterface]:
        """
        Create default storage backend (ChromaDB).

        Returns:
            StorageInterface instance or None if creation fails
        """
        return StorageFactory.create_storage(storage_type="chromadb")


# -------------------------------------------------------------------
# Convenience function for creating metrics storage
# -------------------------------------------------------------------
def create_metrics_storage(
    storage_type: Optional[str] = None
) -> Optional[StorageInterface]:
    """
    Create default metrics storage backend.

    Resolution order if storage_type is None:
    1. METRICS_STORAGE_TYPE (if metrics separate from knowledge base)
    2. VECTOR_STORE_PROVIDER (use same storage as knowledge base)
    3. Defaults to "chromadb"

    Args:
        storage_type: Type of storage ("chromadb" or "alloydb")

    Returns:
        StorageInterface instance or None if creation fails
    """
    # Determine storage type
    if storage_type is None:
        # Use VECTOR_STORE_PROVIDER to select metrics backend
        # Default to "alloydb" for metrics storage
        storage_type = os.getenv("VECTOR_STORE_PROVIDER") or "alloydb"

    storage_type = storage_type.lower()
    logger.info(f"Creating metrics storage backend: {storage_type}")

    if storage_type == "alloydb":
        return StorageFactory.create_storage(storage_type="alloydb")
    else:
        # Default to ChromaDB
        return StorageFactory.create_storage(storage_type="chromadb")
