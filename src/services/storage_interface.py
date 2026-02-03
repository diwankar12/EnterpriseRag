"""
Storage Interface for Usage Metrics.

Defines the contract for different storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class StorageInterface(ABC):
    """
    Abstract base class for usage metrics storage.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if storage is available and properly initialized.

        Returns:
            bool: True if storage is available, False otherwise
        """
        pass

    @abstractmethod
    def store_metric(self, metric: Dict[str, Any]) -> bool:
        """
        Store a single metric.

        Args:
            metric: Dictionary containing metric data

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def query_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tribe_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query metrics with optional filters.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            tribe_name: Filter by tribe name
            limit: Maximum number of results

        Returns:
            List of metric dictionaries
        """
        pass

    @abstractmethod
    def get_total_count(self) -> int:
        """
        Get total number of metrics stored.

        Returns:
            int: Total metric count
        """
        pass
