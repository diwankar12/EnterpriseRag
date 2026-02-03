"""
ChromaDB implementation of usage metrics storage.

Uses ChromaDB collections with metadata for storing and querying metrics.
"""

import logging
import os
import ssl
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Disable SSL verification globally (as per original code)
# ------------------------------------------------------------------

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

try:
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
except Exception:
    pass

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

import chromadb
from chromadb.config import Settings

from src.services.storage_interface import StorageInterface

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# ChromaDB Metrics Storage
# ------------------------------------------------------------------

class ChromaDBMetricsStorage(StorageInterface):
    """
    ChromaDB implementation of usage metrics storage.
    """

    COLLECTION_NAME = "usage_metrics"

    def __init__(self, persist_path: str = "./chroma_usage_metrics"):
        """Initialize ChromaDB metrics storage."""
        self.persist_path = persist_path
        self.client = None
        self.collection = None
        self._initialize_client()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Extra SSL bypass for httpx/httpcore (used by newer ChromaDB)
            try:
                import httpcore
                original_request = httpcore.SyncHTTPTransport.request

                def patched_request(self, *args, **kwargs):
                    kwargs["verify"] = False
                    return original_request(self, *args, **kwargs)

                httpcore.SyncHTTPTransport.request = patched_request
            except (ImportError, AttributeError):
                pass

            Path(self.persist_path).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    chroma_client_auth_credentials=None,
                    chroma_client_auth_provider=None,
                    chroma_server_ssl_enabled=False,
                ),
            )

            # ----------------------------------------------------------
            # Simple embedding function (metadata-only usage)
            # ----------------------------------------------------------
            from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

            class SimpleEmbeddingFunction(EmbeddingFunction):
                """Simple embedding function that avoids network calls."""
                def __call__(self, input: Documents) -> Embeddings:
                    return [[0.0] * 384 for _ in input]

            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Usage metrics for token tracking"},
                embedding_function=SimpleEmbeddingFunction(),
            )

            logger.info(
                f"ChromaDB metrics storage initialized at {self.persist_path}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize ChromaDB metrics storage: {e}"
            )
            self.client = None
            self.collection = None

    # ------------------------------------------------------------------
    # Availability / Schema
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if ChromaDB storage is available."""
        return self.client is not None and self.collection is not None

    def initialize_schema(self) -> bool:
        """Initialize storage schema (noop for ChromaDB)."""
        return self.is_available()

    # ------------------------------------------------------------------
    # Insert APIs
    # ------------------------------------------------------------------

    def insert_metrics_batch(
        self,
        metrics: List[Dict[str, Any]]
    ) -> bool:
        """Insert multiple metrics in a batch."""
        if not self.is_available() or not metrics:
            return False

        try:
            ids = []
            documents = []
            metadatas = []

            for metric in metrics:
                metric_id = (
                    f"{metric['timestamp'].isoformat()}_"
                    f"{metric.get('session_id', 'nosession')}"
                )

                document = (
                    f"Tribe: {metric.get('tribe_name')} | "
                    f"Request: {metric.get('request_type')} | "
                    f"Tokens: {metric.get('total_tokens')}"
                )

                timestamp_unix = metric["timestamp"].timestamp()

                metadata = {
                    "timestamp": timestamp_unix,
                    "timestamp_iso": metric["timestamp"].isoformat(),
                    "tribe_name": metric["tribe_name"].lower()
                        if metric.get("tribe_name") else "",
                    "squad_name": metric.get("squad_name") or "",
                    "user_id": metric.get("user_id") or "",
                    "request_type": metric.get("request_type"),
                    "model_name": metric.get("model_name"),
                    "input_tokens": int(metric.get("input_tokens", 0)),
                    "output_tokens": int(metric.get("output_tokens", 0)),
                    "total_tokens": int(metric.get("total_tokens", 0)),
                    "duration_ms": (
                        int(metric.get("duration_ms", 0))
                        if metric.get("duration_ms") else 0
                    ),
                    "success": str(metric.get("success", True)),
                }

                ids.append(metric_id)
                documents.append(document)
                metadatas.append(metadata)

            # SSL-safe add
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                original_context = None
                try:
                    original_context = ssl._create_default_https_context
                    ssl._create_default_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass

                try:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                    )
                finally:
                    if original_context:
                        try:
                            ssl._create_default_https_context = original_context
                        except Exception:
                            pass

            logger.debug(f"Inserted {len(metrics)} metrics into ChromaDB")
            return True

        except Exception as e:
            logger.error(
                f"Failed to insert metrics batch into ChromaDB: {e}"
            )
            return False

    def store_metric(self, metric: Dict[str, Any]) -> bool:
        """Store a single metric."""
        return self.insert_metrics_batch([metric])

    # ------------------------------------------------------------------
    # Query APIs
    # ------------------------------------------------------------------

    def query_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        tribe_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query metrics with filters."""
        if not self.is_available():
            return []

        try:
            start_ts = start_date.timestamp()
            end_ts = end_date.timestamp()

            where_filter = {
                "$and": [
                    {"timestamp": {"$gte": start_ts}},
                    {"timestamp": {"$lte": end_ts}},
                ]
            }

            if tribe_name and tribe_name != "All Tribes":
                where_filter["$and"].append(
                    {"tribe_name": tribe_name.lower()}
                )

            results = self.collection.get(
                where=where_filter,
                include=["metadatas"],
            )

            metrics = []
            if results and results.get("metadatas"):
                for metadata in results["metadatas"]:
                    timestamp = datetime.fromtimestamp(metadata["timestamp"])

                    metrics.append({
                        "timestamp": timestamp,
                        "tribe_name": metadata["tribe_name"],
                        "squad_name": metadata.get("squad_name") or None,
                        "user_id": metadata.get("user_id") or None,
                        "request_type": metadata["request_type"],
                        "model_name": metadata["model_name"],
                        "input_tokens": int(metadata["input_tokens"]),
                        "output_tokens": int(metadata["output_tokens"]),
                        "total_tokens": int(metadata["total_tokens"]),
                        "duration_ms": (
                            int(metadata["duration_ms"])
                            if metadata.get("duration_ms") else None
                        ),
                        "success": metadata["success"] == "True",
                    })

            return sorted(metrics, key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to query metrics from ChromaDB: {e}")
            return []

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def get_summary_stats(
        self,
        start_date: datetime,
        end_date: datetime,
        tribe_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated summary statistics."""
        metrics = self.query_metrics(start_date, end_date, tribe_name)

        if not metrics:
            return {
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "avg_duration_ms": 0,
                "unique_users": 0,
                "success_count": 0,
                "error_count": 0,
            }

        total_requests = len(metrics)
        total_input = sum(m["input_tokens"] for m in metrics)
        total_output = sum(m["output_tokens"] for m in metrics)
        total_tokens = sum(m["total_tokens"] for m in metrics)

        durations = [
            m["duration_ms"] for m in metrics
            if m.get("duration_ms") is not None
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        unique_users = len(
            set(m["user_id"] for m in metrics if m.get("user_id"))
        )
        success_count = sum(
            1 for m in metrics if m.get("success", True)
        )

        return {
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "avg_duration_ms": avg_duration,
            "unique_users": unique_users,
            "success_count": success_count,
            "error_count": total_requests - success_count,
        }

    def get_daily_aggregates(
        self,
        start_date: datetime,
        end_date: datetime,
        tribe_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get daily aggregated metrics."""
        metrics = self.query_metrics(start_date, end_date, tribe_name)

        if not metrics:
            return []

        daily_data = {}

        for metric in metrics:
            date_str = metric["timestamp"].strftime("%Y-%m-%d")

            if date_str not in daily_data:
                daily_data[date_str] = {
                    "date": date_str,
                    "requests": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "unique_users": set(),
                }

            d = daily_data[date_str]
            d["requests"] += 1
            d["total_tokens"] += metric["total_tokens"]
            d["input_tokens"] += metric["input_tokens"]
            d["output_tokens"] += metric["output_tokens"]

            if metric.get("user_id"):
                d["unique_users"].add(metric["user_id"])

        result = []
        for date_str in sorted(daily_data.keys()):
            data = daily_data[date_str]
            result.append({
                "date": data["date"],
                "requests": data["requests"],
                "total_tokens": data["total_tokens"],
                "input_tokens": data["input_tokens"],
                "output_tokens": data["output_tokens"],
                "unique_users": len(data["unique_users"]),
            })

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_total_count(self) -> int:
        """Get total number of stored metrics."""
        if not self.is_available():
            return 0

        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0
        
    def store_metric(self, metric: Dict[str, Any]) -> bool:
        """
        Store a single metric.
        Required by StorageInterface.
        """
        if not metric:
            return False
        return self.insert_metrics_batch([metric])
    
