from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class UsageMetricsService:
    """
    Service responsible for querying, aggregating, and summarizing
    LLM usage metrics across tribes and time ranges.
    """

    def __init__(self, storage=None):
        self.storage = storage

    # ------------------------------------------------------------------
    # Summary Metrics
    # ------------------------------------------------------------------
    def get_metrics_summary(
        self,
        tribe_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get summary metrics for the specified period.
        """
        if not self.storage or not self.storage.is_available():
            return {"error": "Storage not available"}

        # Calculate date range
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=days)

        # Get stats from storage
        stats = self.storage.get_summary_stats(
            start_date, end_date, tribe_name
        )

        # Add calculated fields
        if stats.get("total_requests", 0) > 0:
            stats["avg_tokens_per_request"] = (
                stats["total_tokens"] / stats["total_requests"]
            )
        else:
            stats["avg_tokens_per_request"] = 0

        # Calculate success rate
        if stats.get("total_requests", 0) > 0:
            stats["success_rate"] = (
                stats.get("success_count", 0)
                / stats["total_requests"]
            ) * 100
        else:
            stats["success_rate"] = 0

        # Estimated cost (Gemini Flash pricing)
        # $0.15 per 1M input tokens, $0.60 per 1M output tokens
        input_cost = (stats.get("total_input_tokens", 0) / 1_000_000) * 0.15
        output_cost = (stats.get("total_output_tokens", 0) / 1_000_000) * 0.60
        stats["estimated_cost_usd"] = input_cost + output_cost

        # Add date range info
        stats["date_range"] = {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
        }

        return stats

    # ------------------------------------------------------------------
    # Daily Usage
    # ------------------------------------------------------------------
    def get_daily_usage(
        self,
        tribe_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get daily usage aggregates.
        """
        if not self.storage or not self.storage.is_available():
            return []

        # Calculate date range
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=days)

        return self.storage.get_daily_aggregates(
            start_date, end_date, tribe_name
        )

    # ------------------------------------------------------------------
    # Tribe Distribution
    # ------------------------------------------------------------------
    def get_tribe_distribution(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get usage distribution by tribe.
        """
        if not self.storage or not self.storage.is_available():
            return []

        # Calculate date range
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=days)

        # Query all metrics
        metrics = self.storage.query_metrics(
            start_date, end_date, tribe_name=None
        )

        # Aggregate by tribe
        tribe_data: Dict[str, Dict[str, Any]] = {}

        for metric in metrics:
            tribe = metric.get("tribe_name")
            if not tribe:
                continue

            if tribe not in tribe_data:
                tribe_data[tribe] = {
                    "tribe_name": tribe,
                    "requests": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "unique_users": set(),
                }

            tribe_data[tribe]["requests"] += 1
            tribe_data[tribe]["total_tokens"] += metric.get(
                "total_tokens", 0
            )
            tribe_data[tribe]["input_tokens"] += metric.get(
                "input_tokens", 0
            )
            tribe_data[tribe]["output_tokens"] += metric.get(
                "output_tokens", 0
            )

            if metric.get("user_id"):
                tribe_data[tribe]["unique_users"].add(
                    metric["user_id"]
                )

        # Convert to list and finalize unique user counts
        result = []
        for _, data in tribe_data.items():
            result.append(
                {
                    "tribe_name": data["tribe_name"],
                    "requests": data["requests"],
                    "total_tokens": data["total_tokens"],
                    "input_tokens": data["input_tokens"],
                    "output_tokens": data["output_tokens"],
                    "unique_users": len(data["unique_users"]),
                }
            )

        return sorted(
            result, key=lambda x: x["total_tokens"], reverse=True
        )


# ----------------------------------------------------------------------
# Singleton instance
# ----------------------------------------------------------------------
usage_metrics_service = UsageMetricsService()
