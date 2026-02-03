"""
Usage tracking service for capturing token usage metrics.
Provides singleton tracker with background worker for async storage.
"""

import logging
import threading
import queue
import time
import inspect
import os
import uuid
import hashlib
from typing import Optional
from datetime import datetime

from src.services.storage_factory import create_metrics_storage

logger = logging.getLogger(__name__)


class UsageTracker:
    """
    Tracks token usage and writes to storage asynchronously.
    Implements a singleton pattern with background worker thread.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UsageTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._storage = create_metrics_storage()
        self._queue = queue.Queue(maxsize=1000)

        # Recent-enqueue dedupe store: key -> timestamp
        self._recent = {}
        self._recent_lock = threading.Lock()

        # Window (seconds) to consider duplicate enqueues
        self._dedupe_window = 10

        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True
        )
        self._worker_thread.start()

        self._initialized = True
        logger.info("UsageTracker initialized with background worker")

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------
    def _worker(self):
        """Background worker that processes queue and writes to storage."""
        batch = []
        batch_size = 10
        last_flush = time.time()
        flush_interval = 5

        while not self._stop_event.is_set():
            try:
                try:
                    item = self._queue.get(timeout=1.0)
                    batch.append(item)
                except queue.Empty:
                    pass

                current_time = time.time()
                should_flush = (
                    len(batch) >= batch_size
                    or (batch and current_time - last_flush >= flush_interval)
                )

                if should_flush and batch:
                    try:
                        self._storage.insert_metrics_batch(batch)
                        logger.debug(
                            f"Flushed {len(batch)} metrics to storage"
                        )
                        batch = []
                        last_flush = current_time
                    except Exception as e:
                        logger.error(f"Failed to flush metrics: {e}")
                        batch = []

            except Exception as e:
                logger.error(f"Error in worker thread: {e}")

        # Final flush on shutdown
        if batch:
            try:
                self._storage.insert_metrics_batch(batch)
                logger.info(f"Final flush: {len(batch)} metrics")
            except Exception as e:
                logger.error(f"Failed final flush: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start tracker (no-op since worker auto-starts)."""
        if not self._initialized:
            self.__init__()
        logger.info("UsageTracker started")

    def stop(self):
        """Stop the worker thread and flush remaining metrics."""
        logger.info("Stopping UsageTracker...")
        self._stop_event.set()
        self._worker_thread.join(timeout=10)
        logger.info("UsageTracker stopped")

    # ------------------------------------------------------------------
    # Core Tracking
    # ------------------------------------------------------------------
    def track(
        self,
        tribe_name: str,
        request_type: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        squad_name: Optional[str] = None,
        database_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """Track a usage event."""
        if not self._storage.is_available():
            logger.warning("Storage not available, cannot track usage")
            return

        try:
            metric = {
                "timestamp": datetime.now(),
                "tribe_name": tribe_name,
                "squad_name": squad_name,
                "database_name": database_name,
                "user_id": user_id,
                "session_id": session_id,
                "request_type": request_type,
                "model_name": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "duration_ms": duration_ms,
                "success": success,
                "error_message": error_message,
            }

            # ----------------------------------------------------------
            # Generate deterministic session_id if missing
            # ----------------------------------------------------------
            if not metric.get("session_id"):
                try:
                    ts_sec = int(time.time())
                    fingerprint_src = (
                        f"{metric.get('tribe_name','')}:"
                        f"{metric.get('user_id','')}:"
                        f"{metric.get('request_type','')}:"
                        f"{metric.get('model_name','')}:"
                        f"{metric.get('input_tokens',0)}:"
                        f"{metric.get('output_tokens',0)}:"
                        f"{ts_sec}"
                    )
                    metric["session_id"] = hashlib.sha1(
                        fingerprint_src.encode("utf-8")
                    ).hexdigest()
                except Exception:
                    metric["session_id"] = str(uuid.uuid4())

            logger.debug(
                f"Enqueue metric: request_type={metric.get('request_type')} "
                f"session_id={metric.get('session_id')}"
            )

            # ----------------------------------------------------------
            # Deduplication
            # ----------------------------------------------------------
            dedupe_key = (
                metric.get("session_id"),
                metric.get("request_type"),
                metric.get("tribe_name"),
            )
            now_ts = time.time()

            try:
                with self._recent_lock:
                    prev = self._recent.get(dedupe_key)
                    if prev and (now_ts - prev) < self._dedupe_window:
                        logger.info(
                            f"Dropping duplicate metric within "
                            f"{self._dedupe_window}s window: {dedupe_key}"
                        )
                        return

                    self._recent[dedupe_key] = now_ts

                    # Prune old entries
                    if len(self._recent) > 2000:
                        cutoff = now_ts - (self._dedupe_window * 3)
                        keys_to_del = [
                            k for k, v in self._recent.items() if v < cutoff
                        ]
                        for k in keys_to_del:
                            self._recent.pop(k, None)

            except Exception as e:
                logger.debug(f"Dedupe check failed: {e}")

            self._queue.put_nowait(metric)

        except queue.Full:
            logger.warning("Metrics queue full, dropping metric")
        except Exception as e:
            logger.error(f"Failed to track usage: {e}")

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------


_tracker_instance: Optional[UsageTracker] = None


def get_tracker() -> UsageTracker:
    """Get the singleton UsageTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = UsageTracker()
    return _tracker_instance


# ----------------------------------------------------------------------
# Convenience Wrapper
# ----------------------------------------------------------------------
def track_llm_usage(
    tribe_name: str,
    user_id: str,
    model: str,
    request_type: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int,
    success: bool = True,
    squad_name: Optional[str] = None,
    database_name: Optional[str] = None,
    session_id: Optional[str] = None,
    error_message: Optional[str] = None,
):
    """
    Convenience function to track LLM usage.
    """
    try:
        tracker = get_tracker()

        # Log original caller for debugging duplicate calls
        try:
            outer_caller = "unknown"
            for fr in inspect.stack()[1:10]:
                fname = fr.filename or ""
                if "usage_tracker.py" in os.path.basename(fname):
                    continue
                outer_caller = (
                    f"{os.path.basename(fr.filename)}:"
                    f"{fr.lineno}:{fr.function}"
                )
                break

            logger.debug(
                f"track_llm_usage invoked from {outer_caller} "
                f"request_type={request_type} session_id={session_id}"
            )
        except Exception:
            pass

        tracker.track(
            tribe_name=tribe_name,
            request_type=request_type,
            model_name=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            squad_name=squad_name,
            database_name=database_name,
            user_id=user_id,
            session_id=session_id,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )

    except Exception as e:
        logger.debug(f"Non-critical: Failed to track usage: {e}")
