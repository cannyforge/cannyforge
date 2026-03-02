#!/usr/bin/env python3
"""
CannyForge Workers — background processing for learning cycles.

Moves learning off the hot execution path so execute() returns immediately
while learning runs in a background thread.
"""

import logging
import threading
from queue import Queue, Empty
from typing import Optional, Callable

logger = logging.getLogger("Workers")


class LearningWorker:
    """
    Background worker that processes learning cycle requests.

    When async_learning=True in CannyForge, _maybe_auto_learn() enqueues a
    request instead of calling run_learning_cycle() inline. The worker
    processes the queue in a daemon thread.

    When async_learning=False, the worker is not started and requests are
    processed inline (preserves current behavior for tests/demos).
    """

    def __init__(self, learning_fn: Callable[[], None]):
        """
        Args:
            learning_fn: Callable that runs a learning cycle (no args).
        """
        self._learning_fn = learning_fn
        self._queue: Queue = Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        """Start the background worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="cannyforge-learning-worker",
            )
            self._thread.start()
            logger.info("Learning worker started")

    def stop(self):
        """Stop the background worker (waits for current task to finish)."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        self._queue.put(None)  # sentinel to unblock
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Learning worker stopped")

    def enqueue(self):
        """Enqueue a learning cycle request."""
        self._queue.put("learn")

    @property
    def pending(self) -> int:
        """Number of pending learning requests."""
        return self._queue.qsize()

    def _run_loop(self):
        """Worker loop: process learning requests from the queue."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                break  # sentinel

            try:
                logger.info("Learning worker: running learning cycle")
                self._learning_fn()
            except Exception as e:
                logger.error(f"Learning worker error: {e}")
            finally:
                self._queue.task_done()
