"""Shared concurrency and cancellation helpers for local runtime coordination."""
from __future__ import annotations

import time


class CombinedEvent:
    """Observe two or more event-like cancellation tokens as a single waitable flag."""

    def __init__(self, *events):
        self.events = tuple(event for event in events if event is not None)
        if not self.events:
            raise ValueError('CombinedEvent requires at least one event')
        self.first = self.events[0]
        self.second = self.events[1] if len(self.events) > 1 else self.events[0]

    def is_set(self):
        """Return True when any observed cancellation token is set."""
        return any(event.is_set() for event in self.events)

    def wait(self, timeout=None):
        """Poll until any observed event is set or timeout expires."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            self.first.wait(0.05 if remaining is None else min(0.05, remaining))
        return True
