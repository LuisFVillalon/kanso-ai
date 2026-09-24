import time
from collections import defaultdict, deque

from fastapi import HTTPException, status


class RateLimiter:
    """Sliding-window limit of `max_calls` per `period` seconds, per key.

    In-memory, so it's per process: enough to blunt abuse of the one or two
    expensive endpoints on a single small Fly machine, not a distributed quota.
    """

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._calls: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        calls = self._calls[key]
        while calls and now - calls[0] > self.period:
            calls.popleft()
        if len(calls) >= self.max_calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please wait a moment and try again.",
                headers={"Retry-After": str(int(self.period))},
            )
        calls.append(now)
