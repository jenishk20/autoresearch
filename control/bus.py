"""In-process pub/sub for SSE: one asyncio.Queue per connected browser."""
from __future__ import annotations

import asyncio
from collections import defaultdict


class Bus:
    def __init__(self):
        self.subs: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, session_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self.subs[session_id].add(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue):
        self.subs[session_id].discard(q)

    def publish(self, session_id: str, event: dict):
        for q in list(self.subs.get(session_id, ())):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


bus = Bus()
