"""HTTP client for node agents."""
from __future__ import annotations

import os

import httpx


class Node:
    def __init__(self, name: str, url: str, token: str):
        self.name, self.url = name, url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.url, timeout=20,
                                        headers={"Authorization": f"Bearer {token}"} if token else {})
        self.slots, self.running, self.sandboxed, self.backend, self.healthy = 0, 0, False, "?", False

    async def health(self) -> dict:
        try:
            r = await self.client.get("/health")
            r.raise_for_status()
            h = r.json()
            self.slots, self.sandboxed, self.backend, self.healthy = h["slots"], h["sandboxed"], h["backend"], True
            return h
        except Exception:
            self.healthy = False
            return {}

    async def start(self, payload: dict):
        r = await self.client.post("/runs", json=payload)
        r.raise_for_status()

    async def stop(self, run_id):
        await self.client.post(f"/runs/{run_id}/stop")

    async def kill(self, run_id):
        await self.client.post(f"/runs/{run_id}/kill")

    async def status(self, run_id) -> dict:
        r = await self.client.get(f"/runs/{run_id}/status")
        r.raise_for_status()
        return r.json()

    async def metrics(self, run_id, offset) -> dict:
        r = await self.client.get(f"/runs/{run_id}/metrics", params={"offset": offset})
        r.raise_for_status()
        return r.json()

    async def logs(self, run_id, tail=60) -> str:
        try:
            r = await self.client.get(f"/runs/{run_id}/logs", params={"tail": tail})
            return r.json().get("text", "")
        except Exception:
            return ""

    async def file(self, run_id, path) -> bytes | None:
        r = await self.client.get(f"/runs/{run_id}/files/{path}")
        return r.content if r.status_code == 200 else None


def make_nodes(cfg: dict) -> list[Node]:
    return [Node(n["name"], n["url"], os.environ.get(n.get("token_env", "NODE_TOKEN"), "")) for n in cfg["nodes"]]
