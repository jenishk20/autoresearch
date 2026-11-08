"""Worker VMs on Vultr, booted from the worker snapshot (Docker + gVisor + node agent).

shared mode (default): ONE long-lived VM that every session reuses. It is created on first use (or at app start),
remembered in runs/worker_shared.json so an app restart reconnects instead of creating another, and destroyed
automatically at `keep_until` (local time, e.g. "18:00").

per_session mode: a fresh VM per session, destroyed when the session ends (slow: ~3 min boot each time).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import secrets
import time
from pathlib import Path

from control.nodes import Node
from control.settings import ROOT
from control.vultr import Vultr, my_public_ip

SESSION_PREFIX = "autoresearch-session-"
SHARED_LABEL = "autoresearch-worker-shared"
STATE = ROOT / "runs" / "worker_shared.json"   # holds the node token: local only, git-ignored


class Provisioner:
    def __init__(self, engine):
        self.e = engine
        self.cfg = engine.cfg.get("worker", {"mode": "local"})
        self.mode = self.cfg.get("mode", "local")
        self.lifecycle = self.cfg.get("lifecycle", "shared")
        self.vultr: Vultr | None = Vultr() if self.mode == "vultr" else None
        self.active: dict[str, dict] = {}       # per-session VMs
        self.shared: dict | None = None         # {"id", "ip", "token", "created", "node"}
        self._lock = asyncio.Lock()
        self.status_msg = "not started"

    @property
    def price_per_hour(self) -> float:
        return float(self.cfg.get("price_per_hour", 0.0))

    def info(self) -> dict:
        s = self.shared or {}
        return {"mode": self.mode, "lifecycle": self.lifecycle, "status": self.status_msg, "ip": s.get("ip"),
                "keep_until": self.cfg.get("keep_until"), "up_minutes": round((time.time() - s["created"]) / 60) if s else None}

    # ------------------------------------------------------------------ startup
    async def start(self):
        if self.mode != "vultr":
            return
        for i in await asyncio.to_thread(self.vultr.instances, SESSION_PREFIX):  # leftovers from per-session mode
            await asyncio.to_thread(self.vultr.delete_instance, i["id"])
        if self.lifecycle == "shared":
            asyncio.create_task(self._ensure_shared_logged())
            asyncio.create_task(self._expiry_loop())

    async def _ensure_shared_logged(self):
        try:
            await self.ensure_shared()
        except Exception as ex:
            self.status_msg = f"error: {ex}"

    # ------------------------------------------------------------------ shared VM
    async def ensure_shared(self, sid: str | None = None) -> Node:
        async with self._lock:
            if self.shared and self.shared.get("node") and await self.shared["node"].health():
                return self.shared["node"]
            log = (lambda m, **kw: self.e.log_event(sid, "worker", {"msg": m, **kw})) if sid else (lambda m, **kw: None)
            v, c = self.vultr, self.cfg
            saved = json.loads(STATE.read_text()) if STATE.exists() else None
            live = {i["id"]: i for i in await asyncio.to_thread(v.instances, SHARED_LABEL)}
            if saved and saved["id"] in live:
                self.status_msg = "reconnecting to the running shared VM"
                inst = live[saved["id"]]
                token, created = saved["token"], saved["created"]
                for other in live:   # never keep a second, untracked shared VM around
                    if other != saved["id"]:
                        await self._try_delete(other)
            else:
                for other in live:
                    await self._try_delete(other)
                snap = await asyncio.to_thread(v.latest_snapshot, c["snapshot_prefix"])
                if not snap:
                    raise RuntimeError("no worker snapshot - run scripts/build_worker_snapshot.py")
                fw = await asyncio.to_thread(v.ensure_firewall, c["firewall"], await asyncio.to_thread(my_public_ip))
                token, created = secrets.token_urlsafe(32), time.time()
                self.status_msg = f"booting shared VM ({c['plan']}, {c['region']})"
                log(self.status_msg)
                inst = await asyncio.to_thread(v.create_instance, SHARED_LABEL, c["region"], c["plan"],
                                               snapshot_id=snap["id"], firewall_group_id=fw,
                                               user_data=f"NODE_TOKEN={token}\n")
                STATE.parent.mkdir(parents=True, exist_ok=True)
                STATE.write_text(json.dumps({"id": inst["id"], "token": token, "created": created}))
                STATE.chmod(0o600)
            inst = await asyncio.to_thread(v.wait_active, inst["id"], c.get("boot_timeout_s", 900))
            ip = inst["main_ip"]
            node = Node("vultr-shared", f"http://{ip}:8100", token)
            self.status_msg = f"waiting for the node agent on {ip}"
            t0 = time.time()
            while not await node.health():
                if time.time() - t0 > c.get("boot_timeout_s", 900):
                    raise TimeoutError("node agent on the shared VM never became healthy")
                await asyncio.sleep(4)
            if not node.sandboxed:
                raise RuntimeError("shared VM is not running the docker/gVisor backend")
            if self.shared and self.shared.get("node"):
                self.e.remove_node(self.shared["node"])
            self.shared = {"id": inst["id"], "ip": ip, "token": token, "created": created, "node": node}
            self.e.add_node(node)
            self.status_msg = f"ready at {ip} ({node.slots} gVisor slots)"
            log(f"shared worker VM ready: {node.slots} gVisor slots", ip=ip, instance=inst["id"])
            return node

    async def _try_delete(self, iid: str):
        try:
            await asyncio.to_thread(self.vultr.delete_instance, iid)
        except RuntimeError as ex:   # e.g. 409 while Vultr is still restoring a snapshot; retried next time
            print(f"could not delete {iid} yet: {ex}")

    async def destroy_shared(self, why: str):
        async with self._lock:
            saved = json.loads(STATE.read_text()) if STATE.exists() else None
            ids = {i["id"] for i in await asyncio.to_thread(self.vultr.instances, SHARED_LABEL)}
            for iid in ids:
                await self._try_delete(iid)
            if self.shared and self.shared.get("node"):
                self.e.remove_node(self.shared["node"])
            up = (time.time() - saved["created"]) / 3600 if saved else 0
            self.shared = None
            STATE.unlink(missing_ok=True)
            self.status_msg = f"destroyed ({why}) after {up:.1f} h (~${up * self.price_per_hour:.2f})"
            print(f"shared worker VM destroyed: {why}")

    def _deadline(self) -> dt.datetime | None:
        ku = self.cfg.get("keep_until")
        if not ku:
            return None
        h, m = map(int, str(ku).split(":"))
        return dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)

    async def _expiry_loop(self):
        while True:
            await asyncio.sleep(30)
            dl = self._deadline()
            if dl and dt.datetime.now() >= dl and (self.shared or STATE.exists()):
                await self.destroy_shared(f"keep_until {self.cfg['keep_until']} reached")
                return

    # ------------------------------------------------------------------ session hooks
    async def acquire(self, sid: str) -> Node | None:
        if self.mode != "vultr":
            return None
        if self.lifecycle == "shared":
            dl = self._deadline()
            if dl and dt.datetime.now() >= dl:
                raise RuntimeError(f"the shared worker VM was retired at {self.cfg['keep_until']}")
            return await self.ensure_shared(sid)
        return await self._acquire_per_session(sid)

    async def release(self, sid: str):
        if self.mode != "vultr" or self.lifecycle == "shared":
            return  # shared VM stays up for the next session
        info = self.active.pop(sid, None)
        if not info:
            return
        if info.get("node"):
            self.e.remove_node(info["node"])
        await asyncio.to_thread(self.vultr.delete_instance, info["id"])
        mins = (time.time() - info["created"]) / 60
        self.e.log_event(sid, "worker", {"msg": f"VM destroyed after {mins:.1f} min (~${mins / 60 * self.price_per_hour:.3f})"})

    async def _acquire_per_session(self, sid: str) -> Node:
        v, c = self.vultr, self.cfg
        snap = await asyncio.to_thread(v.latest_snapshot, c["snapshot_prefix"])
        fw = await asyncio.to_thread(v.ensure_firewall, c["firewall"], await asyncio.to_thread(my_public_ip))
        token = secrets.token_urlsafe(32)
        inst = await asyncio.to_thread(v.create_instance, SESSION_PREFIX + sid, c["region"], c["plan"],
                                       snapshot_id=snap["id"], firewall_group_id=fw, user_data=f"NODE_TOKEN={token}\n")
        self.active[sid] = {"id": inst["id"], "created": time.time(), "node": None}
        self.e.log_event(sid, "worker", {"msg": f"creating per-session VM {c['plan']}"})
        ip = (await asyncio.to_thread(v.wait_active, inst["id"], c.get("boot_timeout_s", 900)))["main_ip"]
        node = Node(f"vultr-{sid}", f"http://{ip}:8100", token)
        while not await node.health():
            await asyncio.sleep(4)
        self.active[sid]["node"] = node
        self.e.add_node(node)
        self.e.log_event(sid, "worker", {"msg": f"worker ready: {node.slots} gVisor slots", "ip": ip})
        return node
