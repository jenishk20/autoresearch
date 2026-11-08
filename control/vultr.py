"""Minimal Vultr API v2 client for throwaway worker VMs (uses VULTR_API_KEY, the account key)."""
from __future__ import annotations

import base64
import os
import time

import httpx

API = "https://api.vultr.com/v2"
TAG = "autoresearch"


class Vultr:
    def __init__(self, key: str | None = None):
        key = key or os.environ.get("VULTR_API_KEY", "")
        if not key:
            raise RuntimeError("VULTR_API_KEY is not set")
        self.c = httpx.Client(base_url=API, headers={"Authorization": f"Bearer {key}"}, timeout=60)

    def _req(self, method, path, **kw):
        for attempt in range(4):
            r = self.c.request(method, path, **kw)
            if r.status_code in (429, 503) and attempt < 3:
                time.sleep(2 + 2 * attempt)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Vultr {method} {path} -> {r.status_code}: {r.text[:300]}")
            return r.json() if r.content else {}

    # ---------------------------------------------------------------- instances
    def instances(self, label_prefix: str | None = None) -> list[dict]:
        out = self._req("GET", "/instances", params={"per_page": 500, "tag": TAG}).get("instances", [])
        return [i for i in out if label_prefix is None or i["label"].startswith(label_prefix)]

    def create_instance(self, label: str, region: str, plan: str, *, snapshot_id: str | None = None,
                        os_id: int | None = None, sshkey_ids=(), firewall_group_id: str | None = None,
                        user_data: str | None = None) -> dict:
        body = {"region": region, "plan": plan, "label": label, "hostname": label[:60], "tags": [TAG],
                "backups": "disabled", "enable_ipv6": False, "activation_email": False}
        if snapshot_id:
            body["snapshot_id"] = snapshot_id
        else:
            body["os_id"] = os_id
        if sshkey_ids:
            body["sshkey_id"] = list(sshkey_ids)
        if firewall_group_id:
            body["firewall_group_id"] = firewall_group_id
        if user_data:
            body["user_data"] = base64.b64encode(user_data.encode()).decode()
        return self._req("POST", "/instances", json=body)["instance"]

    def instance(self, iid: str) -> dict:
        return self._req("GET", f"/instances/{iid}")["instance"]

    def delete_instance(self, iid: str):
        self._req("DELETE", f"/instances/{iid}")

    def wait_active(self, iid: str, timeout=900, on_tick=None) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            i = self.instance(iid)
            if on_tick:
                on_tick(i)
            if i["status"] == "active" and i["power_status"] == "running" and i["main_ip"] not in ("", "0.0.0.0"):
                return i
            time.sleep(5)
        raise TimeoutError(f"instance {iid} not active after {timeout}s")

    # ---------------------------------------------------------------- snapshots
    def snapshots(self) -> list[dict]:
        return self._req("GET", "/snapshots", params={"per_page": 500}).get("snapshots", [])

    def create_snapshot(self, iid: str, description: str) -> dict:
        return self._req("POST", "/snapshots", json={"instance_id": iid, "description": description})["snapshot"]

    def snapshot(self, sid: str) -> dict:
        return self._req("GET", f"/snapshots/{sid}")["snapshot"]

    def latest_snapshot(self, prefix: str) -> dict | None:
        snaps = [s for s in self.snapshots() if s["description"].startswith(prefix) and s["status"] == "complete"]
        return max(snaps, key=lambda s: s["date_created"]) if snaps else None

    # ---------------------------------------------------------------- ssh keys / firewall
    def ensure_ssh_key(self, name: str, public_key: str) -> str:
        for k in self._req("GET", "/ssh-keys", params={"per_page": 500}).get("ssh_keys", []):
            if k["name"] == name:
                return k["id"]
        return self._req("POST", "/ssh-keys", json={"name": name, "ssh_key": public_key})["ssh_key"]["id"]

    def ensure_firewall(self, description: str, allow_ip: str, ports=("22", "8100")) -> str:
        """Firewall group allowing only `allow_ip` on `ports`; rules for other IPs are replaced."""
        groups = self._req("GET", "/firewalls", params={"per_page": 500}).get("firewall_groups", [])
        gid = next((g["id"] for g in groups if g["description"] == description), None)
        if gid is None:
            gid = self._req("POST", "/firewalls", json={"description": description})["firewall_group"]["id"]
        rules = self._req("GET", f"/firewalls/{gid}/rules", params={"per_page": 500}).get("firewall_rules", [])
        have = set()
        for r in rules:
            if r["port"] in ("80", "443"):   # public web demo (Caddy) stays open
                continue
            if r["subnet"] != allow_ip:
                self._req("DELETE", f"/firewalls/{gid}/rules/{r['id']}")
            else:
                have.add(r["port"])
        for p in ports:
            if p not in have:
                self._req("POST", f"/firewalls/{gid}/rules", json={"ip_type": "v4", "protocol": "tcp", "subnet": allow_ip,
                                                                  "subnet_size": 32, "port": p, "notes": "autoresearch control"})
        return gid

    def allow_public_web(self, gid: str):
        rules = self._req("GET", f"/firewalls/{gid}/rules", params={"per_page": 500}).get("firewall_rules", [])
        have = {r["port"] for r in rules if r["subnet"] == "0.0.0.0"}
        for p in ("80", "443"):
            if p not in have:
                self._req("POST", f"/firewalls/{gid}/rules", json={"ip_type": "v4", "protocol": "tcp", "subnet": "0.0.0.0",
                                                                  "subnet_size": 0, "port": p, "notes": "public demo"})


def my_public_ip() -> str:
    return httpx.get("https://api.ipify.org", timeout=15).text.strip()
