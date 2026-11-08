"""One-time (or after worker code changes): build the worker snapshot that every session VM boots from.

  .venv/bin/python scripts/build_worker_snapshot.py

Creates a builder VM, copies the repo (never .env), runs infra/setup_worker.sh (Docker + gVisor + worker image +
node agent service), smoke-tests a PPO job under runsc, snapshots the VM, then destroys the builder.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.settings import settings  # noqa: E402  (loads .env)
from control.vultr import Vultr, my_public_ip  # noqa: E402

KEY = ROOT / "runs" / "ssh" / "id_ed25519"
KNOWN = ROOT / "runs" / "ssh" / "known_hosts"


def ssh(ip, cmd, check=True):
    return subprocess.run(["ssh", "-i", str(KEY), "-o", "StrictHostKeyChecking=accept-new", "-o",
                           f"UserKnownHostsFile={KNOWN}", "-o", "ConnectTimeout=10", f"root@{ip}", cmd], check=check)


def main():
    w = settings()["worker"]
    v = Vultr()
    KEY.parent.mkdir(parents=True, exist_ok=True)
    if not KEY.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "autoresearch-builder", "-f", str(KEY)], check=True)
    key_id = v.ensure_ssh_key("autoresearch-builder", (KEY.with_suffix(".pub")).read_text().strip())
    fw = v.ensure_firewall(w["firewall"], my_public_ip())
    inst = v.create_instance("autoresearch-builder", w["region"], w["plan"], os_id=2284, sshkey_ids=[key_id],
                             firewall_group_id=fw)
    try:
        ip = v.wait_active(inst["id"])["main_ip"]
        for _ in range(40):
            if ssh(ip, "true", check=False).returncode == 0:
                break
            time.sleep(5)
        subprocess.run(["rsync", "-az", "--delete", "-e", f"ssh -i {KEY} -o UserKnownHostsFile={KNOWN}",
                        "--exclude", ".venv", "--exclude", ".venv-node", "--exclude", "runs", "--exclude", ".git", "--exclude", ".env",
                        "--exclude", "__pycache__", f"{ROOT}/", f"root@{ip}:/opt/autoresearch/"], check=True)
        ssh(ip, "bash /opt/autoresearch/infra/setup_worker.sh")
        ssh(ip, "bash /opt/autoresearch/infra/smoke_test.sh")
        ssh(ip, "systemctl stop autoresearch-node || true; sync")   # token file is rewritten at every boot
        snap = v.create_snapshot(inst["id"], f"{w['snapshot_prefix']}-{time.strftime('%Y%m%d-%H%M')}")
        while v.snapshot(snap["id"])["status"] != "complete":
            time.sleep(15)
        print("snapshot ready:", snap["id"])
    finally:
        v.delete_instance(inst["id"])
        print("builder destroyed")


if __name__ == "__main__":
    main()
