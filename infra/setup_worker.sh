#!/usr/bin/env bash
# One-time setup of a worker VM (Ubuntu 24.04). Expects the repo at /opt/autoresearch.
# After this, snapshot the VM; sessions create workers from that snapshot.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg python3-venv
command -v docker || curl -fsSL https://get.docker.com | sh
curl -fsSL https://gvisor.dev/archive.key | gpg --dearmor --yes -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" \
  > /etc/apt/sources.list.d/gvisor.list
apt-get update && apt-get install -y runsc
runsc install
systemctl restart docker
docker run --rm --runtime=runsc hello-world

cd /opt/autoresearch
docker build -f worker/Dockerfile -t autoresearch-worker:latest .

python3 -m venv /opt/autoresearch/.venv-node
/opt/autoresearch/.venv-node/bin/pip install -q fastapi "uvicorn[standard]" "pydantic>=2"
mkdir -p /srv/runs /srv/datasets /srv/sealed /etc/autoresearch

install -m 755 infra/node-token.sh /usr/local/bin/autoresearch-node-token
install -m 644 infra/autoresearch-node.service /etc/systemd/system/autoresearch-node.service
systemctl daemon-reload
systemctl enable autoresearch-node.service
command -v ufw && ufw allow 8100/tcp || true   # Vultr firewall group restricts the source IP
echo SETUP_DONE
