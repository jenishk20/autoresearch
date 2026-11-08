#!/usr/bin/env bash
# Runs at boot: the per-session node token arrives as instance user-data ("NODE_TOKEN=...").
# A snapshot clone therefore never reuses an old token. No token -> the node agent refuses to start.
set -u
for url in http://169.254.169.254/latest/user-data http://169.254.169.254/user-data; do
  data=$(curl -sf --max-time 5 "$url") && break
done
tok=$(printf '%s\n' "${data:-}" | sed -n 's/^NODE_TOKEN=//p' | head -1 | tr -d '\r')
[ -n "$tok" ] || { echo "no NODE_TOKEN in user-data" >&2; exit 1; }
umask 077
printf 'NODE_TOKEN=%s\n' "$tok" > /etc/autoresearch/node.env
