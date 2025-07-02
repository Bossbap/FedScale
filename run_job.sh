#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 PATH/TO/CONFIG.yml"
  exit 1
fi
CONFIG="$1"

echo "❯ Killing any leftover FedScale processes…"
pkill -9 -f "aggregator.py|executor.py|learner.py|fl.py" || true

# only start ssh-agent & add key if no identities are loaded
if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l &>/dev/null; then
  echo "❯ Remember to add an agent before running the script"
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_rsa
else
  echo "❯ ssh-agent is already running with your key loaded; skipping."
fi

# grab worker_ips lines (allow any indent before the key)

echo "❯ Parsing worker IPs from $CONFIG…"
# look for lines under "worker_ips:" that begin with "-" and extract the part before the colon
WORKER_IPS=($(awk '
  /^[[:space:]]*worker_ips[[:space:]]*:/ { in_block=1; next }
  in_block && /^[[:space:]]*-[[:space:]]*/ {
    # strip “- ” and split at “:”
    line=$0; sub(/^[[:space:]]*-[[:space:]]*/,"",line)
    split(line,a,/:/)
    print a[1]
    next
  }
  # leave block when indented-block ends
  in_block && /^[[:space:]]*[^[:space:]]/ { in_block=0 }
' "$CONFIG"))

if [ ${#WORKER_IPS[@]} -eq 0 ]; then
    echo "⚠️  No worker_ips found in $CONFIG."
    echo "❯ Launching FedScale *locally* (driver.py start) …"
    python docker/driver.py start "$CONFIG"
    echo "✅ Done. Logs under your local log directory."
    exit 0    # ← prevent falling through to the submit below
else
    echo "❯ Ensuring agent-forwarding to each worker:"
    for ip in "${WORKER_IPS[@]}"; do
        printf "   • %s … " "$ip"
        if ssh -o BatchMode=yes -o ConnectTimeout=5 "$USER@$ip" "exit" 2>/dev/null; then
        echo "OK"
        else
        echo "warming up…"
        ssh -o StrictHostKeyChecking=no "$USER@$ip" "echo '[$(hostname)] agent forwarding OK'"
        fi
    done
fi

echo "❯ Launching FedScale job on the parameter server…"
python docker/driver.py submit "$CONFIG"

echo "✅ Done. Check logs under the path defined in your config."