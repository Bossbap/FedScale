#!/bin/bash

echo "==== Starting FedScale Job ===="

# --- 1. Initialize ssh-agent if not already running ---
if [ -z "$SSH_AUTH_SOCK" ]; then
    echo "Starting ssh-agent..."
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa
else
    echo "ssh-agent already running."
fi

# --- 2. Check SSH connectivity to the worker (optional sanity check) ---
echo "Checking connection to 129.104.253.48..."
ssh -o BatchMode=yes -o ConnectTimeout=5 baptiste.geisenberger@129.104.253.48 "hostname"
if [ $? -ne 0 ]; then
    echo "SSH connection to worker failed. Aborting."
    exit 1
fi

# --- 3. Kill existing jobs related to aggregator/executor/learner ---
echo "Killing existing aggregator/executor/learner processes..."
pkill -f aggregator.py
pkill -f executor.py
pkill -f learner.py

# You can also do a check to confirm if desired:
# pgrep -fl "aggregator.py|executor.py|learner.py"

# --- 4. Launch FedScale job ---
echo "Launching job..."
python docker/driver.py start benchmark/configs/speech/google_speech.yml

echo "==== Job launched ===="