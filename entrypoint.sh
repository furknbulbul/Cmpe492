#!/bin/bash
source /opt/python3/venv/base/bin/activate
cd /opt/python3/venv/base/Cmpe492


# Check if the directory exists
if [ -d "/opt/python3/venv/base/Cmpe492" ]; then
  cd /opt/python3/venv/base/Cmpe492
else
  echo "Directory /opt/python3/venv/base/Cmpe492 does not exist."
  exit 1
fi

# Ensure git command is available
if ! command -v git &> /dev/null; then
  echo "git command not found."
  exit 1
fi
echo "Running git pull..."
if git pull origin main; then
  echo "git pull completed successfully."
else
  echo "git pull failed."
  exit 1
fi
export WANDB_API_KEY=$WANDB_API_KEY
exec "$@"