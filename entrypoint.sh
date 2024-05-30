#!/bin/bash
source /opt/python3/venv/base/bin/activate
cd /opt/python3/venv/base/Cmpe492
git pull origin main\
export WANDB_API_KEY=$WANDB_API_KEY

exec "$@"