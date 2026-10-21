#!/usr/bin/env bash
# Fetches MCBO at a fixed commit, applies our patch and creates the environment.
# The patch adds Fast BO, a faster ask/tell path without the DataFrame conversions,
# and the LABS, MaxSAT and Contamination tasks from Bounce.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party

HEBO_URL=https://github.com/huawei-noah/HEBO.git
HEBO_COMMIT=ee6112d39d1a9e9703fecaf9057193e1ec9dae72

if [ ! -d third_party/HEBO ]; then
    git clone "$HEBO_URL" third_party/HEBO
    git -C third_party/HEBO checkout --quiet "$HEBO_COMMIT"
    git -C third_party/HEBO apply "$PWD/../patches/mcbo.patch"
fi

uv sync
echo "Environment ready: combinatorial/.venv"
