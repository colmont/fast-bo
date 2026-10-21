#!/usr/bin/env bash
# Builds the Bencher Docker image for the continuous benchmarks. Our patch makes
# MuJoCo and Lunar Lander run one seeded rollout per call.
# Start the server with: docker run -p 50051:50051 -d bencher-fast-bo
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party

BENCHER_URL=https://github.com/lpapenme/bencher.git
BENCHER_COMMIT=ba9ada07a7f3eaa5efd691200ed4866ff4ca6e1b

if [ ! -d third_party/bencher ]; then
    git clone "$BENCHER_URL" third_party/bencher
    git -C third_party/bencher checkout --quiet "$BENCHER_COMMIT"
    git -C third_party/bencher apply "$PWD/patches/bencher.patch"
fi
docker build --platform linux/amd64 -t bencher-fast-bo third_party/bencher
