#!/usr/bin/env bash
# Runs the CI job "check" of .github/workflows/ci.yml in a Linux container on a clean checkout of
# <sha>. The repo is mounted read-only and this folder beside it; inside, container_check.sh
# clones the repo, checks out the commit, verifies HEAD is that commit and runs the steps, so the
# checkout has a real .git as actions/checkout gives and nothing built lands on the host.
# Exits 2 before starting a container when <sha> is not a commit of <repo>; otherwise exits with
# the container's status, 0 only when every step passed.
# Usage: tools/ci/container_ci.sh <repo> <sha>   (PLATFORM=linux/amd64 for x86-64; default
# linux/arm64)
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: tools/ci/container_ci.sh <repo> <sha>" >&2
  exit 64
fi
repo=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
commit=$(git -C "$repo" rev-parse --verify --quiet "$2^{commit}") \
  || { echo "container ci: $2 is not a commit of $repo" >&2; exit 2; }
exec docker run --rm --memory=4g --platform "${PLATFORM:-linux/arm64}" \
  -v "$repo:/repo:ro" -v "$here:/ci:ro" \
  -e CI=true -e UV_PYTHON=3.12 -e UV_LINK_MODE=copy -e SHA="$commit" \
  python:3.12-slim bash /ci/container_check.sh
