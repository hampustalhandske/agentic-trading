#!/usr/bin/env bash
# The part of tools/ci/container_ci.sh that runs inside the container: installs git and uv,
# clones SOURCE (default /repo) into WORK (default /work), checks out the commit SHA, and runs
# the five steps of .github/workflows/ci.yml on it, printing each step's exit code.
# Exits 2 without running a step when an install, the clone or the checkout fails, or when HEAD
# is not SHA's commit afterwards; otherwise 1 when a step failed, 0 when every step passed.
# Usage: SHA=<sha> bash container_check.sh
set -uo pipefail
source_repo=${SOURCE:-/repo}
work=${WORK:-/work}
fail() {
  echo "container check: $*" >&2
  exit 2
}
if ! { apt-get update -qq >/dev/null && apt-get install -y -qq git procps >/dev/null; }; then
  fail "installing git failed"
fi
pip install -q --root-user-action=ignore uv==0.11.19 || fail "installing uv failed"
git config --global --add safe.directory "*" || fail "git config failed"
git clone -q "$source_repo" "$work" || fail "cloning $source_repo failed"
cd "$work" || fail "no clone at $work"
git checkout -q --detach "$SHA" || fail "checking out $SHA failed"
commit=$(git rev-parse --verify --quiet "$SHA^{commit}") || fail "$SHA is not a commit"
head=$(git rev-parse HEAD) || fail "no HEAD"
[ "$head" = "$commit" ] || fail "HEAD $head is not $SHA"
echo "checked out $(git rev-parse --short HEAD), clean: $(test -z "$(git status --porcelain)" && echo yes || echo no)"
status=0
for step in "uv sync --locked --group dev" "uv run ruff check ." \
            "uv run ruff format --check ." "uv run mypy src/" \
            "uv run pytest -rs -v --durations=15 -p no:cacheprovider"; do
  echo "== $step"
  # shellcheck disable=SC2086  # each step is a word list
  $step 2>&1 | grep -vE "^ [+-] |Docs:"
  code=${PIPESTATUS[0]}
  echo "exit $code"
  [ "$code" -eq 0 ] || status=1
done
exit "$status"
