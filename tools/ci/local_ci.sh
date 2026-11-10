#!/usr/bin/env bash
# Runs the five steps of .github/workflows/ci.yml on a clean checkout of <sha>: a detached
# worktree under .claude/worktrees/ci-<sha>, removed when the script ends. Prints each step's
# exit code. The test step runs the whole suite (`tests`) under tools/ci/with_test_lock.sh, so
# it waits for any other test run of the repo to end first.
# Exits 2 when <sha> is not a commit or the worktree cannot be made at it, 1 when a step failed,
# 0 when every step passed.
# Usage: tools/ci/local_ci.sh <sha>
set -uo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: tools/ci/local_ci.sh <sha>" >&2
  exit 64
fi
sha=$1
here=$(cd "$(dirname "$0")" && pwd) || exit 2
repo=$(git rev-parse --show-toplevel) || exit 2
commit=$(git -C "$repo" rev-parse --verify --quiet "$sha^{commit}") \
  || { echo "local ci: $sha is not a commit of $repo" >&2; exit 2; }
dir="$repo/.claude/worktrees/ci-$sha"
git -C "$repo" worktree add --detach "$dir" "$commit" >/dev/null 2>&1 \
  || { echo "local ci: worktree add failed for $sha" >&2; exit 2; }
trap 'cd "$repo" && git worktree remove --force "$dir"' EXIT
cd "$dir" || exit 2
head=$(git rev-parse HEAD)
if [ "$head" != "$commit" ]; then
  echo "local ci: worktree HEAD $head is not $commit" >&2
  exit 2
fi
echo "checked out $(git rev-parse --short HEAD), clean: $(test -z "$(git status --porcelain)" && echo yes || echo no)"
status=0
for step in "uv sync --locked --group dev" "uv run ruff check ." "uv run ruff format --check ." \
            "uv run mypy src/" "$here/with_test_lock.sh uv run pytest -rs -p no:cacheprovider tests"; do
  echo "== $step"
  # shellcheck disable=SC2086  # each step is a word list
  $step 2>&1 | tail -3
  code=${PIPESTATUS[0]}
  echo "exit $code"
  [ "$code" -eq 0 ] || status=1
done
echo "overall $([ "$status" -eq 0 ] && echo PASS || echo FAIL) on $sha"
exit "$status"
