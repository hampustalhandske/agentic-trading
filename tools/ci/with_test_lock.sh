#!/usr/bin/env bash
# Runs a command while holding this repo's single test lock, so at most one test run of the
# project (main checkout or any worktree) is active at a time. The lock is the directory
# agentic-test.lock in the repo's shared git dir, created atomically with mkdir; the holder's PID
# and command are written inside it and the directory is removed when the command ends. A lock
# whose holder PID no longer exists is taken over. Polls every TEST_LOCK_POLL seconds (default
# 5) for up to TEST_LOCK_WAIT seconds (default 1800), then exits 75 without running the command.
# Otherwise exits with the command's status.
#
# A pytest command must name its targets: when the command contains a `pytest` word, the
# arguments after it that are not options or option values are its targets, and the wrapper
# exits 64 before taking the lock when there is none, or when one of them (the part before any
# `::`) is not an existing path. The whole suite runs only through tools/ci/local_ci.sh.
#
# Usage: tools/ci/with_test_lock.sh <command> [args...]
set -u
# pytest options that take their value as the next argument.
value_options=" -k -m -p -c -o -W -n -r --durations --rootdir --confcutdir --basetemp \
 --timeout --deselect --ignore --ignore-glob --log-level --maxfail --tb --junitxml "
seen_pytest=0; expect_value=0; targets=0
for arg in "$@"; do
  if [ "$seen_pytest" -eq 0 ]; then
    [ "$(basename -- "$arg")" = "pytest" ] && seen_pytest=1
    continue
  fi
  if [ "$expect_value" -eq 1 ]; then expect_value=0; continue; fi
  case "$arg" in
    --*=*) continue ;;
    -*) case "$value_options" in *" $arg "*) expect_value=1 ;; esac; continue ;;
  esac
  if [ ! -e "${arg%%::*}" ]; then
    echo "test lock: refused: pytest target '$arg' does not exist" >&2
    exit 64
  fi
  targets=$((targets + 1))
done
if [ "$seen_pytest" -eq 1 ] && [ "$targets" -eq 0 ]; then
  echo "test lock: refused: a pytest command needs at least one target path" >&2
  exit 64
fi
common=$(git rev-parse --path-format=absolute --git-common-dir) || exit 1
lock="$common/agentic-test.lock"
limit=${TEST_LOCK_WAIT:-1800}
poll=${TEST_LOCK_POLL:-5}
SECONDS=0
announced=0
while ! mkdir "$lock" 2>/dev/null; do
  holder=$(cat "$lock/pid" 2>/dev/null || true)
  if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
    echo "test lock: holder $holder is gone; taking over" >&2
    rm -rf "$lock"
    continue
  fi
  if [ "$SECONDS" -ge "$limit" ]; then
    echo "test lock: still held by $holder ($(cat "$lock/cmd" 2>/dev/null)) after ${limit}s" >&2
    exit 75
  fi
  if [ "$announced" -eq 0 ]; then
    echo "test lock: held by $holder ($(cat "$lock/cmd" 2>/dev/null)); waiting" >&2
    announced=1
  fi
  sleep "$poll"
done
trap 'rm -rf "$lock"' EXIT
echo "$$" > "$lock/pid"
echo "$*" > "$lock/cmd"
"$@"
