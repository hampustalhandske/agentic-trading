#!/usr/bin/env bash
# Runs tools/ab/compare_long.py on two run directories, prints its report and writes it to
# <report file>, and exits with the comparison's status: 0 only when the runs are identical.
# When writing the report fails, exits non-zero as well (pipefail).
# PYTHON is the interpreter command (default "uv run python"), split on spaces.
# Usage: tools/ab/compare_and_record.sh <old run dir> <new run dir> <report file>
#            [compare_long.py flags...]
set -uo pipefail
if [ "$#" -lt 3 ]; then
  echo "usage: tools/ab/compare_and_record.sh <old run dir> <new run dir> <report file> [flags]" >&2
  exit 64
fi
old=$1; new=$2; report=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd) || exit 2
read -r -a python <<< "${PYTHON:-uv run python}"
"${python[@]}" "$here/compare_long.py" "$old" "$new" "$@" | tee "$report"
