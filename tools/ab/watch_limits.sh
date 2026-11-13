#!/usr/bin/env bash
# Watches the process <pid>. Every WATCH_INTERVAL seconds (default 10) while it lives, reads
# free memory (the system-wide free percentage `memory_pressure -Q` prints) and free disk (the
# GB available on /, column 4 of the second line `df -g /` prints), keeping the lowest of each.
# When free memory falls below 20% or free disk below 10 GB, sends TERM to <pid> and its
# descendants and exits 3. When a probe gives no whole-number reading, does the same and exits 2.
# When <pid> ends, logs the lowest readings and exits 0.
# The probe commands are WATCH_MEMORY_CMD (default memory_pressure) and WATCH_DISK_CMD (default
# df), each called with the arguments above.
# Usage: tools/ab/watch_limits.sh <pid>
set -u
if [ "$#" -ne 1 ]; then
  echo "usage: tools/ab/watch_limits.sh <pid>" >&2
  exit 64
fi
pid=$1
memory_cmd=${WATCH_MEMORY_CMD:-memory_pressure}
disk_cmd=${WATCH_DISK_CMD:-df}
interval=${WATCH_INTERVAL:-10}
low_mem=100; low_disk=100000
stamp() { date '+%H:%M:%S'; }
descendants() { local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }
is_whole() { case "$1" in '' | *[!0-9]*) return 1 ;; esac; }
stop() {
  echo "$(stamp) $1; stopping $pid"
  # shellcheck disable=SC2046  # one argument per descendant pid
  kill -TERM $(descendants "$pid") "$pid" 2>/dev/null
  exit "$2"
}
while kill -0 "$pid" 2>/dev/null; do
  mem=$("$memory_cmd" -Q | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  disk=$("$disk_cmd" -g / | awk 'NR==2 {print $4}')
  if ! is_whole "$mem" || ! is_whole "$disk"; then
    stop "NO READING free mem '$mem' disk '$disk'" 2
  fi
  [ "$mem" -lt "$low_mem" ] && low_mem=$mem
  [ "$disk" -lt "$low_disk" ] && low_disk=$disk
  if [ "$mem" -lt 20 ] || [ "$disk" -lt 10 ]; then
    stop "LIMIT free mem $mem% disk ${disk} GB" 3
  fi
  sleep "$interval"
done
echo "$(stamp) watch ended; lowest free mem $low_mem%, lowest free disk $low_disk GB"
