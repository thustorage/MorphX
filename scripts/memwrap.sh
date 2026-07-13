#!/usr/bin/env bash
# memwrap.sh - wrap a build command and log memory usage
# Usage: memwrap.sh <log-prefix> -- <command> [args...]
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <log-prefix> -- <command> [args...]" >&2
  exit 2
fi
PREFIX="$1"
shift
if [ "$1" != "--" ]; then
  echo "Usage: $0 <log-prefix> -- <command> [args...]" >&2
  exit 2
fi
shift
LOGDIR="./memlogs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S-%N)
LOGFILE="$LOGDIR/${PREFIX}_${TIMESTAMP}.log"
# Use /usr/bin/time -v for verbose resource usage. If not available, fall back to /usr/bin/time
TIMECMD="/usr/bin/time -v"
if [ ! -x "/usr/bin/time" ]; then
  echo "/usr/bin/time not found" >&2
  exec "$@"
fi
# Run the command and capture both stdout/stderr and time verbose output
{
  echo "===== CMD: $* ====="
  echo "===== START: $(date) ====="
  $TIMECMD "$@"
  RET=$?
  echo "===== END: $(date) ====="
  exit $RET
} 2>&1 | tee "$LOGFILE"

# Extract peak RSS and exit status to a small summary file
if grep -q "Maximum resident set size" "$LOGFILE"; then
  grep "Maximum resident set size" "$LOGFILE" > "${LOGFILE}.summary"
fi

exit 0
