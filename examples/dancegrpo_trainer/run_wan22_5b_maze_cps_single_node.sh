#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SDE_TYPE="${SDE_TYPE:-cps}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-wan22_5b_maze_cps_single_node_${MAX_STEPS:-1}step}"

exec "$SCRIPT_DIR/run_wan22_5b_maze_tracker_single_node.sh" "$@"
