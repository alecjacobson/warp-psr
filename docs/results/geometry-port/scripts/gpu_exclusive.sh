#!/usr/bin/env bash
# Hold the shared machine's exclusive GPU lock for one bounded benchmark process.
gpu_run() (
    case "${1:-}" in
        shared) mode=-s ;;
        exclusive) mode=-x ;;
        *) echo 'Usage: gpu_run shared|exclusive COMMAND [ARGS...]' >&2; exit 2 ;;
    esac
    shift
    if [ "$#" -eq 0 ]; then
        echo 'gpu_run: missing command' >&2
        exit 2
    fi

    exec 9>>/home/horde/.gpu-admission.lock || exit 1
    exec 8>>/home/horde/.gpu-usage.lock || exit 1

    # Serialize admission; hold the gate while existing GPU work drains.
    flock -x -w 60 -E 75 9 || exit "$?"
    flock "$mode" -w 60 -E 75 8 || exit "$?"
    flock -u 9 || exit "$?"
    exec 9>&-

    "$@"
)

check_and_run() {
    local gpu_activity
    gpu_activity=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader) || return 1
    if [ -n "$gpu_activity" ]; then
        echo "Defer: GPU activity remains inside exclusive window: $gpu_activity" >&2
        return 75
    fi
    "$@"
}

gpu_run exclusive check_and_run "$@"
