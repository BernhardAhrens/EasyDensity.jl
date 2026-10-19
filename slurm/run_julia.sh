# Julia can fail to map libgcc_s.so.1 when several jobs start on one node.
# That exits in a few seconds. Retry those fast exits, and leave a real training failure alone.
run_julia() {
    local script=$1
    local attempt=1
    local max_attempts=3
    local start elapsed
    while true; do
        start=$(date +%s)
        if julia --project=. --threads="${SLURM_CPUS_PER_TASK}" "$script"; then
            return 0
        fi
        elapsed=$(( $(date +%s) - start ))
        if (( attempt >= max_attempts || elapsed >= 60 )); then
            echo "julia ${script} failed after ${elapsed}s on attempt ${attempt}" >&2
            return 1
        fi
        echo "julia ${script} exited after ${elapsed}s on attempt ${attempt}; retrying in $(( attempt * 20 ))s" >&2
        sleep $(( attempt * 20 ))
        attempt=$(( attempt + 1 ))
    done
}
