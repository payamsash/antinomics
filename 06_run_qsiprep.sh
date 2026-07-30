#!/bin/bash
#
# Per-subject QSIPrep: DWI denoise/degibbs/eddy-current+motion correction
# (FSL eddy via QSIPrep's --hmc-model eddy) + bias correction + resampling to
# 1.2mm isotropic, MNI152NLin2009cAsym anatomical space. No SDC -- there are
# no fieldmaps in this dataset (confirmed in step 2 research), so this is
# left as QSIPrep's natural default (no --use-syn-sdc), not a special flag.
#
# GPU: confirmed via a smoke test (sub-asjt, 2026-07-30) that QSIPrep 26.0.0's
# container only ships eddy_cpu -- no eddy_cuda binary -- so --hmc-model eddy
# never actually touches the GPU here despite --nv. There is therefore no real
# GPU contention to serialize against; N_JOBS=4 runs subjects concurrently as
# a CPU-bound job (30 cores/117GB box), matching the original conservative
# concurrency budget. Revisit if a future QSIPrep image ships GPU eddy.
#
# Excluded subjects (derivatives/excluded_subjects.tsv, MRIQC-based QC calls)
# are skipped by default -- pass them explicitly as positional args to
# override (same convention as 05_run_mriqc.sh).
#
# Idempotent, per-subject: completion marker is QSIPrep's own top-level
# derivatives/qsiprep/sub-<id>.html report. Safe to re-run after a partial
# failure -- only subjects without a report are (re)processed; failed work
# dirs are kept for inspection, successful ones deleted to reclaim space.
#
# Run with:
#   ./06_run_qsiprep.sh                   # every BIDS subject not excluded
#   ./06_run_qsiprep.sh sub-dmxi sub-cwax # only these subjects (bypasses exclusion list)

set -euo pipefail

VOLUME=/home/ubuntu/volume
BIDS_ROOT="$VOLUME/antinomics"
OUT_DIR="$BIDS_ROOT/derivatives/qsiprep"
WORK_ROOT="$VOLUME/work/qsiprep"
LOG_DIR="$BIDS_ROOT/derivatives/logs/06_qsiprep"
MANIFEST="$LOG_DIR/qsiprep_status.tsv"
EXCLUDE_FILE="$BIDS_ROOT/derivatives/excluded_subjects.tsv"

APPTAINER_BIN="$VOLUME/miniconda3/envs/apptainer/bin/apptainer"
export APPTAINER_CACHEDIR="$VOLUME/apptainer_cache"
export APPTAINER_TMPDIR="$VOLUME/apptainer_tmp"
TEMPLATEFLOW_CACHE="$VOLUME/templateflow_cache"
QSIPREP_SIF="$VOLUME/containers/qsiprep-26.0.0.sif"
FS_LICENSE_CONTAINER_PATH="/data/derivatives/freesurfer_license.txt"

N_JOBS=4             # CPU-bound concurrency -- no GPU contention (see note above)
NPROCS=6
OMP_NTHREADS=4
MEM_MB=18000         # x4 concurrent = 72GB of 117GB total, leaves headroom
OUTPUT_RESOLUTION=1.2

mkdir -p "$OUT_DIR" "$WORK_ROOT" "$LOG_DIR"

if [ ! -f "$MANIFEST" ]; then
    printf "subject\tstatus\ttimestamp\n" > "$MANIFEST"
fi

update_manifest () {
    local subject="$1" status="$2"
    ( flock -x 200
      printf "%s\t%s\t%s\n" "$subject" "$status" "$(date -Iseconds)" >> "$MANIFEST"
    ) 200>"$MANIFEST.lock"
}
export -f update_manifest
export MANIFEST APPTAINER_BIN APPTAINER_CACHEDIR APPTAINER_TMPDIR TEMPLATEFLOW_CACHE \
       QSIPREP_SIF BIDS_ROOT OUT_DIR WORK_ROOT LOG_DIR NPROCS OMP_NTHREADS MEM_MB \
       OUTPUT_RESOLUTION FS_LICENSE_CONTAINER_PATH

process_subject () {
    set -uo pipefail   # no -e: a failed apptainer run must fall through to the
                       # FAILED branch below, not kill the parallel worker

    local bids_id=$1        # e.g. sub-dmxi
    local code=${bids_id#sub-}
    local work_dir="$WORK_ROOT/$bids_id"
    local log_file="$LOG_DIR/${bids_id}.log"
    local report="$OUT_DIR/${bids_id}.html"

    if [ -f "$report" ]; then
        echo "[skip] $bids_id already has a QSIPrep report"
        update_manifest "$bids_id" DONE
        return 0
    fi

    if [ ! -e "$BIDS_ROOT/$bids_id/dwi" ]; then
        echo "[skip] $bids_id has no dwi/ -- nothing for QSIPrep to do"
        update_manifest "$bids_id" NO_DWI
        return 0
    fi

    echo "[$bids_id] running QSIPrep"
    update_manifest "$bids_id" RUNNING
    mkdir -p "$work_dir"

    "$APPTAINER_BIN" run --nv --cleanenv \
        -B "$BIDS_ROOT":/data:ro \
        -B "$OUT_DIR":/out \
        -B "$work_dir":/work \
        -B "$TEMPLATEFLOW_CACHE":/opt/templateflow \
        --env TEMPLATEFLOW_HOME=/opt/templateflow \
        "$QSIPREP_SIF" \
        /data /out participant \
        --participant-label "$code" \
        --output-resolution "$OUTPUT_RESOLUTION" \
        --fs-license-file "$FS_LICENSE_CONTAINER_PATH" \
        --hmc-model eddy \
        --nprocs "$NPROCS" --omp-nthreads "$OMP_NTHREADS" --mem "$MEM_MB" \
        --work-dir /work \
        --skip-bids-validation \
        --notrack \
        > "$log_file" 2>&1
    local rc=$?

    if [ "$rc" -eq 0 ] && [ -f "$report" ]; then
        update_manifest "$bids_id" DONE
        rm -rf "$work_dir"
        echo "[$bids_id] done"
    else
        update_manifest "$bids_id" FAILED
        echo "[error] $bids_id: QSIPrep failed (exit $rc) -- see $log_file, work dir kept at $work_dir" >&2
    fi
}
export -f process_subject

if [ "$#" -gt 0 ]; then
    subjects=("$@")
else
    declare -A excluded
    if [ -f "$EXCLUDE_FILE" ]; then
        while IFS=$'\t' read -r subj excl _rest; do
            [ "$subj" = "subject" ] && continue   # header
            [ "$excl" = "yes" ] && excluded["$subj"]=1
        done < "$EXCLUDE_FILE"
    fi
    subjects=()
    while IFS= read -r s; do
        [ -n "${excluded[$s]:-}" ] && { echo "[exclude] $s (see $EXCLUDE_FILE)"; continue; }
        subjects+=("$s")
    done < <(cd "$BIDS_ROOT" && ls -d sub-*)
fi

echo "Subjects to process (${#subjects[@]}): ${subjects[*]}"
printf '%s\n' "${subjects[@]}" | parallel --jobs "$N_JOBS" process_subject {}

echo
echo "Done. Manifest: $MANIFEST"
n_done=$(tail -n +2 "$MANIFEST" | awk -F'\t' '{last[$1]=$2} END{c=0; for (s in last) if (last[s]=="DONE") c++; print c}')
n_failed=$(tail -n +2 "$MANIFEST" | awk -F'\t' '{last[$1]=$2} END{for (s in last) if (last[s]=="FAILED") print s}')
echo "Subjects DONE: $n_done / ${#subjects[@]}"
if [ -n "$n_failed" ]; then
    echo "Subjects currently FAILED (re-run this script to retry): $n_failed" >&2
fi
