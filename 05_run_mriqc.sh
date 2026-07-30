#!/bin/bash
#
# Per-subject MRIQC (T1w, T2w, both BOLD runs) image-quality metrics + reports.
# Runs first, alone, ahead of the much longer QSIPrep/fMRIPrep stages: cheap
# (no GPU, ~2-4GB RAM/subject), gives an early data-quality read before
# committing multi-day compute to the rest of the pipeline. DWI is intentionally
# excluded here -- QSIPrep/eddy_quad covers dMRI QC in a later step.
#
# Idempotent, per-subject: completion marker is the top-level
# derivatives/mriqc/sub-<id>.html report MRIQC writes on a successful
# participant-level run. Safe to re-run after a partial failure -- only
# subjects without a report are (re)processed.
#
# Run with:
#   ./05_run_mriqc.sh                   # every BIDS subject
#   ./05_run_mriqc.sh sub-dmxi sub-cwax  # only these subjects
#
# Subjects listed in $SKIP_T2W_FILE (one sub-<id> per line) are run with
# -m T1w bold only. Some acq-highreshippo_T2w slabs (partial-FOV, hippocampus-
# targeted) deterministically crash MRIQC's anatomical MNI-registration/IQM
# node ("Input inhomogeneity-corrected data seem empty") -- confirmed via
# crash-log + fslstats inspection to be a real per-subject slab-angulation
# issue, not corrupted data or resource contention (identical failure on
# repeated retry with the machine otherwise idle). T1w/bold IQMs are unaffected
# and worth keeping; T2w is skipped for these subjects rather than endlessly
# retried.

set -euo pipefail

VOLUME=/home/ubuntu/volume
BIDS_ROOT="$VOLUME/antinomics"
OUT_DIR="$BIDS_ROOT/derivatives/mriqc"
WORK_ROOT="$VOLUME/work/mriqc"
LOG_DIR="$BIDS_ROOT/derivatives/logs/05_mriqc"
MANIFEST="$LOG_DIR/mriqc_status.tsv"
SKIP_T2W_FILE="$LOG_DIR/skip_t2w_subjects.txt"

APPTAINER_BIN="$VOLUME/miniconda3/envs/apptainer/bin/apptainer"
export APPTAINER_CACHEDIR="$VOLUME/apptainer_cache"
export APPTAINER_TMPDIR="$VOLUME/apptainer_tmp"
TEMPLATEFLOW_CACHE="$VOLUME/templateflow_cache"
MRIQC_SIF="$VOLUME/containers/mriqc-24.0.2.sif"

N_JOBS=8            # subjects concurrent; MRIQC is light (no GPU), 30 cores available
NPROCS=3
OMP_NTHREADS=1
MEM_GB=6

mkdir -p "$OUT_DIR" "$WORK_ROOT" "$LOG_DIR"

if [ ! -f "$MANIFEST" ]; then
    printf "subject\tstatus\ttimestamp\n" > "$MANIFEST"
fi
touch "$SKIP_T2W_FILE"

update_manifest () {
    local subject="$1" status="$2"
    ( flock -x 200
      printf "%s\t%s\t%s\n" "$subject" "$status" "$(date -Iseconds)" >> "$MANIFEST"
    ) 200>"$MANIFEST.lock"
}
export -f update_manifest
export MANIFEST APPTAINER_BIN APPTAINER_CACHEDIR APPTAINER_TMPDIR TEMPLATEFLOW_CACHE \
       MRIQC_SIF BIDS_ROOT OUT_DIR WORK_ROOT LOG_DIR NPROCS OMP_NTHREADS MEM_GB SKIP_T2W_FILE

process_subject () {
    set -uo pipefail   # no -e: a failed apptainer run must fall through to the
                       # FAILED branch below, not kill the parallel worker

    local bids_id=$1        # e.g. sub-dmxi
    local code=${bids_id#sub-}
    local work_dir="$WORK_ROOT/$bids_id"
    local log_file="$LOG_DIR/${bids_id}.log"

    local skip_t2w=0
    grep -qxF "$bids_id" "$SKIP_T2W_FILE" && skip_t2w=1

    local modalities="T1w T2w bold"
    [ "$skip_t2w" -eq 1 ] && modalities="T1w bold"

    # MRIQC 24.x writes one report per source scan (not a single sub-<id>.html),
    # named after the source file with .nii.gz -> .html, flat in $OUT_DIR, e.g.
    # sub-asjt_T1w.html, sub-asjt_acq-highreshippo_T2w.html,
    # sub-asjt_task-rest_run-1_bold.html. Derive the expected set from whatever
    # T1w/T2w/bold files this subject actually has, so subjects with missing
    # runs (e.g. sub-uaxk, no run-1 BOLD) are handled correctly. Subjects in
    # $SKIP_T2W_FILE have T2w excluded from the expected set (and from the
    # MRIQC invocation below) since it is known to be unprocessable for them.
    local expected_reports=()
    local f bn
    local anat_globs=("$BIDS_ROOT/$bids_id"/anat/*_T1w.nii.gz)
    if [ "$skip_t2w" -eq 0 ]; then
        anat_globs+=("$BIDS_ROOT/$bids_id"/anat/*_T2w.nii.gz)
    fi
    for f in "${anat_globs[@]}" "$BIDS_ROOT/$bids_id"/func/*_bold.nii.gz; do
        [ -e "$f" ] || continue
        bn=$(basename "$f")
        expected_reports+=("$OUT_DIR/${bn%.nii.gz}.html")
    done

    local all_present=1
    for f in "${expected_reports[@]}"; do
        [ -f "$f" ] || { all_present=0; break; }
    done

    if [ "$all_present" -eq 1 ] && [ "${#expected_reports[@]}" -gt 0 ]; then
        echo "[skip] $bids_id already has all ${#expected_reports[@]} MRIQC reports"
        update_manifest "$bids_id" DONE
        return 0
    fi

    echo "[$bids_id] running MRIQC"
    update_manifest "$bids_id" RUNNING
    mkdir -p "$work_dir"

    "$APPTAINER_BIN" run --cleanenv \
        -B "$BIDS_ROOT":/data:ro \
        -B "$OUT_DIR":/out \
        -B "$work_dir":/work \
        -B "$TEMPLATEFLOW_CACHE":/opt/templateflow \
        --env TEMPLATEFLOW_HOME=/opt/templateflow \
        "$MRIQC_SIF" \
        /data /out participant \
        --participant-label "$code" \
        -m $modalities \
        --nprocs "$NPROCS" --omp-nthreads "$OMP_NTHREADS" --mem "$MEM_GB" \
        --work-dir /work \
        --no-sub \
        > "$log_file" 2>&1
    local rc=$?

    all_present=1
    for f in "${expected_reports[@]}"; do
        [ -f "$f" ] || { all_present=0; break; }
    done

    if [ "$rc" -eq 0 ] && [ "$all_present" -eq 1 ]; then
        update_manifest "$bids_id" DONE
        rm -rf "$work_dir"
        echo "[$bids_id] done"
    else
        update_manifest "$bids_id" FAILED
        echo "[error] $bids_id: MRIQC failed (exit $rc) -- see $log_file, work dir kept at $work_dir" >&2
    fi
}
export -f process_subject

if [ "$#" -gt 0 ]; then
    subjects=("$@")
else
    mapfile -t subjects < <(cd "$BIDS_ROOT" && ls -d sub-* )
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
