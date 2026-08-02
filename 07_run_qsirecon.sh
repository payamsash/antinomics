#!/bin/bash
#
# Per-subject QSIRecon: single-shell constrained spherical deconvolution
# (SS3T/dhollander), ACT-HSVS-anchored probabilistic tractography (iFOD2,
# 10M streamlines) + SIFT2 filtering, and structural connectome construction
# against QSIRecon's built-in "4S" cortex+subcortex atlases. Reads QSIPrep's
# derivatives (step 06) as input; requires FreeSurfer (step 01/02) for the
# HSVS 5-tissue-type segmentation.
#
# recon-spec: mrtrix_singleshell_ss3t_ACT-hsvs (built into the container --
# confirmed via `find` inside the .sif, matches this dataset: single-shell
# 32-dir b=1000 DWI, no second shell for msmt_csd). Not a custom YAML.
#
# Atlases: built-in 4S parcellations only for this pass (4S456Parcels,
# 4S856Parcels -- moderate and high resolution). The plan's "custom
# pure-cortex Schaefer 100-1000" atlas (nilearn-built, BIDS-Atlas dataset via
# `-d`) is NOT included here -- it needs its own prep script first. QSIRecon
# supports adding atlases in a later rerun without redoing tractography (the
# csd/track_ifod2 nodes are reused; only the connectivity node re-executes
# for new atlases), so this can follow on later without re-running from
# scratch.
#
# CPU-only (no --nv, no GPU-touching steps in this recon-spec). Concurrency
# kept at the same conservative N_JOBS=4 budget as 06_run_qsiprep.sh since
# per-subject runtime here is unmeasured (iFOD2 with 10M streamlines can be
# substantial) -- recommend a smoke test before the full batch, same as 06.
#
# FreeSurfer subjects dir: derivatives/freesurfer/sub-<id> are symlinks
# (../../../Tinception/subjects_fs_dir/<code>) materialized by
# 03_prepare_bids_scaffold.py. Because BIDS_ROOT is bound into the container
# at /data, the symlinks' relative targets resolve (from the container's
# point of view) to /Tinception/subjects_fs_dir/<code> -- so that real host
# directory is bound at the matching absolute container path below.
#
# Excluded subjects (derivatives/excluded_subjects.tsv) are skipped by
# default, same convention as 05/06. Subjects without a completed QSIPrep
# report are also skipped (NO_QSIPREP status) -- this stage hard-depends on
# QSIPrep output.
#
# Idempotent, per-subject: completion marker is QSIRecon's own top-level
# derivatives/qsirecon/sub-<id>.html report (--report-output-level root is
# the default, confirmed via --help). Safe to re-run after a partial
# failure -- only subjects without a report are (re)processed; failed work
# dirs are kept for inspection, successful ones deleted to reclaim space.
#
# Run with:
#   ./07_run_qsirecon.sh                   # every QSIPrep-DONE subject not excluded
#   ./07_run_qsirecon.sh sub-dmxi sub-cwax # only these subjects (bypasses exclusion list)

set -euo pipefail

VOLUME=/home/ubuntu/volume
BIDS_ROOT="$VOLUME/antinomics"
QSIPREP_DIR="$BIDS_ROOT/derivatives/qsiprep"
FS_SUBJECTS_HOST_DIR="$VOLUME/Tinception/subjects_fs_dir"
OUT_DIR="$BIDS_ROOT/derivatives/qsirecon"
WORK_ROOT="$VOLUME/work/qsirecon"
LOG_DIR="$BIDS_ROOT/derivatives/logs/07_qsirecon"
MANIFEST="$LOG_DIR/qsirecon_status.tsv"
EXCLUDE_FILE="$BIDS_ROOT/derivatives/excluded_subjects.tsv"

APPTAINER_BIN="$VOLUME/miniconda3/envs/apptainer/bin/apptainer"
export APPTAINER_CACHEDIR="$VOLUME/apptainer_cache"
export APPTAINER_TMPDIR="$VOLUME/apptainer_tmp"
TEMPLATEFLOW_CACHE="$VOLUME/templateflow_cache"
QSIRECON_SIF="$VOLUME/containers/qsirecon-26.0.0.sif"
FS_LICENSE_CONTAINER_PATH="/data/derivatives/freesurfer_license.txt"

RECON_SPEC="mrtrix_singleshell_ss3t_ACT-hsvs"
ATLASES="4S456Parcels 4S856Parcels"

N_JOBS=4             # CPU-bound, unmeasured per-subject cost -- smoke test first
NPROCS=6
OMP_NTHREADS=4
MEM_MB=18000
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
       QSIRECON_SIF BIDS_ROOT QSIPREP_DIR FS_SUBJECTS_HOST_DIR OUT_DIR WORK_ROOT LOG_DIR \
       NPROCS OMP_NTHREADS MEM_MB OUTPUT_RESOLUTION FS_LICENSE_CONTAINER_PATH RECON_SPEC ATLASES

process_subject () {
    set -uo pipefail   # no -e: a failed apptainer run must fall through to the
                       # FAILED branch below, not kill the parallel worker

    local bids_id=$1        # e.g. sub-dmxi
    local code=${bids_id#sub-}
    local work_dir="$WORK_ROOT/$bids_id"
    local log_file="$LOG_DIR/${bids_id}.log"
    local report="$OUT_DIR/${bids_id}.html"
    local qsiprep_report="$QSIPREP_DIR/${bids_id}.html"

    if [ -f "$report" ]; then
        echo "[skip] $bids_id already has a QSIRecon report"
        update_manifest "$bids_id" DONE
        return 0
    fi

    if [ ! -f "$qsiprep_report" ]; then
        echo "[skip] $bids_id has no completed QSIPrep report -- nothing for QSIRecon to do"
        update_manifest "$bids_id" NO_QSIPREP
        return 0
    fi

    echo "[$bids_id] running QSIRecon"
    update_manifest "$bids_id" RUNNING
    mkdir -p "$work_dir"

    "$APPTAINER_BIN" run --cleanenv \
        -B "$BIDS_ROOT":/data:ro \
        -B "$FS_SUBJECTS_HOST_DIR":/Tinception/subjects_fs_dir:ro \
        -B "$OUT_DIR":/out \
        -B "$work_dir":/work \
        -B "$TEMPLATEFLOW_CACHE":/opt/templateflow \
        --env TEMPLATEFLOW_HOME=/opt/templateflow \
        "$QSIRECON_SIF" \
        /data/derivatives/qsiprep /out participant \
        --participant-label "$code" \
        --recon-spec "$RECON_SPEC" \
        --atlases $ATLASES \
        --fs-subjects-dir /data/derivatives/freesurfer \
        --fs-license-file "$FS_LICENSE_CONTAINER_PATH" \
        --output-resolution "$OUTPUT_RESOLUTION" \
        --nprocs "$NPROCS" --omp-nthreads "$OMP_NTHREADS" --mem "$MEM_MB" \
        --work-dir /work \
        --notrack \
        > "$log_file" 2>&1
    local rc=$?

    if [ "$rc" -eq 0 ] && [ -f "$report" ]; then
        update_manifest "$bids_id" DONE
        rm -rf "$work_dir"
        echo "[$bids_id] done"
    else
        update_manifest "$bids_id" FAILED
        echo "[error] $bids_id: QSIRecon failed (exit $rc) -- see $log_file, work dir kept at $work_dir" >&2
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
