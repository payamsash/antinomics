#!/bin/bash
#
# Per-subject XCP-D: denoising + parcellation/connectivity of fMRIPrep's
# preprocessed BOLD (task-rest_run-1/run-2, combined into one per-subject
# connectivity matrix per variant after independent per-run denoising --
# XCP-D denoises/bandpass-filters each run first, then concatenates, so the
# filter never operates across the run boundary; confirmed via docs before
# relying on it, no separate downstream concatenation script needed).
#
# Run TWICE per subject (--mode none, full manual control):
#   - primary, no GSR:  -p acompcor      -> derivatives/xcp_d/
#   - sensitivity, GSR: -p acompcor_gsr  -> derivatives/xcp_d_gsr/
# XCP-D's built-in "acompcor" strategy is 24 motion parameters + 5 WM/5 CSF
# aCompCor components -- a direct match for the plan's "aCompCor +
# 24-motion-parameters" spec, no custom YAML needed. FD scrubbing threshold
# 0.5mm, same threshold already used for the exclusion-list QC decision.
#
# Atlases: 4S456Parcels 4S856Parcels -- matches QSIRecon's atlas choice
# exactly, giving the SFC module matched SC/FC parcellations on the same
# node scheme instead of needing a cross-walk between atlas systems.
#
# --file-format nifti: step 08's fMRIPrep run was NIfTI-only (no
# --cifti-output), so this must be explicit rather than left on auto.
#
# --create-matrices all: REQUIRED to actually get a correlation/connectivity
# matrix out of XCP-D at all -- found via smoke test (2026-08-13): the first
# smoke-test run "finished successfully" and produced parcellated timeseries
# (stat-mean_timeseries.tsv) but no correlation matrix (*_relmat.tsv), for
# either the per-run or the combined-run output. Traced into the container's
# xcp_d/workflows/bold/concatenation.py source: the combined-run correlation
# node (TSVConnect -> *_relmat.tsv) is gated behind
# `if config.workflow.correlation_lengths and file_format == 'nifti'`, which
# is only ever populated by --create-matrices (an "abcd/hbcd mode option"
# despite being usable in --mode none too). Without it, --combine-runs alone
# only concatenates+denoises the timeseries -- it does NOT compute the
# correlation matrix. "all" uses the full amount of low-motion data (no
# subsampling to a fixed duration), which is what we want for a single
# per-subject connectivity matrix.
#
# Input is fMRIPrep's output dir directly as the fmri_dir positional arg
# (not via -d, which is for extra atlas/derivative datasets).
#
# CPU-only, no GPU steps. Excluded subjects (derivatives/excluded_subjects.tsv)
# skipped by default, same convention as 05-08. Hard-depends on the subject's
# fMRIPrep manifest entry being DONE (checked via the report file directly).
#
# Idempotent, per-subject, per-variant: completion marker is XCP-D's own
# top-level <out_dir>/sub-<id>.html report (--report-output-level defaults to
# "root" for --mode none, confirmed via --help -- lands directly in the
# output dir, same convention as fMRIPrep's step 08).
#
# Run with:
#   ./09_run_xcpd.sh                   # every BIDS subject not excluded
#   ./09_run_xcpd.sh sub-dmxi sub-cwax # only these subjects (bypasses exclusion list)

set -euo pipefail

VOLUME=/home/ubuntu/volume
BIDS_ROOT="$VOLUME/antinomics"
FMRIPREP_DIR="$BIDS_ROOT/derivatives/fmriprep"
OUT_DIR="$BIDS_ROOT/derivatives/xcp_d"
OUT_DIR_GSR="$BIDS_ROOT/derivatives/xcp_d_gsr"
WORK_ROOT="$VOLUME/work/xcpd"
LOG_DIR="$BIDS_ROOT/derivatives/logs/09_xcpd"
MANIFEST="$LOG_DIR/xcpd_status.tsv"
EXCLUDE_FILE="$BIDS_ROOT/derivatives/excluded_subjects.tsv"

APPTAINER_BIN="$VOLUME/miniconda3/envs/apptainer/bin/apptainer"
export APPTAINER_CACHEDIR="$VOLUME/apptainer_cache"
export APPTAINER_TMPDIR="$VOLUME/apptainer_tmp"
XCPD_SIF="$VOLUME/containers/xcp_d-26.1.1.sif"
FS_LICENSE_CONTAINER_PATH="/data/derivatives/freesurfer_license.txt"

ATLASES="4S456Parcels 4S856Parcels"

N_JOBS=4             # CPU-bound, no GPU steps -- same conservative budget as 06-08
NPROCS=6
OMP_NTHREADS=4
MEM_MB=18000          # x4 concurrent = 72GB of 117GB total, leaves headroom

mkdir -p "$OUT_DIR" "$OUT_DIR_GSR" "$WORK_ROOT" "$LOG_DIR"

if [ ! -f "$MANIFEST" ]; then
    printf "subject\tstatus\ttimestamp\n" > "$MANIFEST"
fi

update_manifest () {
    local row="$1" status="$2"
    ( flock -x 200
      printf "%s\t%s\t%s\n" "$row" "$status" "$(date -Iseconds)" >> "$MANIFEST"
    ) 200>"$MANIFEST.lock"
}
export -f update_manifest
export MANIFEST APPTAINER_BIN APPTAINER_CACHEDIR APPTAINER_TMPDIR \
       XCPD_SIF BIDS_ROOT FMRIPREP_DIR OUT_DIR OUT_DIR_GSR WORK_ROOT LOG_DIR \
       NPROCS OMP_NTHREADS MEM_MB FS_LICENSE_CONTAINER_PATH ATLASES

run_xcpd_variant () {
    # args: bids_id  code  variant(nogsr|gsr)  nuisance_strategy  out_dir
    local bids_id=$1 code=$2 variant=$3 strategy=$4 out_dir=$5
    local work_dir="$WORK_ROOT/${bids_id}_${variant}"
    local log_file="$LOG_DIR/${bids_id}_${variant}.log"
    local report="$out_dir/${bids_id}.html"
    local row="${bids_id}_${variant}"

    if [ -f "$report" ]; then
        echo "[skip] $bids_id ($variant) already has an XCP-D report"
        update_manifest "$row" DONE
        return 0
    fi

    echo "[$bids_id] running XCP-D ($variant, $strategy)"
    update_manifest "$row" RUNNING
    mkdir -p "$work_dir"

    "$APPTAINER_BIN" run --cleanenv \
        -B "$BIDS_ROOT":/data:ro \
        -B "$out_dir":/out \
        -B "$work_dir":/work \
        "$XCPD_SIF" \
        /data/derivatives/fmriprep /out participant \
        --mode none \
        --participant-label "$code" \
        --input-type fmriprep \
        --nuisance-regressors "$strategy" \
        --combine-runs y \
        --fd-thresh 0.5 \
        --atlases $ATLASES \
        --file-format nifti \
        --despike n \
        --smoothing 0 \
        --motion-filter-type none \
        --output-type censored \
        --output-run-wise-correlations n \
        --create-matrices all \
        --min-coverage 0.5 \
        --linc-qc y \
        --abcc-qc n \
        --warp-surfaces-native2std n \
        --fs-license-file "$FS_LICENSE_CONTAINER_PATH" \
        --nprocs "$NPROCS" --omp-nthreads "$OMP_NTHREADS" --mem-mb "$MEM_MB" \
        --work-dir /work \
        --notrack \
        > "$log_file" 2>&1
    local rc=$?

    if [ "$rc" -eq 0 ] && [ -f "$report" ]; then
        update_manifest "$row" DONE
        rm -rf "$work_dir"
        echo "[$bids_id] ($variant) done"
    else
        update_manifest "$row" FAILED
        echo "[error] $bids_id ($variant): XCP-D failed (exit $rc) -- see $log_file, work dir kept at $work_dir" >&2
    fi
}
export -f run_xcpd_variant

process_subject () {
    set -uo pipefail   # no -e: a failed apptainer run must fall through, not kill the parallel worker

    local bids_id=$1        # e.g. sub-dmxi
    local code=${bids_id#sub-}
    local fmriprep_report="$FMRIPREP_DIR/${bids_id}.html"

    if [ ! -f "$fmriprep_report" ]; then
        echo "[skip] $bids_id has no fMRIPrep report -- nothing for XCP-D to postprocess"
        update_manifest "${bids_id}_nogsr" NO_FMRIPREP
        update_manifest "${bids_id}_gsr" NO_FMRIPREP
        return 0
    fi

    run_xcpd_variant "$bids_id" "$code" nogsr acompcor "$OUT_DIR"
    run_xcpd_variant "$bids_id" "$code" gsr acompcor_gsr "$OUT_DIR_GSR"
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
echo "Rows DONE: $n_done / $(( ${#subjects[@]} * 2 ))"
if [ -n "$n_failed" ]; then
    echo "Rows currently FAILED (re-run this script to retry): $n_failed" >&2
fi
