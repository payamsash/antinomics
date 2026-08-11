#!/bin/bash
#
# Per-subject fMRIPrep: minimal preprocessing of anatomical (T1w, reusing the
# existing FreeSurfer recon -- no recon-all re-run) and both rs-fMRI BOLD
# runs (task-rest_run-1/run-2), each preprocessed independently per
# fMRIPrep's normal per-run logic. No SDC -- there are no fieldmaps in this
# dataset (confirmed in step 2 research), so this is left as fMRIPrep's
# natural default (no --use-syn-sdc), matching the same no-SDC stance as
# 06_run_qsiprep.sh.
#
# Output spaces: MNI152NLin2009cAsym, MNI152NLin6Asym:res-2, fsnative,
# fsaverage5. NIfTI-only for this phase -- no --cifti-output (doubles
# runtime/storage; revisit if a later gradient/SFC phase needs grayordinates).
#
# FreeSurfer subjects dir: unlike 07_run_qsirecon.sh (which only reads the
# recon), fMRIPrep's --fs-subjects-dir must be WRITABLE -- confirmed via two
# smoke-test crashes (2026-08-11): its fsdir_run node copytree's fsaverage/
# fsaverage5 templates into the subjects dir root, and its surface pipeline
# writes derived per-subject files (e.g. lh.midthickness) directly into
# sub-<id>/surf/. The real recon lives in a lab-wide shared directory
# (Tinception/subjects_fs_dir, ~600 other projects' subjects) that must
# never be made writable. Fix: derivatives/freesurfer_writable/sub-<id> holds
# real (dereferenced, not symlinked) per-subject copies that fMRIPrep is free
# to modify -- staged on demand below if not already present (idempotent;
# the initial 76-subject batch was seeded in bulk from copies already made
# for the QSIRecon cluster-migration staging, an `mv` since same filesystem).
#
# CPU-only -- fMRIPrep has no GPU-accelerated steps by default, so no --nv
# (unlike 06_run_qsiprep.sh, which requests --nv even though eddy_cpu ends
# up being used anyway).
#
# Excluded subjects (derivatives/excluded_subjects.tsv) are skipped by
# default, same convention as 06/07. Subjects without func/ data are also
# skipped (NO_FUNC status) -- fMRIPrep's BOLD stage has nothing to do for them.
#
# Idempotent, per-subject: completion marker is fMRIPrep's own top-level
# derivatives/fmriprep/sub-<id>.html report ("bids" output-layout, the
# default, places it directly in the output dir -- unlike QSIRecon's nested
# pipeline-name subfolder). NOT yet empirically confirmed against a real run
# -- smoke test before the full batch, same discipline as 06/07 (both of
# which caught real bugs this way).
#
# Run with:
#   ./08_run_fmriprep.sh                   # every BIDS subject not excluded
#   ./08_run_fmriprep.sh sub-dmxi sub-cwax # only these subjects (bypasses exclusion list)

set -euo pipefail

VOLUME=/home/ubuntu/volume
BIDS_ROOT="$VOLUME/antinomics"
FS_WRITABLE_DIR="$BIDS_ROOT/derivatives/freesurfer_writable"
OUT_DIR="$BIDS_ROOT/derivatives/fmriprep"
WORK_ROOT="$VOLUME/work/fmriprep"
LOG_DIR="$BIDS_ROOT/derivatives/logs/08_fmriprep"
MANIFEST="$LOG_DIR/fmriprep_status.tsv"
EXCLUDE_FILE="$BIDS_ROOT/derivatives/excluded_subjects.tsv"

APPTAINER_BIN="$VOLUME/miniconda3/envs/apptainer/bin/apptainer"
export APPTAINER_CACHEDIR="$VOLUME/apptainer_cache"
export APPTAINER_TMPDIR="$VOLUME/apptainer_tmp"
TEMPLATEFLOW_CACHE="$VOLUME/templateflow_cache"
FMRIPREP_SIF="$VOLUME/containers/fmriprep-25.2.5.sif"
FS_LICENSE_CONTAINER_PATH="/data/derivatives/freesurfer_license.txt"

OUTPUT_SPACES="MNI152NLin2009cAsym MNI152NLin6Asym:res-2 fsnative fsaverage5"

N_JOBS=4             # CPU-bound, no GPU steps -- same conservative budget as 06/07
NPROCS=6
OMP_NTHREADS=4
MEM_MB=18000          # x4 concurrent = 72GB of 117GB total, leaves headroom

mkdir -p "$OUT_DIR" "$WORK_ROOT" "$LOG_DIR" "$FS_WRITABLE_DIR"

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
       FMRIPREP_SIF BIDS_ROOT FS_WRITABLE_DIR OUT_DIR WORK_ROOT LOG_DIR \
       NPROCS OMP_NTHREADS MEM_MB FS_LICENSE_CONTAINER_PATH OUTPUT_SPACES

process_subject () {
    set -uo pipefail   # no -e: a failed apptainer run must fall through to the
                       # FAILED branch below, not kill the parallel worker

    local bids_id=$1        # e.g. sub-dmxi
    local code=${bids_id#sub-}
    local work_dir="$WORK_ROOT/$bids_id"
    local log_file="$LOG_DIR/${bids_id}.log"
    local report="$OUT_DIR/${bids_id}.html"

    if [ -f "$report" ]; then
        echo "[skip] $bids_id already has an fMRIPrep report"
        update_manifest "$bids_id" DONE
        return 0
    fi

    if [ ! -e "$BIDS_ROOT/$bids_id/func" ]; then
        echo "[skip] $bids_id has no func/ -- nothing for fMRIPrep to do"
        update_manifest "$bids_id" NO_FUNC
        return 0
    fi

    local fs_writable_subject_dir="$FS_WRITABLE_DIR/$bids_id"
    if [ ! -d "$fs_writable_subject_dir" ]; then
        local fs_src
        fs_src=$(readlink -f "$BIDS_ROOT/derivatives/freesurfer/$bids_id" 2>/dev/null || true)
        if [ -z "$fs_src" ] || [ ! -d "$fs_src" ]; then
            echo "[skip] $bids_id has no FreeSurfer recon -- nothing for fMRIPrep's anatomical stage to reuse"
            update_manifest "$bids_id" NO_FREESURFER
            return 0
        fi
        echo "[$bids_id] staging a writable FreeSurfer copy (one-time, ~320MB)"
        cp -rL "$fs_src" "${fs_writable_subject_dir}.tmp"
        mv "${fs_writable_subject_dir}.tmp" "$fs_writable_subject_dir"
    fi

    echo "[$bids_id] running fMRIPrep"
    update_manifest "$bids_id" RUNNING
    mkdir -p "$work_dir"

    "$APPTAINER_BIN" run --cleanenv \
        -B "$BIDS_ROOT":/data:ro \
        -B "$FS_WRITABLE_DIR":/fs_subjects_dir \
        -B "$OUT_DIR":/out \
        -B "$work_dir":/work \
        -B "$TEMPLATEFLOW_CACHE":/opt/templateflow \
        --env TEMPLATEFLOW_HOME=/opt/templateflow \
        "$FMRIPREP_SIF" \
        /data /out participant \
        --participant-label "$code" \
        --output-spaces $OUTPUT_SPACES \
        --fs-subjects-dir /fs_subjects_dir \
        --fs-license-file "$FS_LICENSE_CONTAINER_PATH" \
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
        echo "[error] $bids_id: fMRIPrep failed (exit $rc) -- see $log_file, work dir kept at $work_dir" >&2
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
