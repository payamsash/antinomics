#!/bin/bash
#
# Fill in missing FreeSurfer structural-segmentation steps for the ANTINOMICS cohort.
#
# Source : /home/ubuntu/volume/antinomics/sub-<id>/anat/  (BIDS T1w + high-res hippocampal T2w)
# Target : /home/ubuntu/volume/Tinception/subjects_fs_dir/<id>/ (shared FreeSurfer subjects dir)
#
# Idempotent, per-step: reads the manifest produced by 01_check_freesurfer_status.py
# and, for each subject, runs only the steps that are missing (MISSING/NO_DIR). It
# never re-runs a step already marked OK. Safe to re-run after a partial failure.
#
# Steps (matching the deleted ant_01_reconostruct.sh / ant_02_subsegment.sh):
#   recon-all -> segmentHA_T2.sh -> segmentThalamicNuclei.sh -> segmentBS.sh ->
#   segmentAAN.sh -> mri_segment_hypothalamic_subunits -> Choi/Buckner striatum+
#   cerebellum masks (mri_vol2vol) -> Schaefer 100/200/400/800/1000 (mris_ca_label +
#   mri_aparc2aseg) -> Glasser/HCPMMP1 (mri_surf2surf + mri_aparc2aseg).
#
# Run with:
#   ./02_run_freesurfer_pipeline.sh                 # process every subject missing something
#   ./02_run_freesurfer_pipeline.sh sub-dmxi sub-cwax   # process only these subjects
#
# Before running: `python3 01_check_freesurfer_status.py` to refresh the manifest.

set -euo pipefail

export FREESURFER_HOME=/usr/local/freesurfer/8.0.0
export SUBJECTS_DIR=/home/ubuntu/volume/Tinception/subjects_fs_dir
export LD_LIBRARY_PATH=$FREESURFER_HOME/MCRv97/runtime/glnxa64:$FREESURFER_HOME/MCRv97/bin/glnxa64:$FREESURFER_HOME/MCRv97/sys/os/glnxa64:$FREESURFER_HOME/MCRv97/extern/bin/glnxa64

# SetUpFreeSurfer.sh references unset vars (e.g. FREESURFER_FSPYTHON), and pipes
# things like `grep ... | wc -l` where a no-match grep exits 1 — both incompatible
# with `set -euo pipefail`. Relax all of it just for the source.
set +euo pipefail
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
set -euo pipefail

export PATH=/usr/lib/mrtrix3/bin:$PATH
export PATH=/home/ubuntu/fsl/bin:$PATH
export PATH=/home/ubuntu/data/src_codes/ants-2.5.4/bin:$PATH
export ANTSPATH=/home/ubuntu/data/src_codes/ants-2.5.4/bin

BIDS_ROOT=/home/ubuntu/volume/antinomics
MANIFEST="$BIDS_ROOT/derivatives/freesurfer_status.tsv"
N_JOBS=6          # subjects processed in parallel; recon-all alone is multi-threaded,
                  # so this is deliberately well under the 30 cores available.
N_THREADS_PER_JOB=4

sch_gcs_dir="$FREESURFER_HOME/gcs"

if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found at $MANIFEST — run 01_check_freesurfer_status.py first." >&2
    exit 1
fi

process_subject () {
    # GNU parallel runs this in a fresh bash -c that does NOT inherit the
    # top-level `set -euo pipefail` (only exported vars/functions cross over,
    # not shell options) — without this, a failed step here does not stop the
    # subject: every later step keeps running against missing/broken input and
    # still logs "completed".
    set -euo pipefail

    # bash cannot export arrays into a subshell's environment (GNU parallel
    # spawns a fresh `bash -c` per job), so these must be declared here,
    # not exported from the top-level script scope.
    local mask_options=("Tight" "Loose")
    local hemis=("lh" "rh")

    local bids_id=$1   # e.g. sub-dmxi
    local subject_id=${bids_id#sub-}   # e.g. dmxi
    local anat_dir="$BIDS_ROOT/$bids_id/anat"
    local t1="$anat_dir/${bids_id}_T1w.nii.gz"
    local t2="$anat_dir/${bids_id}_acq-highreshippo_T2w.nii.gz"
    local subj_dir="$SUBJECTS_DIR/$subject_id"
    local log_file="$subj_dir/scripts/pipeline_fill_in.log"

    echo "=== $subject_id ==="

    # --- recon-all ---
    if [ ! -f "$subj_dir/mri/aparc+aseg.mgz" ]; then
        if [ -d "$subj_dir" ]; then
            # partial/interrupted run already converted input — recon-all
            # refuses -i on an existing subject dir, so resume without it
            echo "[$subject_id] resuming recon-all -all (existing partial subject dir, no -i)"
            recon-all -s "$subject_id" -all -threads "$N_THREADS_PER_JOB"
        else
            echo "[$subject_id] running recon-all -all (fresh)"
            recon-all -s "$subject_id" -i "$t1" -all -threads "$N_THREADS_PER_JOB"
        fi
    fi
    mkdir -p "$subj_dir/scripts"
    echo "$(date): pipeline fill-in started for $subject_id" >> "$log_file"

    # --- hippocampus + amygdala subfields (needs high-res T2) ---
    if [ ! -f "$subj_dir/mri/lh.hippoAmygLabels-T1-T2.v22.mgz" ]; then
        echo "[$subject_id] running segmentHA_T2.sh"
        segmentHA_T2.sh "$subject_id" "$t2" "T2" 1
        echo "$(date): hippo/amygdala segmentation completed" >> "$log_file"
    fi

    # --- thalamic nuclei ---
    if [ ! -f "$subj_dir/mri/ThalamicNuclei.v13.T1.mgz" ]; then
        echo "[$subject_id] running segmentThalamicNuclei.sh"
        segmentThalamicNuclei.sh "$subject_id"
        echo "$(date): thalamic nuclei segmentation completed" >> "$log_file"
    fi

    # --- brainstem substructures ---
    if [ "${SKIP_BS_AAN_HYPO:-0}" = "1" ]; then
        :
    elif [ ! -f "$subj_dir/mri/brainstemSsLabels.v13.mgz" ]; then
        echo "[$subject_id] running segmentBS.sh"
        segmentBS.sh "$subject_id"
        echo "$(date): brainstem segmentation completed" >> "$log_file"
    fi

    # --- ascending arousal network ---
    if [ "${SKIP_BS_AAN_HYPO:-0}" = "1" ]; then
        :
    elif [ ! -f "$subj_dir/mri/arousalNetworkLabels.v10.mgz" ]; then
        echo "[$subject_id] running segmentAAN.sh"
        segmentAAN.sh "$subject_id"
        echo "$(date): AAN segmentation completed" >> "$log_file"
    fi

    # --- hypothalamic subunits ---
    if [ "${SKIP_BS_AAN_HYPO:-0}" = "1" ]; then
        :
    elif [ ! -f "$subj_dir/mri/hypothalamic_subunits_seg.v1.mgz" ]; then
        echo "[$subject_id] running mri_segment_hypothalamic_subunits"
        mri_segment_hypothalamic_subunits --s "$subject_id" --threads "$N_THREADS_PER_JOB"
        echo "$(date): hypothalamus segmentation completed" >> "$log_file"
    fi

    # --- Choi/Buckner striatum + cerebellum functional masks ---
    if [ "${SKIP_STRIATUM:-0}" = "1" ]; then
        : # skipped for this run (SKIP_STRIATUM=1)
    elif [ ! -f "$subj_dir/mri/striatum_17_network_Tight_mask.nii.gz" ]; then
        echo "[$subject_id] running striatum/cerebellum vol2vol masks"
        for mask_option in "${mask_options[@]}"; do
            mri_vol2vol --mov "$subj_dir/mri/norm.mgz" \
                        --s "$subject_id" \
                        --targ "$SUBJECTS_DIR/MNI152/choi_atlas/17_network_${mask_option}_mask.nii.gz" \
                        --m3z "$SUBJECTS_DIR/MNI152/mri/transforms/talairach.m3z" \
                        --noDefM3zPath \
                        --o "$subj_dir/mri/striatum_17_network_${mask_option}_mask.nii.gz" \
                        --inv-morph --interp nearest

            mri_vol2vol --mov "$subj_dir/mri/norm.mgz" \
                        --s "$subject_id" \
                        --targ "$SUBJECTS_DIR/MNI152/buckner_atlas/17_network_${mask_option}_mask.nii.gz" \
                        --m3z "$SUBJECTS_DIR/MNI152/mri/transforms/talairach.m3z" \
                        --noDefM3zPath \
                        --o "$subj_dir/mri/cerebellum_17_network_${mask_option}_mask.nii.gz" \
                        --inv-morph --interp nearest
        done
        echo "$(date): striatum/cerebellum masks completed" >> "$log_file"
    fi

    # --- Schaefer 2018 parcellation ---
    if [ ! -f "$subj_dir/mri/Schaefer2018_400_7Networks.mgz" ]; then
        echo "[$subject_id] running Schaefer2018 parcellation"
        for n in 100 200 400 800 1000; do
            for hemi in "${hemis[@]}"; do
                mris_ca_label -l "$subj_dir/label/${hemi}.cortex.label" \
                                "$subject_id" "$hemi" \
                                "$subj_dir/surf/${hemi}.sphere.reg" \
                                "$sch_gcs_dir/${hemi}.Schaefer2018_${n}Parcels_7Networks.gcs" \
                                "$subj_dir/label/${hemi}.Schaefer2018_${n}Parcels_7Networks_order.annot"
            done
            mri_aparc2aseg --s "$subject_id" \
                            --annot "Schaefer2018_${n}Parcels_7Networks_order" \
                            --o "$subj_dir/mri/Schaefer2018_${n}_7Networks.mgz"
        done
        echo "$(date): Schaefer parcellation completed" >> "$log_file"
    fi

    # --- Glasser / HCPMMP1 ---
    if [ ! -f "$subj_dir/mri/HCPMMP1.mgz" ]; then
        echo "[$subject_id] running Glasser/HCPMMP1 parcellation"
        for hemi in "${hemis[@]}"; do
            mri_surf2surf --srcsubject fsaverage \
                            --trgsubject "$subject_id" \
                            --hemi "$hemi" \
                            --sval-annot "$SUBJECTS_DIR/fsaverage/label/${hemi}.HCPMMP1.annot" \
                            --tval "$subj_dir/label/${hemi}.HCPMMP1.annot"
        done
        mri_aparc2aseg --s "$subject_id" \
                        --annot HCPMMP1 \
                        --o "$subj_dir/mri/HCPMMP1.mgz"
        echo "$(date): Glasser/HCPMMP1 parcellation completed" >> "$log_file"
    fi

    echo "$(date): pipeline fill-in finished for $subject_id" >> "$log_file"
    echo "=== $subject_id done ==="
}
export -f process_subject
export SUBJECTS_DIR BIDS_ROOT sch_gcs_dir N_THREADS_PER_JOB

if [ "$#" -gt 0 ]; then
    subjects=("$@")
else
    # every subject with at least one MISSING/NO_DIR marker in the manifest
    mapfile -t subjects < <(tail -n +2 "$MANIFEST" | awk -F'\t' '{for(i=2;i<=NF;i++) if ($i!="OK") {print $1; break}}')
fi

echo "Subjects to process (${#subjects[@]}): ${subjects[*]}"
printf '%s\n' "${subjects[@]}" | parallel --jobs "$N_JOBS" process_subject {}

echo "Done. Re-run 01_check_freesurfer_status.py to confirm all markers are OK."
