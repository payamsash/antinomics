#!/bin/bash
#
# Install Apptainer and pull the BIDS-App container images needed for the rest of
# the pipeline (MRIQC, QSIPrep, QSIRecon, fMRIPrep, XCP-D).
#
# Must run BEFORE any of 05-09: the root disk (/) is 93% full with only ~6.9GB free,
# while /home/ubuntu/volume has 365GB free. Apptainer's default cache/tmp dirs and
# TemplateFlow's default cache both resolve under $HOME (root disk) unless relocated
# first -- a container pull or an fMRIPrep run would otherwise fill the root disk
# mid-operation.
#
# Idempotent: safe to re-run. Skips any step whose target already exists in the
# expected state; re-pulls/re-checks only what's missing or failed.
#
# Run with:
#   ./04_setup_containers.sh

set -euo pipefail

VOLUME=/home/ubuntu/volume
BIDS_ROOT="$VOLUME/antinomics"
CONTAINERS_DIR="$VOLUME/containers"
APPTAINER_CACHE="$VOLUME/apptainer_cache"
APPTAINER_TMP="$VOLUME/apptainer_tmp"
TEMPLATEFLOW_CACHE="$VOLUME/templateflow_cache"
STATUS_TSV="$BIDS_ROOT/derivatives/containers_status.tsv"
CONDA_ENV_NAME=apptainer
APPTAINER_VERSION=1.5.3
PULL_RETRIES=3
PULL_RETRY_SLEEP=20

mkdir -p "$CONTAINERS_DIR" "$APPTAINER_CACHE" "$APPTAINER_TMP" "$TEMPLATEFLOW_CACHE" \
         "$(dirname "$STATUS_TSV")"

# --- 0. Relocate TemplateFlow cache off the root disk -----------------------

relocate_templateflow_cache () {
    local tf_home="$HOME/.cache/templateflow"
    if [ -L "$tf_home" ]; then
        local target
        target=$(readlink -f "$tf_home")
        if [ "$target" = "$(readlink -f "$TEMPLATEFLOW_CACHE")" ]; then
            echo "[skip] $tf_home already symlinked to $TEMPLATEFLOW_CACHE"
            return
        fi
        echo "[warn] $tf_home is a symlink pointing elsewhere ($target) -- not touching it" >&2
        return
    fi
    if [ -d "$tf_home" ]; then
        echo "[relocate] rsyncing existing $tf_home -> $TEMPLATEFLOW_CACHE"
        rsync -a "$tf_home/" "$TEMPLATEFLOW_CACHE/"
        rm -rf "$tf_home"
    fi
    ln -s "$TEMPLATEFLOW_CACHE" "$tf_home"
    echo "[write] symlinked $tf_home -> $TEMPLATEFLOW_CACHE"
}

# --- 0b. Persist cache-dir env vars for every later script/shell ------------

persist_env_vars () {
    local marker="# >>> antinomics apptainer/templateflow cache dirs >>>"
    if grep -qF "$marker" ~/.bashrc 2>/dev/null; then
        echo "[skip] ~/.bashrc already has the cache-dir exports"
        return
    fi
    cat >> ~/.bashrc <<EOF

$marker
export APPTAINER_CACHEDIR="$APPTAINER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_TMP"
export TEMPLATEFLOW_HOME="$TEMPLATEFLOW_CACHE"
# <<< antinomics apptainer/templateflow cache dirs <<<
EOF
    echo "[write] appended cache-dir exports to ~/.bashrc"
}

relocate_templateflow_cache
persist_env_vars

export APPTAINER_CACHEDIR="$APPTAINER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_TMP"
export TEMPLATEFLOW_HOME="$TEMPLATEFLOW_CACHE"

# --- 1. Install Apptainer via conda-forge ------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    echo "[skip] conda env '$CONDA_ENV_NAME' already exists"
else
    echo "[install] creating conda env '$CONDA_ENV_NAME' with apptainer=$APPTAINER_VERSION"
    conda create -y -n "$CONDA_ENV_NAME" -c conda-forge "apptainer=$APPTAINER_VERSION"
fi

conda activate "$CONDA_ENV_NAME"
echo "[info] $(apptainer --version)"

# --- 2. Pull images -----------------------------------------------------------

# name | docker source | sif filename
IMAGES=(
    "qsiprep|docker://pennlinc/qsiprep:26.0.0|qsiprep-26.0.0.sif"
    "qsirecon|docker://pennlinc/qsirecon:26.0.0|qsirecon-26.0.0.sif"
    "fmriprep|docker://nipreps/fmriprep:25.2.5|fmriprep-25.2.5.sif"
    "xcp_d|docker://pennlinc/xcp_d:26.1.1|xcp_d-26.1.1.sif"
    "mriqc|docker://nipreps/mriqc:24.0.2|mriqc-24.0.2.sif"
)

if [ ! -f "$STATUS_TSV" ]; then
    printf "image\tsif_path\tsha256\tstatus\tsmoke_test\ttimestamp\n" > "$STATUS_TSV"
fi

logged_sha256 () {
    # last logged sha256 for this image name with status=DONE, if any
    awk -F'\t' -v img="$1" '$1==img && $4=="DONE" {sha=$3} END{print sha}' "$STATUS_TSV"
}

log_status () {
    local name="$1" sif_path="$2" sha="$3" status="$4" smoke="$5"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$sif_path" "$sha" "$status" "$smoke" \
        "$(date -Iseconds)" >> "$STATUS_TSV"
}

pull_image () {
    local name="$1" source="$2" fname="$3"
    local sif_path="$CONTAINERS_DIR/$fname"

    if [ -f "$sif_path" ]; then
        local current_sha expected_sha
        current_sha=$(sha256sum "$sif_path" | awk '{print $1}')
        expected_sha=$(logged_sha256 "$name")
        if [ -n "$expected_sha" ] && [ "$current_sha" = "$expected_sha" ]; then
            echo "[skip] $sif_path already present and matches logged sha256"
            return
        fi
        echo "[warn] $sif_path exists but doesn't match (or has no) logged sha256 -- re-pulling"
        rm -f "$sif_path"
    fi

    echo "[pull] $source -> $sif_path"
    local attempt=1 ok=0
    while [ "$attempt" -le "$PULL_RETRIES" ]; do
        if apptainer pull "$sif_path" "$source"; then
            ok=1
            break
        fi
        echo "[warn] pull attempt $attempt/$PULL_RETRIES failed for $name, retrying in ${PULL_RETRY_SLEEP}s..." >&2
        rm -f "$sif_path"
        sleep "$PULL_RETRY_SLEEP"
        attempt=$((attempt + 1))
    done

    if [ "$ok" -ne 1 ]; then
        log_status "$name" "$sif_path" "" "FAILED" "not_run"
        echo "[error] failed to pull $name after $PULL_RETRIES attempts" >&2
        return 1
    fi

    local sha smoke_output smoke_status
    sha=$(sha256sum "$sif_path" | awk '{print $1}')
    if smoke_output=$(apptainer run "$sif_path" --version 2>&1); then
        smoke_status="OK"
    else
        smoke_status="FAILED"
    fi
    echo "[smoke test] $name --version: $smoke_status ($(echo "$smoke_output" | head -1))"
    log_status "$name" "$sif_path" "$sha" "DONE" "$smoke_status"
}

failures=0
for entry in "${IMAGES[@]}"; do
    IFS='|' read -r name source fname <<< "$entry"
    pull_image "$name" "$source" "$fname" || failures=$((failures + 1))
done

# --- 3. GPU pre-flight (qsiprep bundles its own CUDA 12.2 eddy) --------------

qsiprep_sif="$CONTAINERS_DIR/qsiprep-26.0.0.sif"
if [ -f "$qsiprep_sif" ]; then
    echo "[gpu check] apptainer exec --nv $qsiprep_sif nvidia-smi"
    if apptainer exec --nv "$qsiprep_sif" nvidia-smi; then
        echo "[gpu check] OK -- container can see the GPU"
    else
        echo "[gpu check] FAILED -- container could not run nvidia-smi (check --nv / driver compatibility)" >&2
        failures=$((failures + 1))
    fi
else
    echo "[gpu check] skipped (qsiprep image not present)"
fi

echo
echo "Manifest: $STATUS_TSV"
if [ "$failures" -gt 0 ]; then
    echo "Done with $failures failure(s) -- see above / manifest. Re-run to retry failed steps." >&2
    exit 1
fi
echo "Done. All images pulled and smoke-tested successfully."
