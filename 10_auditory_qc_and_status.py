#!/usr/bin/env python3
"""
Cross-tool pipeline readiness matrix + auditory-cortex ROI data-quality report
for the ANTINOMICS cohort.

Part A -- derivatives/pipeline_status.tsv
    Collapses each of the five upstream stage manifests (05_mriqc, 06_qsiprep,
    07_qsirecon, 08_fmriprep, 09_xcpd) to its current per-subject status (last
    row wins, by timestamp) and joins them into one row-per-subject table, plus
    a derived `ready_for_sfc` flag (qsirecon DONE + both xcp_d variants DONE --
    matched SC/FC parcellation, everything the later SFC module needs).

Part B -- derivatives/qc/auditory_roi_report.tsv
    Flagging-only QC (mirrors 05b_mriqc_group_report.py's precedent: surface the
    signal, don't act on it). This dataset runs with no fieldmap-based SDC, so
    susceptibility dropout right around the ear canal/petrous bone is a real,
    previously-undiagnosed risk specifically in the region this study cares
    about most. For every subject with fMRIPrep DONE, computes per-run tSNR and
    brain-mask coverage in a bilateral Heschl's-Gyrus + Planum-Temporale ROI
    (Harvard-Oxford cortical atlas, resampled onto fMRIPrep's own
    MNI152NLin6Asym:res-2 grid), split into bilateral/left/right, merged with
    QSIPrep's own native per-subject DWI QC scalars (image_qc.tsv -- QSIPrep
    does not run FSL's eddy_quad; this is its equivalent). Flags outliers via
    the same MAD-based modified z-score convention as 05b (threshold 3.5).

Run with:
    /home/ubuntu/volume/miniconda3/envs/tinnorm/bin/python3 10_auditory_qc_and_status.py
"""

from __future__ import annotations

import multiprocessing as mp
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

BIDS_ROOT = Path("/home/ubuntu/volume/antinomics")
DERIV = BIDS_ROOT / "derivatives"
LOGS = DERIV / "logs"

PARTICIPANTS_TSV = BIDS_ROOT / "participants.tsv"
EXCLUDED_TSV = DERIV / "excluded_subjects.tsv"

# (pipeline_status.tsv column name, manifest path) -- all share the
# subject / status / timestamp schema established in 05-08.
STAGE_MANIFESTS = {
    "mriqc": LOGS / "05_mriqc" / "mriqc_status.tsv",
    "qsiprep": LOGS / "06_qsiprep" / "qsiprep_status.tsv",
    "qsirecon": LOGS / "07_qsirecon" / "qsirecon_status.tsv",
    "fmriprep": LOGS / "08_fmriprep" / "fmriprep_status.tsv",
}
XCPD_MANIFEST = LOGS / "09_xcpd" / "xcpd_status.tsv"
XCPD_KEY_RE = re.compile(r"^(sub-\w+)_(nogsr|gsr)$")

PIPELINE_STATUS_TSV = DERIV / "pipeline_status.tsv"

QC_DIR = DERIV / "qc"
AUDITORY_REPORT_TSV = QC_DIR / "auditory_roi_report.tsv"

FMRIPREP_SPACE = "MNI152NLin6Asym_res-2"
DWI_QC_COLUMNS = [
    "mean_fd", "max_fd", "raw_num_bad_slices", "t1_neighbor_corr",
    "CNR0_mean", "CNR1_mean",
]

HO_ATLAS_NII = Path(
    "/home/ubuntu/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
)
HO_ATLAS_XML = Path("/home/ubuntu/fsl/data/atlases/HarvardOxford-Cortical.xml")
ROI_LABEL_SUBSTRINGS = ["Heschl", "Planum Temporale"]

FLAG_Z_THRESH = 3.5  # match 05b_mriqc_group_report.py's modified-z-score convention

_ROI_MASKS: dict[str, np.ndarray] | None = None  # set once, inherited by forked workers


# --------------------------------------------------------------------------
# Part A: pipeline_status.tsv
# --------------------------------------------------------------------------

def bids_subjects() -> list[str]:
    return sorted(pd.read_csv(PARTICIPANTS_TSV, sep="\t")["participant_id"])


def load_excluded() -> dict[str, bool]:
    if not EXCLUDED_TSV.exists():
        return {}
    df = pd.read_csv(EXCLUDED_TSV, sep="\t")
    return dict(zip(df["subject"], df["excluded"].astype(str).str.lower() == "yes"))


def collapse_manifest(path: Path) -> dict[str, str]:
    """Append-only subject/status/timestamp manifest -> {key: latest status}."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return {}
    last = df.sort_values("timestamp").groupby("subject", as_index=False).tail(1)
    return dict(zip(last["subject"], last["status"]))


def collapse_xcpd_manifest(path: Path) -> dict[str, dict[str, str]]:
    """xcp_d's manifest keys rows as sub-xxxx_{nogsr,gsr} -> split before use."""
    raw = collapse_manifest(path)
    out: dict[str, dict[str, str]] = {}
    for key, status in raw.items():
        m = XCPD_KEY_RE.match(key)
        if not m:
            continue
        subj, variant = m.groups()
        out.setdefault(subj, {})[variant] = status
    return out


def build_pipeline_status(
    subjects: list[str],
    excluded: dict[str, bool],
    stage_status: dict[str, dict[str, str]],
    xcpd_status: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for subj in subjects:
        row = {"subject": subj, "excluded": "yes" if excluded.get(subj) else "no"}
        for stage in ("mriqc", "qsiprep", "qsirecon", "fmriprep"):
            row[stage] = stage_status[stage].get(subj, "NOT_STARTED")
        xv = xcpd_status.get(subj, {})
        row["xcpd_nogsr"] = xv.get("nogsr", "NOT_STARTED")
        row["xcpd_gsr"] = xv.get("gsr", "NOT_STARTED")
        row["ready_for_sfc"] = (
            "yes"
            if row["qsirecon"] == "DONE" and row["xcpd_nogsr"] == "DONE" and row["xcpd_gsr"] == "DONE"
            else "no"
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    PIPELINE_STATUS_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PIPELINE_STATUS_TSV, sep="\t", index=False)
    print(f"Wrote pipeline status manifest: {PIPELINE_STATUS_TSV} ({len(df)} subjects)\n")

    for col in ("mriqc", "qsiprep", "qsirecon", "fmriprep", "xcpd_nogsr", "xcpd_gsr"):
        counts = df[col].value_counts().to_dict()
        print(f"  {col:10s}: {counts}")
    n_ready = (df["ready_for_sfc"] == "yes").sum()
    print(f"\n  ready_for_sfc: {n_ready}/{len(df)}")
    not_ready_not_excluded = df[(df["ready_for_sfc"] == "no") & (df["excluded"] == "no")]
    if len(not_ready_not_excluded):
        print(
            f"  Not-yet-ready, not excluded ({len(not_ready_not_excluded)}): "
            + ", ".join(not_ready_not_excluded["subject"])
        )
    return df


# --------------------------------------------------------------------------
# Part B: auditory_roi_report.tsv
# --------------------------------------------------------------------------

def parse_ho_label(xml_text: str, name_substring: str) -> tuple[int, str]:
    hits = []
    for m in re.finditer(r'<label index="(\d+)"[^>]*>([^<]+)</label>', xml_text):
        idx, text = int(m.group(1)), m.group(2)
        if name_substring.lower() in text.lower():
            hits.append((idx, text))
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one Harvard-Oxford label matching {name_substring!r}, found {hits}"
        )
    idx, text = hits[0]
    return idx + 1, text  # XML label index is 0-based; NIfTI voxel values are 1-based (0=background)


def split_hemispheres(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right split via the sign of each ROI voxel's MNI x-world-coordinate
    (positive x = right hemisphere in the RAS+ convention TemplateFlow spaces use)."""
    idx = np.argwhere(mask)
    left, right = np.zeros_like(mask), np.zeros_like(mask)
    if idx.size == 0:
        return left, right
    world_x = nib.affines.apply_affine(affine, idx)[:, 0]
    left_idx, right_idx = idx[world_x < 0], idx[world_x >= 0]
    if left_idx.size:
        left[tuple(left_idx.T)] = True
    if right_idx.size:
        right[tuple(right_idx.T)] = True
    return left, right


def get_reference_grid(subjects: list[str]) -> nib.Nifti1Image:
    for subj in subjects:
        func_dir = DERIV / "fmriprep" / subj / "func"
        candidates = sorted(func_dir.glob(f"{subj}_task-rest_run-*_space-{FMRIPREP_SPACE}_desc-brain_mask.nii.gz"))
        if candidates:
            return nib.load(candidates[0])
    raise RuntimeError("No fMRIPrep brain mask found in any subject to use as the reference grid")


def build_roi_masks(ref_img: nib.Nifti1Image) -> dict[str, np.ndarray]:
    xml_text = HO_ATLAS_XML.read_text()
    labels = [parse_ho_label(xml_text, s) for s in ROI_LABEL_SUBSTRINGS]

    atlas_img = nib.load(HO_ATLAS_NII)
    atlas_data = np.asarray(atlas_img.dataobj)
    present = set(np.unique(atlas_data).astype(int).tolist())
    for val, name in labels:
        if val not in present:
            raise RuntimeError(
                f"Expected label value {val} ({name!r}) not found in {HO_ATLAS_NII.name}; "
                f"present values: {sorted(present)}"
            )
    print("[auditory ROI] labels: " + ", ".join(f"{name!r}=value {val}" for val, name in labels))

    roi_atlas_space = np.isin(atlas_data, [val for val, _ in labels])
    if atlas_img.shape == ref_img.shape and np.allclose(atlas_img.affine, ref_img.affine):
        roi_ref_space = roi_atlas_space
    else:
        roi_img = nib.Nifti1Image(roi_atlas_space.astype(np.uint8), atlas_img.affine, atlas_img.header)
        resampled = resample_from_to(roi_img, ref_img, order=0)
        roi_ref_space = np.asarray(resampled.dataobj) > 0.5

    left, right = split_hemispheres(roi_ref_space, ref_img.affine)
    assert left.sum() + right.sum() == roi_ref_space.sum(), "L+R voxel counts must sum to the bilateral total"
    print(
        f"[auditory ROI] bilateral={roi_ref_space.sum()} voxels, "
        f"left={left.sum()}, right={right.sum()} (grid: {ref_img.shape})"
    )
    return {"bilateral": roi_ref_space, "left": left, "right": right}


def load_dwi_qc(subject: str) -> dict[str, float]:
    path = DERIV / "qsiprep" / subject / "dwi" / f"{subject}_space-ACPC_desc-image_qc.tsv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return {}
    row = df.iloc[0]
    return {f"dwi_{c}": float(row[c]) for c in DWI_QC_COLUMNS if c in row}


def run_qc(bold_path: Path, mask_path: Path, roi_masks: dict[str, np.ndarray]) -> dict[str, float]:
    bold = nib.load(bold_path).get_fdata(dtype=np.float32)
    brain_mask = np.asarray(nib.load(mask_path).dataobj) > 0.5
    mean_t = bold.mean(axis=3)
    std_t = bold.std(axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr_vol = np.where(std_t > 0, mean_t / std_t, np.nan)

    out = {}
    for roi_name, roi_mask in roi_masks.items():
        sel = roi_mask & brain_mask
        out[f"tsnr_{roi_name}"] = float(np.nanmean(tsnr_vol[sel])) if sel.any() else np.nan
        out[f"coverage_{roi_name}"] = float(sel.sum()) / float(roi_mask.sum()) if roi_mask.sum() else np.nan
    return out


def process_subject(subject: str) -> list[dict]:
    assert _ROI_MASKS is not None
    func_dir = DERIV / "fmriprep" / subject / "func"
    dwi_qc = load_dwi_qc(subject)
    bold_files = sorted(func_dir.glob(f"{subject}_task-rest_run-*_space-{FMRIPREP_SPACE}_desc-preproc_bold.nii.gz")) \
        if func_dir.exists() else []
    if not bold_files:
        return [{"subject": subject, "run": None, **dwi_qc}]

    rows = []
    for bold_path in bold_files:
        run_match = re.search(r"run-(\d+)", bold_path.name)
        run_id = f"run-{run_match.group(1)}" if run_match else "run-unknown"
        mask_path = Path(str(bold_path).replace("desc-preproc_bold", "desc-brain_mask"))
        qc = run_qc(bold_path, mask_path, _ROI_MASKS)
        rows.append({"subject": subject, "run": run_id, **qc, **dwi_qc})
    return rows


def compute_modified_z_flags(series: pd.Series) -> pd.Series:
    x = series.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if not mad or np.isnan(mad):
        return pd.Series(False, index=x.index)
    z = 0.6745 * (x - med) / mad
    return z.abs() > FLAG_Z_THRESH


def build_auditory_qc(subjects: list[str], excluded: dict[str, bool]) -> pd.DataFrame:
    global _ROI_MASKS
    ref_img = get_reference_grid(subjects)
    _ROI_MASKS = build_roi_masks(ref_img)

    with mp.Pool(processes=6) as pool:
        results = pool.map(process_subject, subjects)
    rows = [row for subject_rows in results for row in subject_rows]

    df = pd.DataFrame(rows)
    df["excluded"] = df["subject"].map(lambda s: "yes" if excluded.get(s) else "no")

    metric_cols = [c for c in df.columns if c.startswith(("tsnr_", "coverage_", "dwi_"))]
    flag_cols = []
    for col in metric_cols:
        flag_col = f"flag_{col}"
        df[flag_col] = compute_modified_z_flags(df[col])
        flag_cols.append(flag_col)
    df["any_flag"] = df[flag_cols].any(axis=1) if flag_cols else False

    QC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDITORY_REPORT_TSV, sep="\t", index=False)
    print(f"\nWrote auditory ROI QC report: {AUDITORY_REPORT_TSV} ({len(df)} rows)")

    flagged_subjects = sorted(df.loc[df["any_flag"], "subject"].unique())
    print(
        f"Subjects with >=1 flagged auditory-ROI/DWI metric "
        f"(|modified z| > {FLAG_Z_THRESH}): {len(flagged_subjects)}"
    )
    if flagged_subjects:
        print("  " + ", ".join(flagged_subjects))
    return df


# --------------------------------------------------------------------------

def main() -> None:
    subjects = bids_subjects()
    excluded = load_excluded()
    stage_status = {name: collapse_manifest(path) for name, path in STAGE_MANIFESTS.items()}
    xcpd_status = collapse_xcpd_manifest(XCPD_MANIFEST)

    build_pipeline_status(subjects, excluded, stage_status, xcpd_status)

    fmriprep_done = [s for s in subjects if stage_status["fmriprep"].get(s) == "DONE"]
    print(f"\n{len(fmriprep_done)}/{len(subjects)} subjects have fMRIPrep DONE -- computing auditory ROI QC.\n")
    if fmriprep_done:
        build_auditory_qc(fmriprep_done, excluded)


if __name__ == "__main__":
    main()
