#!/opt/homebrew/Caskroom/mambaforge/base/bin/python3
"""
Convert the ANTINOMICS Philips PAR/REC raw MRI export into a BIDS dataset.

Source : /Volumes/G_USZ_ORL$/Research/ANTINOMICS/data/mri/<ID>_antinomics/
Target : /Volumes/G_USZ_ORL$/Research/ANTINOMICS/data/mri_bids/sub-<ID>/...

Per subject, four series are pulled out of the raw export:
  - anat/T1w                 <- "t1w_3d" series (NOT "vt1w_3d")
  - anat/acq-highreshippo_T2w <- "3dt2_tra" series
  - func/task-rest run-1,2   <- the two "fmri" series, ordered by acquisition time
  - dwi                      <- the "dti_32" series

Conversion rule (per user instruction):
  - DWI is ALWAYS converted from .par/.rec with MRtrix3's `mrconvert`
    (it also exports .bval/.bvec). It is NOT run through fslreorient2std:
    doing so would require rotating .bvec by the same transform, which is
    extra risk for little benefit here, so DWI keeps its native orientation.
  - All other modalities (T1w, T2w-hippo, both fMRI runs) use a pre-existing
    sibling .nii if the scanner already reconstructed one; otherwise they are
    converted from .par/.rec with FSL's `parrec2nii`. Either way, the result
    is then passed through `fslreorient2std`, since Philips PAR/REC series
    can each land in a different acquisition-native orientation (e.g. T1w
    already close to standard, fMRI flipped relative to it).

Run with:  python3 convert_to_bids.py [--subjects sub1,sub2,...] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SRC_ROOT = Path("/Volumes/G_USZ_ORL$/Research/ANTINOMICS/data/mri")
BIDS_ROOT = Path("/Volumes/G_USZ_ORL$/Research/ANTINOMICS/data/mri_bids")
TASK_NAME = "rest"

SUBJECT_DIR_RE = re.compile(r"^([a-zA-Z]{4})_antinomics$")

T1W_RE = re.compile(r"_t1w_3d.*\.par$", re.IGNORECASE)
VT1W_MARKER_RE = re.compile(r"_vt1w_3d", re.IGNORECASE)
HIPPO_RE = re.compile(r"3dt2.?tra.*\.par$", re.IGNORECASE)
FMRI_RE = re.compile(r"_fmri\.par$", re.IGNORECASE)
DTI_RE = re.compile(r"_dti_?32\.par$", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class StepResult:
    subject: str
    modality: str
    status: str  # OK / WARNING / ERROR / SKIPPED
    method: str = ""
    source: str = ""
    target: str = ""
    message: str = ""


@dataclass
class Report:
    rows: list = field(default_factory=list)

    def add(self, r: StepResult):
        self.rows.append(r)
        tag = {"OK": "  OK ", "WARNING": " WARN", "ERROR": "ERROR", "SKIPPED": " SKIP"}[r.status]
        print(f"[{tag}] {r.subject:6s} {r.modality:22s} {r.method:12s} {r.message}")

    def write_csv(self, path: Path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["subject", "modality", "status", "method", "source", "target", "message"])
            for r in self.rows:
                w.writerow([r.subject, r.modality, r.status, r.method, r.source, r.target, r.message])

    def summary(self) -> str:
        n_ok = sum(1 for r in self.rows if r.status == "OK")
        n_warn = sum(1 for r in self.rows if r.status == "WARNING")
        n_err = sum(1 for r in self.rows if r.status == "ERROR")
        n_skip = sum(1 for r in self.rows if r.status == "SKIPPED")
        lines = [
            "",
            "=" * 70,
            "SUMMARY",
            "=" * 70,
            f"Total steps run : {len(self.rows)}",
            f"  OK            : {n_ok}",
            f"  WARNING       : {n_warn}",
            f"  ERROR         : {n_err}",
            f"  SKIPPED       : {n_skip}",
        ]
        if n_warn or n_err or n_skip:
            lines.append("")
            lines.append("Issues:")
            for r in self.rows:
                if r.status in ("WARNING", "ERROR", "SKIPPED"):
                    lines.append(f"  [{r.status}] {r.subject} / {r.modality}: {r.message}")
        lines.append("=" * 70)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# PAR header helpers
# --------------------------------------------------------------------------- #


def parse_par_header(par_path: Path) -> dict:
    info = {}
    text = par_path.read_text(errors="replace")
    patterns = {
        "ProtocolName": r"Protocol name\s*:\s*(.+)",
        "Technique": r"Technique\s*:\s*(.+)",
        "RepetitionTimeMs": r"Repetition time \[msec\]\s*:\s*([\d.]+)",
        "ExaminationDate": r"Examination date/time\s*:\s*(.+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            info[key] = m.group(1).strip()
    return info


def clean_subprocess_text(text: str) -> list:
    """Strip \\r progress-bar spam from mrtrix/fsl output, keep meaningful lines."""
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)  # ANSI escapes
    parts = re.split(r"[\r\n]+", text)
    lines = []
    for p in parts:
        p = p.strip()
        if not p or re.match(r"^mrconvert:\s*\[\s*\d+%\]", p):
            continue
        lines.append(p)
    # de-duplicate while preserving order
    seen = set()
    out = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


# --------------------------------------------------------------------------- #
# Conversion primitives
# --------------------------------------------------------------------------- #


def find_sibling_nii(par_path: Path) -> Path | None:
    nii = par_path.with_suffix(".nii")
    if nii.exists():
        return nii
    niigz = par_path.with_suffix(".nii.gz")
    if niigz.exists():
        return niigz
    return None


def write_json_sidecar(json_path: Path, par_path: Path, extra: dict, dry_run: bool):
    hdr = parse_par_header(par_path)
    payload = {
        "SeriesDescription": hdr.get("ProtocolName", par_path.stem),
        "ScanningSequence": hdr.get("Technique", ""),
    }
    if "RepetitionTimeMs" in hdr:
        payload["RepetitionTime"] = round(float(hdr["RepetitionTimeMs"]) / 1000.0, 6)
    payload.update(extra)
    if dry_run:
        return
    json_path.write_text(json.dumps(payload, indent=4) + "\n")


def convert_with_existing_nii(nii_src: Path, out_nii_gz: Path, dry_run: bool) -> tuple:
    if dry_run:
        return "existing_nii", []
    out_nii_gz.parent.mkdir(parents=True, exist_ok=True)
    if nii_src.suffix == ".gz":
        shutil.copyfile(nii_src, out_nii_gz)
    else:
        with open(nii_src, "rb") as fin:
            import gzip

            with gzip.open(out_nii_gz, "wb") as fout:
                shutil.copyfileobj(fin, fout)
    return "existing_nii", []


def convert_with_parrec2nii(par_path: Path, out_nii_gz: Path, dry_run: bool) -> tuple:
    if dry_run:
        return "parrec2nii", []
    out_nii_gz.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["parrec2nii", "-c", "-o", tmpdir, str(par_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        produced = list(Path(tmpdir).glob("*.nii.gz"))
        if proc.returncode != 0 or not produced:
            raise RuntimeError(
                f"parrec2nii failed (exit {proc.returncode}): "
                + " | ".join(clean_subprocess_text(proc.stdout + proc.stderr))
            )
        shutil.move(str(produced[0]), str(out_nii_gz))
        msgs = clean_subprocess_text(proc.stdout + proc.stderr)
        msgs = [m for m in msgs if "warn" in m.lower() or "error" in m.lower()]
        return "parrec2nii", msgs


def convert_with_mrconvert(par_path: Path, out_nii_gz: Path, out_bval: Path, out_bvec: Path, dry_run: bool) -> tuple:
    if dry_run:
        return "mrconvert", []
    out_nii_gz.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mrconvert",
        str(par_path),
        str(out_nii_gz),
        "-export_grad_fsl",
        str(out_bvec),
        str(out_bval),
        "-force",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    msgs = clean_subprocess_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not out_nii_gz.exists():
        raise RuntimeError(f"mrconvert failed (exit {proc.returncode}): " + " | ".join(msgs))
    warn_msgs = [m for m in msgs if "warn" in m.lower() or "error" in m.lower()]
    return "mrconvert", warn_msgs


def reorient_to_std(nii_gz: Path) -> list:
    """Run FSL's fslreorient2std in place. Not used for DWI, since that would
    require rotating the .bvec file by the same transform to stay consistent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / nii_gz.name
        proc = subprocess.run(
            ["fslreorient2std", str(nii_gz), str(tmp_out)], capture_output=True, text=True
        )
        msgs = clean_subprocess_text(proc.stdout + proc.stderr)
        if proc.returncode != 0 or not tmp_out.exists():
            raise RuntimeError(f"fslreorient2std failed (exit {proc.returncode}): " + " | ".join(msgs))
        shutil.move(str(tmp_out), str(nii_gz))
        return [m for m in msgs if "warn" in m.lower() or "error" in m.lower()]


def validate_nii(path: Path) -> str | None:
    """Return an error string if the NIfTI fails to load / looks empty, else None."""
    try:
        import nibabel as nib

        img = nib.load(str(path))
        shape = img.shape
        if any(s <= 0 for s in shape):
            return f"suspicious shape {shape}"
    except Exception as e:  # noqa: BLE001
        return f"failed to load with nibabel: {e}"
    return None


# --------------------------------------------------------------------------- #
# Per-subject processing
# --------------------------------------------------------------------------- #


def acquisition_sort_key(path: Path) -> str:
    m = re.search(r"_(\d{8}_\d{6})_", path.name)
    return m.group(1) if m else path.name


def process_t1w(sub_dir: Path, sub_id: str, out_dir: Path, report: Report, dry_run: bool):
    modality = "anat/T1w"
    candidates = [
        p for p in sub_dir.glob("*.par") if T1W_RE.search(p.name) and not VT1W_MARKER_RE.search(p.name)
    ]
    if len(candidates) != 1:
        report.add(StepResult(sub_id, modality, "ERROR", message=f"expected 1 t1w_3d .par, found {len(candidates)}"))
        return
    par = candidates[0]
    out_nii = out_dir / "anat" / f"sub-{sub_id}_T1w.nii.gz"
    out_json = out_dir / "anat" / f"sub-{sub_id}_T1w.json"
    _run_generic_modality(sub_id, modality, par, out_nii, out_json, {}, report, dry_run)


def process_hippo(sub_dir: Path, sub_id: str, out_dir: Path, report: Report, dry_run: bool):
    modality = "anat/T2w-hippo"
    candidates = [p for p in sub_dir.glob("*.par") if HIPPO_RE.search(p.name)]
    if len(candidates) != 1:
        report.add(StepResult(sub_id, modality, "ERROR", message=f"expected 1 3dt2_tra .par, found {len(candidates)}"))
        return
    par = candidates[0]
    out_nii = out_dir / "anat" / f"sub-{sub_id}_acq-highreshippo_T2w.nii.gz"
    out_json = out_dir / "anat" / f"sub-{sub_id}_acq-highreshippo_T2w.json"
    _run_generic_modality(sub_id, modality, par, out_nii, out_json, {}, report, dry_run)


def process_fmri(sub_dir: Path, sub_id: str, out_dir: Path, report: Report, dry_run: bool):
    candidates = sorted(
        [p for p in sub_dir.glob("*.par") if FMRI_RE.search(p.name)], key=acquisition_sort_key
    )
    if len(candidates) == 0:
        report.add(StepResult(sub_id, "func/bold", "ERROR", message="no fmri .par files found"))
        return
    if len(candidates) != 2:
        report.add(
            StepResult(
                sub_id,
                "func/bold",
                "WARNING",
                message=f"expected 2 fmri runs, found {len(candidates)} -- processing what is available",
            )
        )
    for i, par in enumerate(candidates, start=1):
        modality = f"func/run-{i}"
        out_nii = out_dir / "func" / f"sub-{sub_id}_task-{TASK_NAME}_run-{i}_bold.nii.gz"
        out_json = out_dir / "func" / f"sub-{sub_id}_task-{TASK_NAME}_run-{i}_bold.json"
        _run_generic_modality(sub_id, modality, par, out_nii, out_json, {"TaskName": TASK_NAME}, report, dry_run)


def _run_generic_modality(sub_id, modality, par, out_nii, out_json, extra_json, report, dry_run):
    nii_src = find_sibling_nii(par)
    try:
        if nii_src is not None:
            method, msgs = convert_with_existing_nii(nii_src, out_nii, dry_run)
            source = str(nii_src)
        else:
            method, msgs = convert_with_parrec2nii(par, out_nii, dry_run)
            source = str(par)

        if not dry_run:
            reorient_msgs = reorient_to_std(out_nii)
            msgs = msgs + reorient_msgs
            method = f"{method}+reorient2std"

        write_json_sidecar(out_json, par, extra_json, dry_run)

        status = "OK"
        message = f"{source} -> {out_nii}"
        if msgs:
            status = "WARNING"
            message += " | " + " || ".join(msgs)

        if not dry_run:
            err = validate_nii(out_nii)
            if err:
                status = "ERROR"
                message += f" | validation failed: {err}"

        report.add(StepResult(sub_id, modality, status, method, source, str(out_nii), message))
    except Exception as e:  # noqa: BLE001
        report.add(StepResult(sub_id, modality, "ERROR", source=str(par), message=str(e)))


def process_dwi(sub_dir: Path, sub_id: str, out_dir: Path, report: Report, dry_run: bool):
    modality = "dwi"
    candidates = [p for p in sub_dir.glob("*.par") if DTI_RE.search(p.name)]
    if len(candidates) != 1:
        report.add(StepResult(sub_id, modality, "ERROR", message=f"expected 1 dti_32 .par, found {len(candidates)}"))
        return
    par = candidates[0]
    out_nii = out_dir / "dwi" / f"sub-{sub_id}_dwi.nii.gz"
    out_bval = out_dir / "dwi" / f"sub-{sub_id}_dwi.bval"
    out_bvec = out_dir / "dwi" / f"sub-{sub_id}_dwi.bvec"
    out_json = out_dir / "dwi" / f"sub-{sub_id}_dwi.json"
    try:
        method, msgs = convert_with_mrconvert(par, out_nii, out_bval, out_bvec, dry_run)
        write_json_sidecar(out_json, par, {}, dry_run)

        status = "OK"
        message = f"{par} -> {out_nii}"
        if msgs:
            status = "WARNING"
            message += " | " + " || ".join(msgs)

        if not dry_run:
            err = validate_nii(out_nii)
            if err:
                status = "ERROR"
                message += f" | validation failed: {err}"
            elif not out_bval.exists() or not out_bvec.exists():
                status = "ERROR"
                message += " | bval/bvec not produced"

        report.add(StepResult(sub_id, modality, status, method, str(par), str(out_nii), message))
    except Exception as e:  # noqa: BLE001
        report.add(StepResult(sub_id, modality, "ERROR", source=str(par), message=str(e)))


# --------------------------------------------------------------------------- #
# Dataset-level scaffolding
# --------------------------------------------------------------------------- #


def write_dataset_scaffold(subjects: list, dry_run: bool):
    if dry_run:
        return
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    (BIDS_ROOT / "code").mkdir(exist_ok=True)

    dd = {
        "Name": "ANTINOMICS",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "convert_to_bids.py", "Description": "PAR/REC to BIDS conversion script"}],
    }
    (BIDS_ROOT / "dataset_description.json").write_text(json.dumps(dd, indent=4) + "\n")

    with open(BIDS_ROOT / "participants.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["participant_id"])
        for s in subjects:
            w.writerow([f"sub-{s}"])

    readme = (
        "ANTINOMICS MRI dataset, converted to BIDS from Philips PAR/REC raw export.\n\n"
        "anat/T1w                    <- t1w_3d series, reoriented to standard (fslreorient2std)\n"
        "anat/acq-highreshippo_T2w   <- 3dt2_tra series (high-res hippocampus), reoriented to standard\n"
        "func/task-rest run-1,run-2  <- the two fmri series, ordered by acquisition time, reoriented to standard\n"
        "dwi                         <- dti_32 series, converted with MRtrix3 mrconvert "
        "(bval/bvec exported alongside). NOT passed through fslreorient2std -- kept in native "
        "orientation to avoid having to rotate the bvecs.\n\n"
        "See code/convert_to_bids.py for the conversion script and "
        "code/conversion_report_*.csv for the per-subject conversion log.\n"
    )
    (BIDS_ROOT / "README").write_text(readme)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", help="Comma-separated list of 4-letter subject IDs to process (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="Discover and validate only, do not write/convert anything")
    args = ap.parse_args()

    if not SRC_ROOT.exists():
        print(f"ERROR: source root not found: {SRC_ROOT}", file=sys.stderr)
        sys.exit(1)

    all_subject_dirs = sorted(
        [p for p in SRC_ROOT.iterdir() if p.is_dir() and SUBJECT_DIR_RE.match(p.name)]
    )
    subj_filter = set(s.strip().lower() for s in args.subjects.split(",")) if args.subjects else None

    id_to_dir = {}
    for d in all_subject_dirs:
        sub_id = SUBJECT_DIR_RE.match(d.name).group(1).lower()
        if subj_filter and sub_id not in subj_filter:
            continue
        id_to_dir[sub_id] = d

    if not id_to_dir:
        print("No matching subject directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(id_to_dir)} subject(s) to process.")
    print(f"Source : {SRC_ROOT}")
    print(f"Target : {BIDS_ROOT}")
    print(f"Dry run: {args.dry_run}")
    print()

    write_dataset_scaffold(sorted(id_to_dir.keys()), args.dry_run)

    report = Report()
    for sub_id in sorted(id_to_dir.keys()):
        sub_dir = id_to_dir[sub_id]
        out_dir = BIDS_ROOT / f"sub-{sub_id}"
        process_t1w(sub_dir, sub_id, out_dir, report, args.dry_run)
        process_hippo(sub_dir, sub_id, out_dir, report, args.dry_run)
        process_fmri(sub_dir, sub_id, out_dir, report, args.dry_run)
        process_dwi(sub_dir, sub_id, out_dir, report, args.dry_run)

    print(report.summary())

    if not args.dry_run:
        code_dir = BIDS_ROOT / "code"
        code_dir.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).resolve(), code_dir / "convert_to_bids.py")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = code_dir / f"conversion_report_{ts}.csv"
        report.write_csv(report_path)
        print(f"\nFull per-step report written to: {report_path}")


if __name__ == "__main__":
    main()
