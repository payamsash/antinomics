#!/usr/bin/env python3
"""
Make the ANTINOMICS dataset BIDS-valid and materialize derivatives infra needed
before MRIQC/QSIPrep/fMRIPrep/XCP-D can run (Step 2, item 3 of the pipeline plan).

This script is idempotent and safe to re-run: every write is existence/content
checked first, nothing is silently overwritten.

What it does:
  1. Writes dataset_description.json, .bidsignore, README, CHANGES at the BIDS root
     (currently absent -> dataset fails bids-validator/BIDS-App strict checks as-is).
  2. Writes participants.tsv + participants.json. Without --demographics-csv, this
     is just a participant_id stub. With --demographics-csv, merges in the source
     columns (subject_id, tide_id, sex, age, pta, pta_hf, THI, TFI, pitch,
     tin_loudness, site), auto-detecting whether `subject_id` or `tide_id` is the
     column that matches the BIDS sub-<code> values.
  3. Materializes derivatives/freesurfer/sub-<code> symlinks into the shared
     FreeSurfer subjects dir, for every subject marked recon_all=OK in
     derivatives/freesurfer_status.tsv (from 01_check_freesurfer_status.py).
  4. Copies the FreeSurfer license into derivatives/ so it sits inside the tree
     that gets bind-mounted into containers later.
  5. Patches PhaseEncodingDirection into DWI/BOLD JSON sidecars, ONLY where the
     key is currently absent (this dataset has no fieldmaps and no PE info at
     all; AP ("j-") is assumed per the old, deleted ant_03_processDTI.sh and
     confirmed by you). Every patch is logged, never silently unlogged.
  6. Patches TotalReadoutTime into DWI/BOLD JSON sidecars, ONLY where absent.
     QSIPrep's eddy step requires this numeric value even with no SDC/topup
     requested; the original scanner files that would carry the true value no
     longer exist on this machine, so a nominal placeholder (0.05s) is used --
     an assumption, not a measurement, confirmed by you. Logged to
     derivatives/logs/03_json_patch_manifest_readout_time.tsv.
  7. Attempts a bids-validator pass (via `npx bids-validator`) and writes a
     report. If Node/npx isn't installed on this machine, this step is skipped
     with a clear message rather than failing the whole script.

Run with:
  python3 03_prepare_bids_scaffold.py --demographics-csv /path/to/demographics.csv
  python3 03_prepare_bids_scaffold.py                       # scaffold only, participants.tsv stub
  python3 03_prepare_bids_scaffold.py --demographics-csv X --overwrite-participants
                                                             # force-regenerate participants.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

BIDS_ROOT = Path("/home/ubuntu/volume/antinomics")
FS_SUBJECTS_DIR = Path("/home/ubuntu/volume/Tinception/subjects_fs_dir")
DERIV = BIDS_ROOT / "derivatives"
LOG_DIR = DERIV / "logs"
FS_STATUS_TSV = DERIV / "freesurfer_status.tsv"
FS_LICENSE_SRC = Path("/usr/local/freesurfer/8.0.0/license.txt")
FS_LICENSE_DST = DERIV / "freesurfer_license.txt"

PE_DIRECTION_VALUE = "j-"  # AP, inferred (not measured) -- see README note

# QSIPrep's eddy-prep step (hmc_sdc_wf.gather_inputs) requires a numeric
# TotalReadoutTime for every DWI/BOLD file to build FSL eddy's acqparams.txt,
# even when no SDC/topup is requested (confirmed via a 06_run_qsiprep.sh smoke
# test crash: TypeError formatting spec['TotalReadoutTime'] == None). Original
# PAR/REC scanner files (which would carry the real EchoSpacing/WFS) no longer
# exist on this machine (/home/ubuntu/volume/raws is empty) -- there is no way
# to recover a measured value. 0.05s is a commonly used nominal placeholder for
# single-shot EPI when the true value is unknown; confirmed with you to use it.
# This is an ASSUMPTION, not a measurement -- see README note and the
# 03_json_patch_manifest_readout_time.tsv log for exactly which files were patched.
TOTAL_READOUT_TIME_VALUE = 0.05

# Expected demographics CSV columns (as given), and the participants.json
# descriptions for each. Units marked TBD are genuinely unconfirmed -- flagged
# rather than guessed.
DEMOGRAPHICS_COLUMNS = [
    "subject_id", "tide_id", "sex", "age", "pta", "pta_hf",
    "THI", "TFI", "pitch", "tin_loudness", "site",
]
PARTICIPANTS_DESCRIPTIONS = {
    "subject_id": "Original internal subject code from the source demographics file "
                  "(matched against BIDS participant_id).",
    "tide_id": "TIDE consortium participant identifier.",
    "sex": "Biological sex as recorded in the source demographics file.",
    "age": "Age in years, as recorded in the source demographics file.",
    "pta": "Pure-tone average hearing threshold. Units/frequency set TBD -- "
           "confirm against the source data dictionary before analysis.",
    "pta_hf": "High-frequency pure-tone average hearing threshold. Units/frequency "
              "set TBD -- confirm against the source data dictionary before analysis.",
    "THI": "Tinnitus Handicap Inventory total score.",
    "TFI": "Tinnitus Functional Index total score.",
    "pitch": "Tinnitus pitch-match frequency. Units TBD (likely Hz).",
    "tin_loudness": "Tinnitus loudness-match level. Units TBD.",
    "site": "Data-collection / recruitment site.",
}


def bids_subject_codes() -> list[str]:
    return sorted(p.name.removeprefix("sub-") for p in BIDS_ROOT.glob("sub-*") if p.is_dir())


# --- 1. scaffold files ------------------------------------------------------

def write_dataset_description() -> None:
    path = BIDS_ROOT / "dataset_description.json"
    if path.exists():
        print(f"[skip] {path} already exists")
        return
    content = {
        "Name": "ANTINOMICS",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "03_prepare_bids_scaffold.py"}],
    }
    path.write_text(json.dumps(content, indent=4) + "\n")
    print(f"[write] {path}")


def write_bidsignore() -> None:
    path = BIDS_ROOT / ".bidsignore"
    if path.exists():
        print(f"[skip] {path} already exists")
        return
    path.write_text("derivatives/\n")
    print(f"[write] {path}")


def write_readme() -> None:
    path = BIDS_ROOT / "README"
    if path.exists():
        print(f"[skip] {path} already exists")
        return
    path.write_text(
        "ANTINOMICS tinnitus dataset (BIDS)\n"
        "===================================\n\n"
        "T1w, high-res hippocampal T2w, single-shell DWI (32 dir, b=1000 + 1 b0), "
        "and two runs of resting-state BOLD (task-rest_run-1, task-rest_run-2) per "
        "subject.\n\n"
        "Known limitations (documented, not hidden):\n"
        "- No fieldmaps of any kind (no fmap/, no reverse phase-encoded b0). DWI and "
        "functional preprocessing proceed WITHOUT susceptibility distortion "
        "correction (SDC). QSIPrep runs with no fmap available; fMRIPrep runs "
        "without --use-syn-sdc.\n"
        "- Original JSON sidecars did not record PhaseEncodingDirection for DWI or "
        "BOLD. This has been patched to an assumed value of \"j-\" (AP), based on "
        "the site's historical acquisition protocol. This is an inferred default, "
        "not a measured value -- see derivatives/logs/03_json_patch_manifest.tsv "
        "for exactly which files were patched.\n"
        "- Original JSON sidecars also did not record TotalReadoutTime for DWI or "
        "BOLD, and the original scanner files that would carry the true value no "
        "longer exist on this machine. QSIPrep's eddy step requires this value "
        "even with no SDC/topup requested, so a nominal placeholder of 0.05s has "
        "been patched in where absent. This is an assumption, not a measurement -- "
        "see derivatives/logs/03_json_patch_manifest_readout_time.tsv for exactly "
        "which files were patched.\n"
    )
    print(f"[write] {path}")


def write_changes() -> None:
    path = BIDS_ROOT / "CHANGES"
    if path.exists():
        print(f"[skip] {path} already exists")
        return
    path.write_text(
        "1.0.0 2026-07-24\n"
        "  - Initial BIDS scaffold: dataset_description.json, .bidsignore, README, "
        "CHANGES, participants.tsv/json.\n"
        "  - Patched PhaseEncodingDirection (assumed \"j-\"/AP) into DWI/BOLD "
        "sidecars where absent.\n"
    )
    print(f"[write] {path}")


# --- 2. participants.tsv / participants.json --------------------------------

def _normalize(code: str) -> str:
    return code.strip().lower().removeprefix("sub-")


def _pick_join_column(rows: list[dict[str, str]], bids_codes: set[str]) -> str:
    best_col, best_overlap = None, -1
    for col in ("subject_id", "tide_id"):
        if not rows or col not in rows[0]:
            continue
        overlap = sum(1 for r in rows if _normalize(r.get(col, "")) in bids_codes)
        print(f"  column '{col}' matches {overlap}/{len(rows)} CSV rows against BIDS codes")
        if overlap > best_overlap:
            best_col, best_overlap = col, overlap
    if best_col is None or best_overlap == 0:
        raise SystemExit(
            "Could not match either 'subject_id' or 'tide_id' column to any BIDS "
            "sub-<code>. Check the CSV values against the BIDS subject list before "
            "re-running."
        )
    return best_col


def build_participants_table(demographics_csv: Path | None, overwrite: bool) -> None:
    tsv_path = BIDS_ROOT / "participants.tsv"
    json_path = BIDS_ROOT / "participants.json"
    codes = bids_subject_codes()
    bids_codes = set(codes)

    if tsv_path.exists() and not overwrite:
        print(f"[skip] {tsv_path} already exists (use --overwrite-participants to regenerate)")
        return

    if demographics_csv is None:
        with tsv_path.open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["participant_id"])
            for code in codes:
                w.writerow([f"sub-{code}"])
        print(f"[write] {tsv_path} (stub: participant_id only, no demographics supplied)")
        json_path.write_text(json.dumps(
            {"participant_id": {"Description": "BIDS participant identifier."}}, indent=4
        ) + "\n")
        print(f"[write] {json_path} (stub)")
        return

    with demographics_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
        csv_columns = reader.fieldnames or []

    missing_expected = [c for c in DEMOGRAPHICS_COLUMNS if c not in csv_columns]
    if missing_expected:
        print(f"[warn] demographics CSV is missing expected columns: {missing_expected} "
              f"-- proceeding with whatever columns are present: {csv_columns}")

    join_col = _pick_join_column(csv_rows, bids_codes)
    print(f"[info] using '{join_col}' to match demographics rows to BIDS participant_id")

    by_code = {_normalize(r[join_col]): r for r in csv_rows if _normalize(r.get(join_col, "")) in bids_codes}

    unmatched_csv = [r[join_col] for r in csv_rows if _normalize(r.get(join_col, "")) not in bids_codes]
    unmatched_bids = [c for c in codes if c not in by_code]
    if unmatched_csv:
        print(f"[warn] {len(unmatched_csv)} CSV row(s) did not match any BIDS subject: {unmatched_csv}")
    if unmatched_bids:
        print(f"[warn] {len(unmatched_bids)} BIDS subject(s) have no demographics row "
              f"(will be written as 'n/a'): {unmatched_bids}")

    out_columns = [c for c in csv_columns if c]  # preserve source column order/names as-is
    with tsv_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["participant_id"] + out_columns)
        for code in codes:
            row = by_code.get(code)
            values = [row.get(c, "n/a") if row else "n/a" for c in out_columns]
            values = [v if v not in ("", None) else "n/a" for v in values]
            w.writerow([f"sub-{code}"] + values)
    print(f"[write] {tsv_path} ({len(codes)} subjects, {len(by_code)} with demographics)")

    data_dict = {"participant_id": {"Description": "BIDS participant identifier."}}
    for c in out_columns:
        data_dict[c] = {"Description": PARTICIPANTS_DESCRIPTIONS.get(
            c, "Source column from demographics CSV (no description on file).")}
    json_path.write_text(json.dumps(data_dict, indent=4) + "\n")
    print(f"[write] {json_path}")


# --- 3. FreeSurfer symlinks --------------------------------------------------

def load_recon_ok_codes() -> set[str]:
    if not FS_STATUS_TSV.exists():
        print(f"[warn] {FS_STATUS_TSV} not found -- run 01_check_freesurfer_status.py first. "
              f"Skipping FreeSurfer symlink step.")
        return set()
    with FS_STATUS_TSV.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {r["subject"].removeprefix("sub-") for r in rows if r.get("recon_all") == "OK"}


def materialize_freesurfer_symlinks() -> None:
    ok_codes = load_recon_ok_codes()
    if not ok_codes:
        return
    out_dir = DERIV / "freesurfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    made, skipped, warned = 0, 0, 0
    for code in sorted(ok_codes):
        link = out_dir / f"sub-{code}"
        target = Path(f"../../../Tinception/subjects_fs_dir/{code}")
        if link.is_symlink():
            if link.readlink() == target:
                skipped += 1
                continue
            print(f"[warn] {link} exists and points elsewhere ({link.readlink()}) -- not touching it")
            warned += 1
            continue
        if link.exists():
            print(f"[warn] {link} exists and is not a symlink -- not touching it")
            warned += 1
            continue
        link.symlink_to(target)
        made += 1
    print(f"[freesurfer symlinks] created {made}, already correct {skipped}, warned {warned}")


def copy_fs_license() -> None:
    if not FS_LICENSE_SRC.exists():
        print(f"[warn] FreeSurfer license not found at {FS_LICENSE_SRC}, skipping")
        return
    if FS_LICENSE_DST.exists():
        if FS_LICENSE_DST.read_bytes() == FS_LICENSE_SRC.read_bytes():
            print(f"[skip] {FS_LICENSE_DST} already present and matches source")
            return
        print(f"[warn] {FS_LICENSE_DST} exists and differs from {FS_LICENSE_SRC} -- not overwriting")
        return
    DERIV.mkdir(parents=True, exist_ok=True)
    shutil.copy(FS_LICENSE_SRC, FS_LICENSE_DST)
    print(f"[write] {FS_LICENSE_DST}")


# --- 4. PhaseEncodingDirection patch -----------------------------------------

def patch_phase_encoding_direction() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = LOG_DIR / "03_json_patch_manifest.tsv"
    rows = []
    for code in bids_subject_codes():
        subj_dir = BIDS_ROOT / f"sub-{code}"
        json_files = sorted((subj_dir / "dwi").glob("*.json")) + sorted((subj_dir / "func").glob("*_bold.json"))
        for jf in json_files:
            data = json.loads(jf.read_text())
            if "PhaseEncodingDirection" in data:
                rows.append((f"sub-{code}", str(jf.relative_to(BIDS_ROOT)), "ALREADY_PRESENT", data["PhaseEncodingDirection"]))
                continue
            data["PhaseEncodingDirection"] = PE_DIRECTION_VALUE
            jf.write_text(json.dumps(data, indent=4) + "\n")
            rows.append((f"sub-{code}", str(jf.relative_to(BIDS_ROOT)), "PATCHED", PE_DIRECTION_VALUE))

    with manifest_path.open("w") as f:
        f.write("subject\tfile\taction\tvalue\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    patched = sum(1 for r in rows if r[2] == "PATCHED")
    present = sum(1 for r in rows if r[2] == "ALREADY_PRESENT")
    print(f"[PhaseEncodingDirection] patched {patched}, already present {present} "
          f"(manifest: {manifest_path})")


def patch_total_readout_time() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = LOG_DIR / "03_json_patch_manifest_readout_time.tsv"
    rows = []
    for code in bids_subject_codes():
        subj_dir = BIDS_ROOT / f"sub-{code}"
        json_files = sorted((subj_dir / "dwi").glob("*.json")) + sorted((subj_dir / "func").glob("*_bold.json"))
        for jf in json_files:
            data = json.loads(jf.read_text())
            if "TotalReadoutTime" in data:
                rows.append((f"sub-{code}", str(jf.relative_to(BIDS_ROOT)), "ALREADY_PRESENT", str(data["TotalReadoutTime"])))
                continue
            data["TotalReadoutTime"] = TOTAL_READOUT_TIME_VALUE
            jf.write_text(json.dumps(data, indent=4) + "\n")
            rows.append((f"sub-{code}", str(jf.relative_to(BIDS_ROOT)), "PATCHED", str(TOTAL_READOUT_TIME_VALUE)))

    with manifest_path.open("w") as f:
        f.write("subject\tfile\taction\tvalue\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    patched = sum(1 for r in rows if r[2] == "PATCHED")
    present = sum(1 for r in rows if r[2] == "ALREADY_PRESENT")
    print(f"[TotalReadoutTime] patched {patched}, already present {present} "
          f"(manifest: {manifest_path})")


# --- 5. bids-validator --------------------------------------------------------

def run_bids_validator() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOG_DIR / "03_bids_validator_report.txt"
    try:
        result = subprocess.run(
            ["npx", "--yes", "bids-validator@latest", str(BIDS_ROOT)],
            capture_output=True, text=True, timeout=600,
        )
        report_path.write_text(result.stdout + "\n" + result.stderr)
        print(f"[bids-validator] exit code {result.returncode}, report: {report_path}")
    except FileNotFoundError:
        msg = ("npx/Node.js not found on this machine -- bids-validator was not run. "
               "Install Node (or run bids-validator from inside a container later) and "
               "re-run this script, or validate manually.")
        report_path.write_text(msg + "\n")
        print(f"[warn] {msg}")
    except subprocess.TimeoutExpired:
        msg = "bids-validator timed out after 600s."
        report_path.write_text(msg + "\n")
        print(f"[warn] {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demographics-csv", type=Path, default=None,
                     help="CSV with columns: " + ", ".join(DEMOGRAPHICS_COLUMNS))
    ap.add_argument("--overwrite-participants", action="store_true",
                     help="Regenerate participants.tsv/json even if they already exist.")
    ap.add_argument("--skip-validator", action="store_true",
                     help="Skip the bids-validator pass.")
    args = ap.parse_args()

    write_dataset_description()
    write_bidsignore()
    write_readme()
    write_changes()
    build_participants_table(args.demographics_csv, args.overwrite_participants)
    materialize_freesurfer_symlinks()
    copy_fs_license()
    patch_phase_encoding_direction()
    patch_total_readout_time()
    if not args.skip_validator:
        run_bids_validator()

    print("\nDone. Re-run with --overwrite-participants once a corrected demographics "
          "CSV is available, if needed.")


if __name__ == "__main__":
    main()
