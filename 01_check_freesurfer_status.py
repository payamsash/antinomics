#!/usr/bin/env python3
"""
Audit FreeSurfer structural-segmentation completeness for the ANTINOMICS cohort.

Source : /home/ubuntu/volume/antinomics/sub-<id>/            (BIDS raw data)
Target : /home/ubuntu/volume/Tinception/subjects_fs_dir/<id>/ (shared FreeSurfer
         subjects directory; subject folders use the bare 4-letter BIDS code,
         no "sub-" prefix, and sit alongside ~600 unrelated projects' subjects)

For each BIDS subject this checks nine FreeSurfer pipeline markers (the steps run
by the old, now-deleted ant_01_reconostruct.sh / ant_02_subsegment.sh):
  recon-all, hippocampus+amygdala subfields (segmentHA_T2.sh), thalamic nuclei,
  brainstem substructures, arousal network (AAN), hypothalamic subunits,
  Choi/Buckner striatum+cerebellum functional masks, Schaefer-400, Glasser/HCPMMP1.

Writes a per-subject/per-step manifest to
  /home/ubuntu/volume/antinomics/derivatives/freesurfer_status.tsv
and prints a summary grouped by completion pattern, so re-running this script is
the single source of truth for "what still needs to be run" instead of re-deriving
it by hand.

Run with:  python3 01_check_freesurfer_status.py
"""

from __future__ import annotations

from pathlib import Path

BIDS_ROOT = Path("/home/ubuntu/volume/antinomics")
FS_SUBJECTS_DIR = Path("/home/ubuntu/volume/Tinception/subjects_fs_dir")
OUT_TSV = BIDS_ROOT / "derivatives" / "freesurfer_status.tsv"

# (column name, path relative to <fs_subjects_dir>/<subject>/ that must exist)
MARKERS = [
    ("recon_all", "mri/aparc+aseg.mgz"),
    ("hippo_amyg", "mri/lh.hippoAmygLabels-T1-T2.v22.mgz"),
    ("thalamic_nuclei", "mri/ThalamicNuclei.v13.T1.mgz"),
    ("brainstem", "mri/brainstemSsLabels.v13.mgz"),
    ("aan", "mri/arousalNetworkLabels.v10.mgz"),
    ("hypothalamus", "mri/hypothalamic_subunits_seg.v1.mgz"),
    ("striatum_cerebellum", "mri/striatum_17_network_Tight_mask.nii.gz"),
    ("schaefer", "mri/Schaefer2018_400_7Networks.mgz"),
    ("glasser", "mri/HCPMMP1.mgz"),
]


def bids_subject_codes() -> list[str]:
    return sorted(p.name.removeprefix("sub-") for p in BIDS_ROOT.glob("sub-*") if p.is_dir())


def check_subject(code: str) -> dict[str, str]:
    subj_dir = FS_SUBJECTS_DIR / code
    if not subj_dir.is_dir():
        return {name: "NO_DIR" for name, _ in MARKERS}
    return {name: "OK" if (subj_dir / rel).exists() else "MISSING" for name, rel in MARKERS}


def main() -> None:
    codes = bids_subject_codes()
    rows = {code: check_subject(code) for code in codes}

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w") as f:
        header = ["subject"] + [name for name, _ in MARKERS]
        f.write("\t".join(header) + "\n")
        for code in codes:
            f.write("\t".join(["sub-" + code] + [rows[code][name] for name, _ in MARKERS]) + "\n")
    print(f"Wrote manifest: {OUT_TSV}\n")

    # summary grouped by completion pattern
    patterns: dict[tuple[str, ...], list[str]] = {}
    for code in codes:
        pattern = tuple(rows[code][name] for name, _ in MARKERS)
        patterns.setdefault(pattern, []).append(code)

    print(f"{len(codes)} BIDS subjects checked against {FS_SUBJECTS_DIR}\n")
    header = [name for name, _ in MARKERS]
    print("N   | " + " | ".join(header))
    for pattern, members in sorted(patterns.items(), key=lambda kv: -len(kv[1])):
        print(f"{len(members):<3} | " + " | ".join(pattern))
        print(f"    subjects: {', '.join(sorted(members))}\n")

    fully_missing = [c for c in codes if rows[c]["recon_all"] == "NO_DIR"]
    incomplete = [
        c for c in codes
        if rows[c]["recon_all"] != "NO_DIR" and any(v == "MISSING" for v in rows[c].values())
    ]
    complete = [c for c in codes if all(v == "OK" for v in rows[c].values())]
    print(f"Complete (all 9 markers OK): {len(complete)}")
    print(f"Incomplete (some markers missing): {len(incomplete)} -> {', '.join(sorted(incomplete))}")
    print(f"No FreeSurfer directory at all: {len(fully_missing)} -> {', '.join(sorted(fully_missing))}")


if __name__ == "__main__":
    main()
