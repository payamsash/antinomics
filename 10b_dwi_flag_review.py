#!/usr/bin/env python3
"""
Deep-dive review of the 12 subjects step 10 flagged on DWI-side QC scalars
(|modified z| > 3.5 on mean_fd/max_fd/t1_neighbor_corr/CNR1_mean), following
the same "flagging is not the decision" precedent as 05b_mriqc_group_report.py.

A flat re-listing of the 12 flags isn't enough to act on -- this script asks
three sharper questions per subject:
  1. Which metric(s) actually triggered the flag, and in which direction?
  2. Is the flag corroborated by other QC signal (bad-slice count, T1-DWI
     registration quality), or isolated to one metric?
  3. Did the flag actually manifest downstream -- does this subject show an
     elevated SFC node-exclusion fraction relative to the group (script 11's
     own independent, later QC layer), or did the pipeline process them fine?

Run with:
    /home/ubuntu/volume/miniconda3/envs/tinnorm/bin/python3 10b_dwi_flag_review.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DERIV = Path("/home/ubuntu/volume/antinomics/derivatives")
QC_REPORT = DERIV / "qc" / "auditory_roi_report.tsv"
SFC_DIR = DERIV / "sfc"
OUT_DIR = DERIV / "qc"
FIG_DIR = OUT_DIR / "figures"

DWI_COLS = ["dwi_mean_fd", "dwi_max_fd", "dwi_raw_num_bad_slices", "dwi_t1_neighbor_corr", "dwi_CNR0_mean", "dwi_CNR1_mean"]


def main() -> None:
    qc = pd.read_csv(QC_REPORT, sep="\t").drop_duplicates("subject").set_index("subject")
    flag_cols = [f"flag_{c}" for c in DWI_COLS]
    flagged = sorted(qc.index[qc[flag_cols].any(axis=1)])
    print(f"{len(flagged)} subjects flagged on >=1 DWI metric: {flagged}\n")

    # Which metric(s) triggered each flag
    trigger = qc.loc[flagged, flag_cols].rename(columns=lambda c: c.replace("flag_dwi_", ""))
    print("=== triggering metric(s) per subject ===")
    for sub in flagged:
        triggered = [c for c in trigger.columns if trigger.loc[sub, c]]
        print(f"  {sub}: {triggered}")

    # Percentile rank in the full 76-subject group, for context/severity
    ranks = qc[DWI_COLS].rank(pct=True)

    # Node-exclusion fraction from the independent SFC-stage QC layer (script 11)
    excl_frac = {}
    for sub in qc.index:
        f = SFC_DIR / sub / "atlas-4S456Parcels" / f"{sub}_atlas-4S456Parcels_desc-nodeexclusion.tsv"
        if f.exists():
            excl_frac[sub] = pd.read_csv(f, sep="\t")["excluded_combined"].mean()
    excl_series = pd.Series(excl_frac, name="sfc_node_exclusion_fraction")
    group_median_excl = excl_series.median()

    report = pd.DataFrame({
        "triggering_metrics": [", ".join(c for c in trigger.columns if trigger.loc[s, c]) for s in flagged],
        "dwi_mean_fd_pct": ranks.loc[flagged, "dwi_mean_fd"].round(2),
        "dwi_max_fd_pct": ranks.loc[flagged, "dwi_max_fd"].round(2),
        "dwi_raw_num_bad_slices": qc.loc[flagged, "dwi_raw_num_bad_slices"],
        "dwi_t1_neighbor_corr": qc.loc[flagged, "dwi_t1_neighbor_corr"].round(3),
        "dwi_CNR1_mean": qc.loc[flagged, "dwi_CNR1_mean"].round(2),
        "dwi_CNR1_mean_pct": ranks.loc[flagged, "dwi_CNR1_mean"].round(2),
        "sfc_node_exclusion_fraction": excl_series.reindex(flagged).round(3),
        "sfc_excl_vs_group_median": (excl_series.reindex(flagged) - group_median_excl).round(3),
    }, index=flagged)
    report.index.name = "subject"

    # Classify into interpretable categories rather than one flat "flagged" bucket
    def classify(sub: str) -> str:
        trig = report.loc[sub, "triggering_metrics"]
        motion_flagged = "mean_fd" in trig or "max_fd" in trig
        cnr_flagged = "CNR1_mean" in trig
        bad_slices = qc.loc[sub, "dwi_raw_num_bad_slices"]
        if motion_flagged and bad_slices >= 10:
            return "high-motion + elevated bad-slice count (strongest concern)"
        if motion_flagged:
            return "high-motion only"
        if cnr_flagged and qc.loc[sub, "dwi_CNR1_mean"] > qc["dwi_CNR1_mean"].median():
            return "anomalously HIGH CNR (not a coupling/motion concern)"
        return "other"

    report["category"] = [classify(s) for s in flagged]
    report.to_csv(OUT_DIR / "dwi_flag_review.tsv", sep="\t")
    print(f"\nWrote {OUT_DIR / 'dwi_flag_review.tsv'}")

    print("\n=== category summary ===")
    print(report["category"].value_counts().to_string())
    print(f"\nGroup median SFC node-exclusion fraction (4S456): {group_median_excl:.3f}")
    print("Flagged subjects' SFC node-exclusion fractions (all should be near-median if the flag isn't manifesting downstream):")
    print(report[["category", "sfc_node_exclusion_fraction", "sfc_excl_vs_group_median"]].sort_values("sfc_excl_vs_group_median"))

    make_figure(qc, flagged, report, group_median_excl)


def make_figure(qc: pd.DataFrame, flagged: list[str], report: pd.DataFrame, group_median_excl: float) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cat_colors = {
        "high-motion + elevated bad-slice count (strongest concern)": "#c23a33",
        "high-motion only": "#e8874a",
        "anomalously HIGH CNR (not a coupling/motion concern)": "#256abf",
        "other": "#8a93a1",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    ax.scatter(qc["dwi_mean_fd"], qc["dwi_CNR1_mean"], s=22, color="#c3c9d1", alpha=0.7, label="not flagged", zorder=2)
    for sub in flagged:
        cat = report.loc[sub, "category"]
        ax.scatter(qc.loc[sub, "dwi_mean_fd"], qc.loc[sub, "dwi_CNR1_mean"], s=60, color=cat_colors[cat],
                   edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_xlabel("DWI mean framewise displacement (mm)", fontsize=9.5, color="#5b6472")
    ax.set_ylabel("DWI CNR1 (b-shell contrast-to-noise)", fontsize=9.5, color="#5b6472")
    ax.set_title("Two distinct flag clusters, not one", fontsize=11, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    order = report.sort_values("sfc_excl_vs_group_median").index.tolist()
    colors = [cat_colors[report.loc[s, "category"]] for s in order]
    ax.barh(order, report.loc[order, "sfc_node_exclusion_fraction"], color=colors)
    ax.axvline(group_median_excl, color="#20242b", linestyle="--", linewidth=1, label=f"group median ({group_median_excl:.3f})")
    ax.set_xlabel("SFC node-exclusion fraction (4S456, script 11's independent QC layer)", fontsize=9, color="#5b6472")
    ax.set_title("Does the DWI flag manifest downstream?", fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, markersize=7, markeredgecolor="white") for c in cat_colors.values()]
    fig.legend(handles, list(cat_colors.keys()), loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Review of step 10's 12 DWI-flagged subjects", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(FIG_DIR / "dwi_flag_review.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote figure: {FIG_DIR / 'dwi_flag_review.png'}")


if __name__ == "__main__":
    main()
