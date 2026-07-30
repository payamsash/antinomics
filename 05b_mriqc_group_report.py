#!/usr/bin/env python3
"""
Aggregate per-subject MRIQC IQMs into group tables + publication-quality figures.

Reads the per-scan IQM JSON sidecars MRIQC already wrote under
derivatives/mriqc/sub-*/{anat,func}/*.json (no re-processing) and produces:
  - long/wide CSV tables (derivatives/mriqc_group/tables/)
  - seaborn/matplotlib QC figures, one metric-grid per modality plus two
    summary figures (derivatives/mriqc_group/figures/{anat,func,summary}/)
  - MRIQC's own official group-level TSVs/HTML, copied into
    derivatives/mriqc_group/official/

Fast (parses existing JSON + one lightweight apptainer group-level call);
safe to re-run any time 05_run_mriqc.sh finishes more subjects.

Run with the tinnorm conda env (has pandas/numpy/scipy/matplotlib/seaborn):
    /home/ubuntu/volume/miniconda3/envs/tinnorm/bin/python 05b_mriqc_group_report.py
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

VOLUME = Path("/home/ubuntu/volume")
BIDS_ROOT = VOLUME / "antinomics"
MRIQC_DIR = BIDS_ROOT / "derivatives" / "mriqc"
OUT_ROOT = BIDS_ROOT / "derivatives" / "mriqc_group"
TABLES_DIR = OUT_ROOT / "tables"
FIG_ANAT = OUT_ROOT / "figures" / "anat"
FIG_FUNC = OUT_ROOT / "figures" / "func"
FIG_SUMMARY = OUT_ROOT / "figures" / "summary"
OFFICIAL_DIR = OUT_ROOT / "official"
MANIFEST = BIDS_ROOT / "derivatives" / "logs" / "05_mriqc" / "mriqc_status.tsv"

APPTAINER_BIN = VOLUME / "miniconda3/envs/apptainer/bin/apptainer"
MRIQC_SIF = VOLUME / "containers/mriqc-24.0.2.sif"
TEMPLATEFLOW_CACHE = VOLUME / "templateflow_cache"

# --- antinomics dataviz defaults (see project's dataviz skill / palette.md) ---
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BOX_FILL = "#f0efec"
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

sns.set_theme(style="ticks", rc={
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRID,
    "font.family": "sans-serif",
})

# MRIQC's own "fber" IQM is a sentinel -1.0 (undefined) for every subject in
# this dataset (no usable air region in the FOV) -- wm2max substitutes as a
# meaningful anatomical contrast metric instead.
ANAT_METRICS = {
    "cjv": ("Coefficient of joint variation (CJV)", "lower is better"),
    "cnr": ("Contrast-to-noise ratio (CNR)", "higher is better"),
    "efc": ("Entropy-focus criterion (EFC)", "lower is better"),
    "wm2max": ("WM-to-max intensity ratio", "closer to ~0.6-0.8 is better"),
    "snr_total": ("SNR (total)", "higher is better"),
    "fwhm_avg": ("Smoothness, avg FWHM (mm)", "lower is better"),
    "qi_2": ("Artifact quality index (QI2)", "lower is better"),
    "inu_med": ("Intensity non-uniformity (median)", "closer to 1.0 is better"),
}
BOLD_METRICS = {
    "fd_mean": ("Mean framewise displacement (mm)", "lower is better"),
    "fd_perc": ("% high-motion frames (>0.5mm)", "lower is better"),
    "tsnr": ("Temporal SNR", "higher is better"),
    "dvars_std": ("DVARS (standardized)", "lower is better"),
    "gcor": ("Global correlation (GCOR)", "closer to 0 is better"),
    "efc": ("Entropy-focus criterion (EFC)", "lower is better"),
    "snr": ("SNR", "higher is better"),
    "aor": ("AFNI outlier ratio (AOR)", "lower is better"),
}

FLAG_Z_THRESH = 3.5  # Iglewicz & Hoaglin modified-z-score convention


def load_anat_table(glob_pattern, label):
    rows = []
    for jf in sorted(MRIQC_DIR.glob(glob_pattern)):
        d = json.loads(jf.read_text())
        row = {"subject": jf.name.split("_")[0], "file": jf.name}
        for k in ANAT_METRICS:
            v = d.get(k, np.nan)
            row[k] = np.nan if (k == "fber" and v == -1.0) else v
        rows.append(row)
    df = pd.DataFrame(rows)
    df["modality"] = label
    return df


def load_bold_table():
    rows = []
    for jf in sorted(MRIQC_DIR.glob("sub-*/func/*_bold.json")):
        run = "run-1" if "run-1" in jf.name else ("run-2" if "run-2" in jf.name else "run-?")
        d = json.loads(jf.read_text())
        row = {"subject": jf.name.split("_")[0], "run": run, "file": jf.name}
        for k in BOLD_METRICS:
            row[k] = d.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_flags(df, metrics, group_col=None):
    """Long-format (subject, group, metric, value, z, flagged) via a per-metric,
    per-group modified z-score (MAD-based, robust to the outliers we're hunting for)."""
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col))
    frames = []
    for metric in metrics:
        for gval, gdf in groups:
            x = gdf[metric].astype(float)
            med = x.median()
            mad = (x - med).abs().median()
            z = pd.Series(0.0, index=x.index) if (not mad or np.isnan(mad)) else 0.6745 * (x - med) / mad
            frames.append(pd.DataFrame({
                "subject": gdf["subject"].values,
                "group": (gval if group_col else "all"),
                "metric": metric,
                "value": x.values,
                "z": z.values,
            }))
    out = pd.concat(frames, ignore_index=True)
    out["flagged"] = out["z"].abs() > FLAG_Z_THRESH
    return out


def _annotate_flagged(ax, xs, ys, subjects, color):
    """Direct-label flagged points, cycling through offset slots whenever a new
    label's pixel position is within ~11px of an already-placed one -- avoids
    stacking overlapping text for clusters of nearby outliers."""
    if len(xs) == 0:
        return
    order = np.argsort(ys)
    offset_cycle = [(4, 2), (4, 14), (4, -12), (-24, 2), (-24, 14), (-24, -12)]
    placed_px = []
    for i in order:
        xi, yi, subj = xs[i], ys[i], subjects[i]
        x_px, y_px = ax.transData.transform((xi, yi))
        slot = sum(1 for (_, py) in placed_px if abs(y_px - py) < 11) % len(offset_cycle)
        dx, dy = offset_cycle[slot]
        ax.annotate(subj, (xi, yi), fontsize=6, color=color, xytext=(dx, dy), textcoords="offset points")
        placed_px.append((x_px, y_px))


def add_panel(ax, groups, title, ylabel, seed):
    """groups: list of {label, color, values: Series[subject->value], z: Series[subject->z]}."""
    rng = np.random.default_rng(seed)
    box_data = [g["values"].dropna().values for g in groups]
    bp = ax.boxplot(box_data, positions=range(len(groups)), widths=0.45,
                     showfliers=False, patch_artist=True, zorder=1)
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_FILL)
        patch.set_edgecolor(INK_MUTED)
        patch.set_linewidth(1.0)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(INK_MUTED)

    any_flag_labeled = False
    for i, g in enumerate(groups):
        vals, z = g["values"], g["z"]
        n = len(vals)
        x = i + rng.uniform(-0.14, 0.14, n)
        flagged = (z.abs() > FLAG_Z_THRESH).fillna(False)
        ok = ~flagged
        ax.scatter(x[ok.values], vals[ok.values], s=22, color=g["color"], alpha=0.75,
                   edgecolor="white", linewidth=0.4, zorder=3)
        if flagged.any():
            ax.scatter(x[flagged.values], vals[flagged.values], s=44, color=STATUS_CRITICAL,
                       edgecolor=INK_PRIMARY, linewidth=0.6, zorder=4)
            fx = x[flagged.values]
            fy = vals[flagged.values].to_numpy()
            fsubj = [s.replace("sub-", "") for s in vals.index[flagged.values]]
            _annotate_flagged(ax, fx, fy, fsubj, STATUS_CRITICAL)
            any_flag_labeled = True

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g["label"] for g in groups], fontsize=9)
    ax.set_title(title, fontsize=10, color=INK_PRIMARY, pad=6)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    return any_flag_labeled


def make_metric_grid(flags_long, metrics_spec, group_order, group_colors, ncols, out_base, suptitle, note=None):
    n = len(metrics_spec)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (metric, (label, direction)) in zip(axes, metrics_spec.items()):
        sub = flags_long[flags_long["metric"] == metric]
        groups = []
        for glabel, color in zip(group_order, group_colors):
            gdf = sub[sub["group"] == glabel].set_index("subject")
            groups.append({"label": glabel, "color": color, "values": gdf["value"], "z": gdf["z"]})
        add_panel(ax, groups, label, f"({direction})", seed=abs(hash(metric)) % (2**32))
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=13, color=INK_PRIMARY, fontweight="bold", y=1.02)
    caption = f"Red points: |modified z-score| > {FLAG_Z_THRESH} within its group, labeled by subject code."
    if note:
        caption += "\n" + note
    fig.text(0.01, -0.02 - 0.015 * caption.count("\n"), caption, fontsize=8, color=INK_MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_flag_summary_figure(flag_counts, n_assessed, out_base):
    flag_counts = flag_counts[flag_counts > 0].sort_values(ascending=True)
    if flag_counts.empty:
        print("[summary] no subjects crossed the outlier threshold on any metric -- skipping flag summary figure")
        return

    def tier_color(n):
        if n >= 6:
            return STATUS_CRITICAL
        if n >= 3:
            return STATUS_SERIOUS
        return STATUS_WARNING

    colors = [tier_color(n) for n in flag_counts.values]
    fig_h = max(3.0, 0.28 * len(flag_counts) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    ax.barh(flag_counts.index, flag_counts.values, color=colors, height=0.6, zorder=3)
    ax.set_xlabel(f"Number of flagged IQMs (of {n_assessed} assessed per subject)")
    ax.set_title(f"Subjects with ≥1 outlier IQM (|modified z| > {FLAG_Z_THRESH})",
                 fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    handles = [Patch(color=STATUS_WARNING, label="1-2 flagged"),
               Patch(color=STATUS_SERIOUS, label="3-5 flagged"),
               Patch(color=STATUS_CRITICAL, label="6+ flagged")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_status_bar(manifest_path, out_base):
    df = pd.read_csv(manifest_path, sep="\t")
    last = df.sort_values("timestamp").groupby("subject").tail(1)
    counts = last["status"].value_counts()
    order = [o for o in ("DONE", "RUNNING", "FAILED") if o in counts.index]
    colors = {"DONE": STATUS_GOOD, "RUNNING": STATUS_WARNING, "FAILED": STATUS_CRITICAL}
    vals = [int(counts[o]) for o in order]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(order, vals, color=[colors[o] for o in order], width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    fontsize=10, color=INK_PRIMARY)
    ax.set_ylabel("Subjects")
    ax.set_title(f"MRIQC completion status (n={len(last)})", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_official_group_report(skip=False):
    if skip:
        print("[official] --skip-official set, not running MRIQC's own group-level aggregation")
        return
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(APPTAINER_BIN), "run", "--cleanenv",
           "-B", f"{BIDS_ROOT}:/data:ro",
           "-B", f"{MRIQC_DIR}:/out",
           "-B", f"{TEMPLATEFLOW_CACHE}:/opt/templateflow",
           "--env", "TEMPLATEFLOW_HOME=/opt/templateflow",
           str(MRIQC_SIF), "/data", "/out", "group"]
    print("[official] running MRIQC group-level aggregation...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OFFICIAL_DIR / "group_level_stdout.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        print(f"[warn] MRIQC group-level exited {r.returncode}; see {OFFICIAL_DIR}/group_level_stdout.log")
        return
    copied = 0
    for pattern in ("group_*.tsv", "group_*.html"):
        for f in MRIQC_DIR.glob(pattern):
            shutil.copy2(f, OFFICIAL_DIR / f.name)
            copied += 1
    print(f"[official] copied {copied} group-level file(s) -> {OFFICIAL_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-official", action="store_true",
                     help="skip MRIQC's own apptainer-based group-level aggregation")
    args = ap.parse_args()

    for d in (TABLES_DIR, FIG_ANAT, FIG_FUNC, FIG_SUMMARY, OFFICIAL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    t1w = load_anat_table("sub-*/anat/*_T1w.json", "T1w")
    t2w = load_anat_table("sub-*/anat/*acq-highreshippo_T2w.json", "T2w")
    bold = load_bold_table()
    print(f"Loaded IQMs: T1w n={len(t1w)}, T2w n={len(t2w)}, bold n={len(bold)}")

    t1w_flags = compute_flags(t1w, list(ANAT_METRICS), group_col=None)
    t2w_flags = compute_flags(t2w, list(ANAT_METRICS), group_col=None)
    bold_flags = compute_flags(bold, list(BOLD_METRICS), group_col="run")

    t1w.to_csv(TABLES_DIR / "t1w_iqms.csv", index=False)
    t2w.to_csv(TABLES_DIR / "t2w_iqms.csv", index=False)
    bold.to_csv(TABLES_DIR / "bold_iqms.csv", index=False)
    t1w_flags.to_csv(TABLES_DIR / "t1w_flags_long.csv", index=False)
    t2w_flags.to_csv(TABLES_DIR / "t2w_flags_long.csv", index=False)
    bold_flags.to_csv(TABLES_DIR / "bold_flags_long.csv", index=False)

    all_flags = pd.concat([
        t1w_flags.assign(modality="T1w"),
        t2w_flags.assign(modality="T2w"),
        bold_flags.assign(modality="bold"),
    ], ignore_index=True)
    all_flags[all_flags["flagged"]].sort_values(["subject", "modality", "metric"]).to_csv(
        TABLES_DIR / "flagged_subjects.csv", index=False)

    n_assessed = len(ANAT_METRICS) * 2 + len(BOLD_METRICS) * 2  # T1w + T2w + bold x2 runs
    flag_counts = all_flags.groupby("subject")["flagged"].sum()

    make_metric_grid(t1w_flags, ANAT_METRICS, ["all"], [SERIES_BLUE], ncols=4,
                      out_base=str(FIG_ANAT / "t1w_iqm_overview"),
                      suptitle=f"T1w image-quality metrics (MRIQC), n={t1w['subject'].nunique()} subjects")

    make_metric_grid(t2w_flags, ANAT_METRICS, ["all"], [SERIES_BLUE], ncols=4,
                      out_base=str(FIG_ANAT / "t2w_iqm_overview"),
                      suptitle=f"T2w hippocampal-slab image-quality metrics, n={t2w['subject'].nunique()} subjects",
                      note=("Caution: acq-highreshippo T2w is a partial-FOV (~60mm) hippocampal slab, not a "
                            "whole-brain scan -- MNI-registration-based IQMs are less reliable here than for T1w."))

    make_metric_grid(bold_flags, BOLD_METRICS, ["run-1", "run-2"], [SERIES_BLUE, SERIES_ORANGE], ncols=4,
                      out_base=str(FIG_FUNC / "bold_iqm_overview"),
                      suptitle=f"Resting-state BOLD image-quality metrics, n={bold['subject'].nunique()} subjects x 2 runs")

    make_flag_summary_figure(flag_counts, n_assessed, str(FIG_SUMMARY / "flagged_subjects_summary"))
    if MANIFEST.exists():
        make_status_bar(MANIFEST, str(FIG_SUMMARY / "pipeline_completion_status"))

    run_official_group_report(skip=args.skip_official)

    print()
    print(f"Flagged >=1 IQM: {int((flag_counts > 0).sum())} / {flag_counts.shape[0]} subjects "
          f"(out of {n_assessed} metrics assessed per subject)")
    print(f"Tables  -> {TABLES_DIR}")
    print(f"Figures -> {OUT_ROOT / 'figures'}")
    print(f"Official MRIQC group report -> {OFFICIAL_DIR}")


if __name__ == "__main__":
    main()
