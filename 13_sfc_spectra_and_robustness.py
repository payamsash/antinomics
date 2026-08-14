#!/usr/bin/env python3
"""
Extends 12_sfc_group_validation.py with the two remaining Fig-2-class robustness
pieces the master plan calls for:

  (A) Harmonic spectra, group-level ("a spectrum, not a scalar" -- §2.3). Each
      subject's cumulative graph-Fourier energy curve (already computed by
      11_run_sfc.py, one row per harmonic) is interpolated onto a common
      normalized-rank grid (harmonic_idx / (n-1), since n varies per subject
      after node pruning) and averaged across the group, with a band. This is
      a legitimate whole-connectome spectral summary; it is NOT a per-region
      spectral decomposition -- reconstructing per-region-per-harmonic energy
      from PSD alone is not actually well-defined (PSD[k] is already a
      whole-connectome quantity, mixing across nodes via the GFT), so this
      script does not attempt that.

  (B) Robustness across the 3 non-primary SC-weight x FC-variant combinations
      (master plan: "reporting their convergence is far stronger than any
      single index"). For each combo: correlation of its group-mean SDI map
      against the primary (weight-sift_fc-nogsr) map, and the exact same
      Moran-null hierarchy significance test from script 12, to confirm the
      sensory->association effect replicates rather than being an artifact of
      one particular SC-weight/denoising choice.

Run with:
    /home/ubuntu/volume/miniconda3/envs/tinnorm/bin/python3 13_sfc_spectra_and_robustness.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainspace.null_models import MoranRandomization
from scipy.stats import pearsonr, spearmanr

DERIV = Path("/home/ubuntu/volume/antinomics/derivatives")
SFC_DIR = DERIV / "sfc"
GROUP_DIR = SFC_DIR / "group"
FIG_DIR = GROUP_DIR / "figures"
ATLASES = ("4S456Parcels", "4S856Parcels")
ATLAS_DSEG_TSV = {
    a: DERIV / "xcp_d" / "sourcedata" / "atlases" / "tpl-MNI152NLin6Asym" / f"tpl-MNI152NLin6Asym_atlas-{a}_res-2_dseg.tsv"
    for a in ATLASES
}

SC_WEIGHTS = ("sift", "count")
FC_VARIANTS = ("nogsr", "gsr")
PRIMARY_COMBO = ("sift", "nogsr")
HIERARCHY_ORDER = {"Vis": 1, "SomMot": 2, "DorsAttn": 3, "SalVentAttn": 4, "Limbic": 5, "Cont": 6, "Default": 7}
MIN_VALID_FRACTION = 0.5
N_PERM = 1000
MORAN_SEED = 20260814
SPECTRUM_GRID = np.linspace(0, 1, 201)

COMBO_COLORS = {
    ("sift", "nogsr"): "#0d366b",   # primary -- darkest
    ("sift", "gsr"): "#3987e5",
    ("count", "nogsr"): "#e8874a",
    ("count", "gsr"): "#f2b880",
}


def completed_subjects() -> list[str]:
    return sorted(p.stem.replace("_sfc-complete", "") for p in SFC_DIR.glob("sub-*_sfc-complete.json"))


def canonical_and_distmat(atlas: str) -> tuple[list[str], np.ndarray]:
    centroids = pd.read_csv(GROUP_DIR / f"atlas-{atlas}_centroids.tsv", sep="\t").sort_values("index")
    canonical = centroids["name"].tolist()
    distmat = np.load(GROUP_DIR / f"atlas-{atlas}_distmat.npy")
    return canonical, distmat


def moran_null_correlations(distmat: np.ndarray, observed_map: np.ndarray, hierarchy_rank: np.ndarray, n_perm: int) -> np.ndarray:
    dist = distmat.astype("float64").copy()
    np.fill_diagonal(dist, 1.0)
    W = dist ** -1
    mr = MoranRandomization(procedure="singleton", spectrum="nonzero", joint=True, n_rep=n_perm, tol=1e-6, random_state=MORAN_SEED)
    mr.fit(W)
    surrogates = mr.randomize(observed_map)
    return np.array([spearmanr(hierarchy_rank, surrogates[i]).statistic for i in range(n_perm)])


def group_mean_sdi_map(atlas: str, weight: str, variant: str, subjects: list[str], canonical: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (group_mean, valid_fraction) over `canonical` order."""
    col = f"sdi_weight-{weight}_fc-{variant}"
    stack = []
    for sub in subjects:
        path = SFC_DIR / sub / f"atlas-{atlas}" / f"{sub}_atlas-{atlas}_desc-sdi.tsv"
        if not path.exists():
            continue
        s = pd.read_csv(path, sep="\t").set_index("region_label")[col].reindex(canonical)
        stack.append(s.to_numpy(dtype=float))
    stack = np.array(stack)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    valid_frac = np.mean(~np.isnan(stack), axis=0)
    return mean, valid_frac


# --------------------------------------------------------------------------
# Part A: group-level harmonic spectra
# --------------------------------------------------------------------------

def load_subject_spectrum_curve(sub: str, atlas: str, weight: str, variant: str) -> tuple[np.ndarray, int, float] | None:
    path = SFC_DIR / sub / f"atlas-{atlas}" / f"{sub}_atlas-{atlas}_weight-{weight}_fc-{variant}_desc-harmonicspectrum.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    n = len(df)
    if n < 3:
        return None
    x = df["harmonic_idx"].to_numpy() / (n - 1)
    y = df["cumulative_frac"].to_numpy()
    y_grid = np.interp(SPECTRUM_GRID, x, y)

    sdi_path = SFC_DIR / sub / f"atlas-{atlas}" / f"{sub}_atlas-{atlas}_desc-sdi.tsv"
    nc_col = f"nc_weight-{weight}_fc-{variant}"
    nc = pd.read_csv(sdi_path, sep="\t")[nc_col].iloc[0]
    return y_grid, n, nc / (n - 1)


def spectra_summary(atlas: str, subjects: list[str]) -> dict:
    curves = {}
    nc_fracs = {}
    for variant in FC_VARIANTS:
        rows, ncs = [], []
        for sub in subjects:
            r = load_subject_spectrum_curve(sub, atlas, "sift", variant)
            if r is None:
                continue
            y_grid, _, nc_frac = r
            rows.append(y_grid)
            ncs.append(nc_frac)
        arr = np.array(rows)
        curves[variant] = {"mean": arr.mean(axis=0), "lo": np.percentile(arr, 2.5, axis=0), "hi": np.percentile(arr, 97.5, axis=0), "n": len(rows)}
        nc_fracs[variant] = {"mean": float(np.mean(ncs)), "std": float(np.std(ncs))}

    out = pd.DataFrame({"normalized_harmonic_rank": SPECTRUM_GRID})
    for variant in FC_VARIANTS:
        out[f"mean_cumfrac_{variant}"] = curves[variant]["mean"]
        out[f"ci_lo_{variant}"] = curves[variant]["lo"]
        out[f"ci_hi_{variant}"] = curves[variant]["hi"]
    out.to_csv(GROUP_DIR / f"atlas-{atlas}_group-harmonic-spectrum.tsv", sep="\t", index=False)

    print(f"[{atlas}] mean Nc/n: nogsr={nc_fracs['nogsr']['mean']:.3f}+/-{nc_fracs['nogsr']['std']:.3f}, "
          f"gsr={nc_fracs['gsr']['mean']:.3f}+/-{nc_fracs['gsr']['std']:.3f} (n={curves['nogsr']['n']} subjects)")

    return {"curves": curves, "nc_fracs": nc_fracs}


def plot_spectra(all_spectra: dict, subjects: list[str]) -> None:
    fig, axes = plt.subplots(1, len(ATLASES), figsize=(6.5 * len(ATLASES), 4.5), squeeze=False)
    colors = {"nogsr": "#0d366b", "gsr": "#e8874a"}
    for ax, atlas in zip(axes[0], ATLASES):
        d = all_spectra[atlas]["curves"]
        for variant in FC_VARIANTS:
            c = colors[variant]
            ax.fill_between(SPECTRUM_GRID, d[variant]["lo"], d[variant]["hi"], color=c, alpha=0.15, linewidth=0)
            ax.plot(SPECTRUM_GRID, d[variant]["mean"], color=c, linewidth=2, label=f"FC={variant} (n={d[variant]['n']})")
        nc = all_spectra[atlas]["nc_fracs"]["nogsr"]["mean"]
        ax.axvline(nc, color="#8a93a1", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="#c3c9d1", linewidth=1, zorder=0)
        ax.text(nc + 0.01, 0.06, f"mean Nc/n={nc:.2f}", fontsize=8.5, color="#5b6472")
        ax.set_title(atlas, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("normalized harmonic rank (low freq -> high freq)", fontsize=9.5, color="#5b6472")
        ax.set_ylabel("cumulative graph-Fourier energy fraction", fontsize=9.5, color="#5b6472")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Group-average connectome harmonic energy spectrum (n={len(subjects)} subjects, primary SC weight)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "harmonic_spectrum_group.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Wrote figure: {FIG_DIR / 'harmonic_spectrum_group.png'}")


# --------------------------------------------------------------------------
# Part B: robustness across SC-weight x FC-variant combinations
# --------------------------------------------------------------------------

def robustness_for_atlas(atlas: str, subjects: list[str]) -> pd.DataFrame:
    canonical, distmat = canonical_and_distmat(atlas)
    dseg = pd.read_csv(ATLAS_DSEG_TSV[atlas], sep="\t").set_index("name")
    network_label = dseg["network_label"].reindex(canonical)
    hierarchy_rank_full = network_label.map(HIERARCHY_ORDER).to_numpy(dtype=float)
    cortical = ~np.isnan(hierarchy_rank_full)

    primary_mean, primary_valid = group_mean_sdi_map(atlas, *PRIMARY_COMBO, subjects, canonical)

    rows = []
    for weight in SC_WEIGHTS:
        for variant in FC_VARIANTS:
            combo = (weight, variant)
            mean_map, valid_frac = group_mean_sdi_map(atlas, weight, variant, subjects, canonical)

            both_valid = ~np.isnan(mean_map) & ~np.isnan(primary_mean)
            pear = pearsonr(mean_map[both_valid], primary_mean[both_valid]).statistic if combo != PRIMARY_COMBO else 1.0
            spear = spearmanr(mean_map[both_valid], primary_mean[both_valid]).statistic if combo != PRIMARY_COMBO else 1.0

            mask = cortical & (valid_frac >= MIN_VALID_FRACTION) & ~np.isnan(mean_map)
            idx = np.where(mask)[0]
            gm, hr, dm = mean_map[idx], hierarchy_rank_full[idx], distmat[np.ix_(idx, idx)]
            observed = spearmanr(hr, gm).statistic
            null_corrs = moran_null_correlations(dm, gm, hr, N_PERM)
            p_two_sided = float((np.abs(null_corrs) >= abs(observed)).mean())

            rows.append({
                "atlas": atlas, "weight": weight, "fc_variant": variant,
                "is_primary": combo == PRIMARY_COMBO,
                "n_regions_in_map": int(both_valid.sum()),
                "pearson_r_vs_primary": pear, "spearman_r_vs_primary": spear,
                "hierarchy_spearman_rho": float(observed), "hierarchy_moran_p": p_two_sided,
                "n_regions_hierarchy_test": int(len(idx)),
            })
            print(f"[{atlas}] weight={weight} fc={variant}: r_vs_primary(pearson)={pear:.3f} spearman={spear:.3f} | "
                  f"hierarchy rho={observed:.3f} p={p_two_sided:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(GROUP_DIR / f"atlas-{atlas}_robustness.tsv", sep="\t", index=False)
    return df


def plot_robustness(all_robustness: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(ATLASES), figsize=(6 * len(ATLASES), 4), squeeze=False)
    for ax, atlas in zip(axes[0], ATLASES):
        df = all_robustness[atlas].copy()
        df["label"] = df.apply(lambda r: f"{r['weight']}+{r['fc_variant']}" + ("\n(primary)" if r["is_primary"] else ""), axis=1)
        colors = [COMBO_COLORS[(r.weight, r.fc_variant)] for r in df.itertuples()]
        bars = ax.bar(df["label"], df["hierarchy_spearman_rho"], color=colors, width=0.55)
        for bar, p in zip(bars, df["hierarchy_moran_p"]):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"p={p:.3f}\n{sig}",
                    ha="center", va="bottom", fontsize=8.5, color="#5b6472")
        ax.axhline(0, color="#c3c9d1", linewidth=1)
        ax.set_title(atlas, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("hierarchy Spearman ρ\n(SDI vs unimodal→transmodal rank)", fontsize=9, color="#5b6472")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(0.35, df["hierarchy_spearman_rho"].max() + 0.12))

    fig.suptitle("Sensory-association hierarchy effect replicates across SC-weight x FC-denoising choices", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "sfc_robustness_combos.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Wrote figure: {FIG_DIR / 'sfc_robustness_combos.png'}")


def main() -> None:
    subjects = completed_subjects()
    print(f"{len(subjects)} subjects with completed SFC output\n")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Part A: harmonic spectra ===")
    all_spectra = {atlas: spectra_summary(atlas, subjects) for atlas in ATLASES}
    plot_spectra(all_spectra, subjects)

    print("\n=== Part B: robustness across SC-weight x FC-variant ===")
    all_robustness = {atlas: robustness_for_atlas(atlas, subjects) for atlas in ATLASES}
    plot_robustness(all_robustness)

    print("\nDone.")


if __name__ == "__main__":
    main()
