#!/usr/bin/env python3
"""
Group-level SFC validation -- tests whether the pipeline recovers the known
sensory-to-association SDI/tethering gradient (Preti & Van De Ville 2019;
Vazquez-Rodriguez et al. 2019 PNAS) before any group-difference or normative
claim is trusted. This is the master plan's Fig 2 ("SFC is regionally
structured and reproducible... establishing the measure before any group
claim") plus its §6 requirement that any brain-map correspondence use a
spatial-autocorrelation-preserving null model.

For each atlas (4S456Parcels, 4S856Parcels):
  1. Group-mean SDI (primary weight-sift_fc-nogsr) and tethering-R^2 per region,
     across all subjects with completed SFC output.
  2. Restrict to cortical (Schaefer-derived, has a Yeo-7 network_label) regions
     with a valid group mean in >=50% of subjects.
  3. Test Spearman correlation between each region's Yeo-network unimodal-to-
     transmodal hierarchy rank (Vis=1, SomMot=2, DorsAttn=3, SalVentAttn=4,
     Limbic=5, Cont=6, Default=7) and its group-mean SDI / tethering-R^2.
  4. Significance via Moran spectral randomization nulls (brainspace, using the
     centroid/distance-matrix scaffolding 11_run_sfc.py already cached) --
     1000 spatially-autocorrelation-preserving surrogate maps, two-sided
     empirical p-value.
  5. A network-level boxplot figure (ordered sensory->association) as the
     visual validation companion to the numeric test.

Run with:
    /home/ubuntu/volume/miniconda3/envs/tinnorm/bin/python3 12_sfc_group_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainspace.null_models import MoranRandomization
from scipy.stats import spearmanr

DERIV = Path("/home/ubuntu/volume/antinomics/derivatives")
SFC_DIR = DERIV / "sfc"
GROUP_DIR = SFC_DIR / "group"
ATLASES = ("4S456Parcels", "4S856Parcels")
ATLAS_DSEG_TSV = {
    a: DERIV / "xcp_d" / "sourcedata" / "atlases" / "tpl-MNI152NLin6Asym" / f"tpl-MNI152NLin6Asym_atlas-{a}_res-2_dseg.tsv"
    for a in ATLASES
}

PRIMARY_SDI_COL = "sdi_weight-sift_fc-nogsr"
MIN_VALID_FRACTION = 0.5  # region must be included in >=50% of subjects to enter the validation
HIERARCHY_ORDER = {"Vis": 1, "SomMot": 2, "DorsAttn": 3, "SalVentAttn": 4, "Limbic": 5, "Cont": 6, "Default": 7}
N_PERM = 1000
MORAN_SEED = 20260814

OUT_DIR = GROUP_DIR
FIG_DIR = SFC_DIR / "group" / "figures"


def completed_subjects() -> list[str]:
    return sorted(p.stem.replace("_sfc-complete", "") for p in SFC_DIR.glob("sub-*_sfc-complete.json"))


def load_region_series(sub: str, atlas: str, canonical: list[str], value_file_suffix: str, value_col: str, key_col: str = "region_label") -> np.ndarray:
    path = SFC_DIR / sub / f"atlas-{atlas}" / f"{sub}_atlas-{atlas}_{value_file_suffix}"
    df = pd.read_csv(path, sep="\t")
    s = df.set_index(key_col)[value_col].reindex(canonical)
    return s.to_numpy(dtype=float)


def moran_null_correlations(distmat: np.ndarray, observed_map: np.ndarray, hierarchy_rank: np.ndarray, n_perm: int) -> np.ndarray:
    dist = distmat.astype("float64").copy()
    np.fill_diagonal(dist, 1.0)
    W = dist ** -1
    mr = MoranRandomization(procedure="singleton", spectrum="nonzero", joint=True, n_rep=n_perm, tol=1e-6, random_state=MORAN_SEED)
    mr.fit(W)
    surrogates = mr.randomize(observed_map)  # (n_perm, n_nodes)
    null_corrs = np.array([spearmanr(hierarchy_rank, surrogates[i]).statistic for i in range(n_perm)])
    return null_corrs


def validate_atlas(atlas: str, subjects: list[str]) -> dict:
    centroids = pd.read_csv(GROUP_DIR / f"atlas-{atlas}_centroids.tsv", sep="\t").sort_values("index")
    canonical = centroids["name"].tolist()
    distmat_full = np.load(GROUP_DIR / f"atlas-{atlas}_distmat.npy")

    dseg = pd.read_csv(ATLAS_DSEG_TSV[atlas], sep="\t").set_index("name")
    network_label = dseg["network_label"].reindex(canonical)

    sdi_stack, teth_stack = [], []
    for sub in subjects:
        try:
            sdi_stack.append(load_region_series(sub, atlas, canonical, "desc-sdi.tsv", PRIMARY_SDI_COL))
            teth_stack.append(load_region_series(sub, atlas, canonical, "desc-tethering-rsq.tsv", "tethering_adj_rsq"))
        except FileNotFoundError:
            continue
    sdi_stack = np.array(sdi_stack)   # (n_subj, n_regions)
    teth_stack = np.array(teth_stack)
    n_subj = len(sdi_stack)

    group_sdi_mean = np.nanmean(sdi_stack, axis=0)
    group_sdi_std = np.nanstd(sdi_stack, axis=0)
    group_teth_mean = np.nanmean(teth_stack, axis=0)
    valid_frac_sdi = np.mean(~np.isnan(sdi_stack), axis=0)
    valid_frac_teth = np.mean(~np.isnan(teth_stack), axis=0)

    pd.DataFrame({
        "region_label": canonical, "network_label": network_label.to_numpy(),
        "group_mean_sdi": group_sdi_mean, "group_std_sdi": group_sdi_std, "valid_fraction": valid_frac_sdi,
    }).to_csv(OUT_DIR / f"atlas-{atlas}_group-sdi-by-region.tsv", sep="\t", index=False)
    pd.DataFrame({
        "region_label": canonical, "network_label": network_label.to_numpy(),
        "group_mean_tethering_rsq": group_teth_mean, "valid_fraction": valid_frac_teth,
    }).to_csv(OUT_DIR / f"atlas-{atlas}_group-tethering-by-region.tsv", sep="\t", index=False)

    hierarchy_rank_full = network_label.map(HIERARCHY_ORDER).to_numpy(dtype=float)
    cortical = ~np.isnan(hierarchy_rank_full)

    results = {}
    for metric_name, group_mean, valid_frac in (("sdi", group_sdi_mean, valid_frac_sdi), ("tethering", group_teth_mean, valid_frac_teth)):
        mask = cortical & (valid_frac >= MIN_VALID_FRACTION) & ~np.isnan(group_mean)
        idx = np.where(mask)[0]
        gm = group_mean[idx]
        hr = hierarchy_rank_full[idx]
        dm = distmat_full[np.ix_(idx, idx)]

        observed = spearmanr(hr, gm).statistic
        null_corrs = moran_null_correlations(dm, gm, hr, N_PERM)
        p_two_sided = float((np.abs(null_corrs) >= abs(observed)).mean())

        results[metric_name] = {
            "n_regions": int(len(idx)), "n_subjects": n_subj,
            "observed_spearman_rho": float(observed),
            "null_mean": float(null_corrs.mean()), "null_std": float(null_corrs.std()),
            "p_two_sided_moran": p_two_sided, "n_perm": N_PERM,
        }
        print(f"[{atlas}] {metric_name}: rho={observed:.3f} (n={len(idx)} regions, {n_subj} subj), "
              f"Moran-null p={p_two_sided:.4f} (null mean={null_corrs.mean():.3f}+/-{null_corrs.std():.3f})")

    with open(OUT_DIR / f"atlas-{atlas}_hierarchy-validation.json", "w") as f:
        json.dump(results, f, indent=2)

    return {
        "atlas": atlas, "canonical": canonical, "network_label": network_label,
        "group_sdi_mean": group_sdi_mean, "group_teth_mean": group_teth_mean,
        "cortical": cortical, "results": results,
    }


def make_figure(atlas_results: dict, subjects: list[str]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = list(HIERARCHY_ORDER.keys())
    # Ordinal sequential ramp (unimodal->transmodal), light->dark -- encodes the same
    # hierarchy the x-axis position already does, reinforcing rather than decorating it.
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    point_color = "#2b2f36"
    text_muted = "#5b6472"

    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": "#8a93a1", "axes.labelcolor": "#2b2f36"})
    fig, axes = plt.subplots(1, len(atlas_results), figsize=(6.5 * len(atlas_results), 4.5), squeeze=False)
    for ax, (atlas, d) in zip(axes[0], atlas_results.items()):
        canonical, cortical = d["canonical"], d["cortical"]
        net = d["network_label"].to_numpy()
        gm = d["group_sdi_mean"]
        df = pd.DataFrame({"network": net[cortical], "sdi": gm[cortical]})
        df = df[df["network"].isin(order)]

        ax.axhline(0, color="#c3c9d1", linewidth=1, zorder=0)
        sns.boxplot(
            data=df, x="network", y="sdi", order=order, hue="network", hue_order=order, legend=False, dodge=False,
            ax=ax, showfliers=False, palette=ramp, linewidth=1, width=0.6,
            boxprops={"edgecolor": "#20242b"}, medianprops={"color": "#20242b", "linewidth": 1.5},
            whiskerprops={"color": "#5b6472"}, capprops={"color": "#5b6472"},
        )
        sns.stripplot(data=df, x="network", y="sdi", order=order, ax=ax, color=point_color, size=2.8, alpha=0.35, jitter=0.2)

        rho = d["results"]["sdi"]["observed_spearman_rho"]
        p = d["results"]["sdi"]["p_two_sided_moran"]
        n = d["results"]["sdi"]["n_regions"]
        ax.set_title(
            f"{atlas}  (n={n} regions)\n" + r"$\mathdefault{Spearman\ \rho}$" + f"={rho:.2f}, Moran-null p={p:.4f}",
            fontsize=11, loc="left", linespacing=1.6,
        )
        ax.title.set_fontweight("bold")
        ax.set_xlabel("Yeo-7 network, unimodal → transmodal", color=text_muted, fontsize=9.5)
        ax.set_ylabel("group-mean SDI (region-level, log₂ decoupled/coupled)", color=text_muted, fontsize=9.5)
        ax.tick_params(axis="x", rotation=25, colors="#2b2f36")
        ax.tick_params(axis="y", colors="#2b2f36")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c9d1")
        ax.spines["bottom"].set_color("#c3c9d1")

    fig.suptitle(f"Structural Decoupling Index recovers the sensory-to-association hierarchy (n={len(subjects)} subjects)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "sdi_hierarchy_validation.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Wrote figure: {FIG_DIR / 'sdi_hierarchy_validation.png'}")


def main() -> None:
    subjects = completed_subjects()
    print(f"{len(subjects)} subjects with completed SFC output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    atlas_results = {}
    for atlas in ATLASES:
        atlas_results[atlas] = validate_atlas(atlas, subjects)

    make_figure(atlas_results, subjects)
    print("\nDone.")


if __name__ == "__main__":
    main()
