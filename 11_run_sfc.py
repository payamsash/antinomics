#!/usr/bin/env python3
"""
Structure-Function Coupling (SFC) module -- Phase 1 static core.

Implements §2.1 (Structural Decoupling Index), §2.3 (harmonic spectra), and §2.4
(SC/FC gradient tethering) of tinnitus_SFC_analysis_plan-4.md, for every subject
with ready_for_sfc=="yes" in derivatives/pipeline_status.tsv, on both the
4S456Parcels and 4S856Parcels atlases already computed by QSIRecon (SC) and
XCP-D (FC).

Deliberately NOT wrapping nigsp (MIPLabCH/nigsp): the graph-Laplacian
eigendecomposition and SDI computation are implemented directly with
numpy/scipy for full transparency/auditability, following Preti & Van De Ville
(2019, Nat Commun) precisely. Two concrete, data-confirmed corrections relative
to a naive reading of that method / of nigsp's own "legacy" code path:
  - SC matrices have real nonzero-diagonal self-loops (MRtrix radial-search
    node assignment artifact) that MUST be zeroed before Laplacian construction.
  - The coupled/decoupled cutoff harmonic Nc is a per-subject median-split of
    the empirical signal's own graph-Fourier power spectrum (trapezoidal AUC
    crossing 50%) -- no surrogate/randomized-SC null is involved in choosing Nc
    (that machinery in nigsp is a separate, optional post-hoc significance
    test, not part of computing SDI itself, and is out of scope here).

Regional SC/FC "tethering" is computed two ways, since the master plan's own
citation (Vazquez-Rodriguez et al. 2019, PNAS) turns out to define it as a
per-region multilinear regression (FC profile ~ path length + communicability +
Euclidean distance), NOT a gradient-embedding distance -- implemented as
primary. A BrainSpace gradient-distance score is also computed as a clearly
labeled secondary/exploratory cross-check.

Run with:
    /home/ubuntu/volume/miniconda3/envs/tinnorm/bin/python3 11_run_sfc.py [sub-xxxx ...]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import multiprocessing as mp
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat
from scipy.linalg import eigh, expm
from scipy.ndimage import center_of_mass
from scipy.sparse.csgraph import shortest_path

from brainspace.gradient import GradientMaps

BIDS_ROOT = Path("/home/ubuntu/volume/antinomics")
DERIV = BIDS_ROOT / "derivatives"

SC_ROOT = DERIV / "qsirecon" / "derivatives" / "qsirecon-MRtrix3_fork-SS3T_act-HSVS"
FC_ROOTS = {"nogsr": DERIV / "xcp_d", "gsr": DERIV / "xcp_d_gsr"}
ATLASES = ("4S456Parcels", "4S856Parcels")
ATLAS_TPL_DIR = DERIV / "xcp_d" / "sourcedata" / "atlases" / "tpl-MNI152NLin6Asym"
ATLAS_DSEG_TSV = {a: ATLAS_TPL_DIR / f"tpl-MNI152NLin6Asym_atlas-{a}_res-2_dseg.tsv" for a in ATLASES}
ATLAS_DSEG_NII = {a: ATLAS_TPL_DIR / f"tpl-MNI152NLin6Asym_atlas-{a}_res-2_dseg.nii.gz" for a in ATLASES}

SC_WEIGHT_KEYS = {"sift": "sift_invnodevol_radius2_count_connectivity", "count": "radius2_count_connectivity"}
FC_VARIANTS = ("nogsr", "gsr")
PRIMARY_WEIGHT = "sift"
PRIMARY_FC = "nogsr"

COVERAGE_THRESHOLD = 0.5
FLAGGED_EXCLUSION_FRACTION = 0.25
CONSISTENCY_RETAIN_PERCENTILE = 75  # keep the 75% most-consistent (lowest-CV) present edges

SFC_DIR = DERIV / "sfc"
GROUP_DIR = SFC_DIR / "group"
LOG_DIR = DERIV / "logs" / "11_sfc"
MANIFEST = LOG_DIR / "sfc_status.tsv"
PIPELINE_STATUS_TSV = DERIV / "pipeline_status.tsv"

N_JOBS = 16
FORCE_RECOMPUTE = False

_CANONICAL: dict[str, list[str]] | None = None
_DISTMATS: dict[str, np.ndarray] | None = None
_CONSISTENCY: dict[tuple[str, str], np.ndarray] | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_manifest(key: str, status: str, detail: str = "") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not MANIFEST.exists()
    with open(MANIFEST, "a") as f:
        if is_new:
            f.write("subject\tstatus\ttimestamp\tdetail\n")
        detail = detail.replace("\t", " ").replace("\n", " ")
        f.write(f"{key}\t{status}\t{now_iso()}\t{detail}\n")


def ready_subjects() -> list[str]:
    df = pd.read_csv(PIPELINE_STATUS_TSV, sep="\t")
    return sorted(df.loc[df["ready_for_sfc"] == "yes", "subject"])


def reindex_to_canonical(labels: list[str], canonical: list[str]) -> np.ndarray:
    if set(labels) != set(canonical):
        missing = sorted(set(canonical) - set(labels))
        extra = sorted(set(labels) - set(canonical))
        raise ValueError(f"label set mismatch: missing={missing[:5]}.. extra={extra[:5]}..")
    pos = {lab: i for i, lab in enumerate(labels)}
    return np.array([pos[lab] for lab in canonical])


# --------------------------------------------------------------------------
# Group-level Step 0: centroids, distance matrix, consistency-based edge mask
# --------------------------------------------------------------------------

def compute_centroids(atlas: str) -> pd.DataFrame:
    out = GROUP_DIR / f"atlas-{atlas}_centroids.tsv"
    if out.exists() and not FORCE_RECOMPUTE:
        return pd.read_csv(out, sep="\t")

    dseg_tsv = pd.read_csv(ATLAS_DSEG_TSV[atlas], sep="\t")
    img = nib.load(ATLAS_DSEG_NII[atlas])
    data = np.asarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    affine = img.affine

    rows = []
    n_outside = 0
    for _, r in dseg_tsv.sort_values("index").iterrows():
        idx, name = int(r["index"]), r["name"]
        mask = data == idx
        if not mask.any():
            raise RuntimeError(f"atlas {atlas}: label {idx} ({name}) has zero voxels in {ATLAS_DSEG_NII[atlas].name}")
        com_vox = center_of_mass(mask)
        world = nib.affines.apply_affine(affine, com_vox)
        vox_round = tuple(int(round(c)) for c in com_vox)
        inside = all(0 <= v < s for v, s in zip(vox_round, data.shape)) and data[vox_round] == idx
        if not inside:
            n_outside += 1
        rows.append({"index": idx, "name": name, "x_mm": world[0], "y_mm": world[1], "z_mm": world[2], "com_inside_parcel": bool(inside)})

    if n_outside:
        print(f"[centroids] atlas={atlas}: {n_outside} parcels have a centroid-of-mass outside their own labeled voxels (split/irregular parcels)")

    df = pd.DataFrame(rows)
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    return df


def compute_distmat(atlas: str, centroids: pd.DataFrame) -> np.ndarray:
    out = GROUP_DIR / f"atlas-{atlas}_distmat.npy"
    if out.exists() and not FORCE_RECOMPUTE:
        return np.load(out)
    xyz = centroids[["x_mm", "y_mm", "z_mm"]].to_numpy()
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    np.save(out, d)
    return d


def compute_consistency_mask(atlas: str, weight: str, subjects: list[str], canonical: list[str]) -> np.ndarray:
    out = GROUP_DIR / f"atlas-{atlas}_weight-{weight}_consistency_mask.npz"
    if out.exists() and not FORCE_RECOMPUTE:
        return np.load(out)["mask"]

    mats = []
    for sub in subjects:
        mat_path = SC_ROOT / sub / "dwi" / f"{sub}_space-ACPC_connectivity.mat"
        if not mat_path.exists():
            continue
        m = loadmat(mat_path)
        labels = [str(x[0]) for x in m[f"atlas_{atlas}_region_labels"][0]]
        idx = reindex_to_canonical(labels, canonical)
        A = m[f"atlas_{atlas}_{SC_WEIGHT_KEYS[weight]}"].astype(float)
        np.fill_diagonal(A, 0)
        mats.append(A[np.ix_(idx, idx)])

    stack = np.stack(mats, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    present = mean > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(present, std / mean, np.inf)
    threshold = np.percentile(cv[present], CONSISTENCY_RETAIN_PERCENTILE)
    mask = present & (cv <= threshold)
    mask = mask | mask.T

    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, mask=mask)
    print(
        f"[consistency] atlas={atlas} weight={weight}: retained {int(mask.sum())}/{int(present.sum())} "
        f"present edges ({mask.sum() / present.size:.1%} of all {present.size} possible node pairs), "
        f"from {len(mats)} subjects"
    )
    return mask


# --------------------------------------------------------------------------
# Core SDI / harmonic-spectrum math
# --------------------------------------------------------------------------

def build_laplacian(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = A.sum(axis=1)
    if (d <= 0).any():
        raise ValueError(f"{int((d <= 0).sum())} nodes have zero degree at Laplacian-construction time")
    dinv_sqrt = d ** -0.5
    L = np.eye(len(A)) - (dinv_sqrt[:, None] * A * dinv_sqrt[None, :])
    L = (L + L.T) / 2  # enforce exact symmetry against float roundoff
    evals, evecs = eigh(L)

    tol = 1e-6
    if evals[0] < -tol or evals[-1] > 2 + tol:
        raise ValueError(f"Laplacian eigenvalues outside expected [0,2] range: min={evals[0]:.4g} max={evals[-1]:.4g}")
    n_zero = int((evals < 1e-8).sum())
    if n_zero != 1:
        raise ValueError(f"expected exactly 1 near-zero eigenvalue (connected graph), found {n_zero} -- pruning left a disconnected graph")
    return evals, evecs


def gft_and_cutoff(evecs: np.ndarray, X: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """X: T x n (time x nodes, node order matching evecs rows). Returns (Nc, psd, Xhat)."""
    Xhat = evecs.T @ X.T  # n x T
    psd = (Xhat ** 2).mean(axis=1)  # n,
    n = len(psd)
    trap_incr = 0.5 * (psd[:-1] + psd[1:])
    cum_trap = np.concatenate([[0.0], np.cumsum(trap_incr)])
    total = cum_trap[-1]
    target = 0.5 * total
    candidates = np.where(cum_trap >= target)[0]
    Nc = int(candidates[0]) if candidates.size else n - 2
    Nc = int(np.clip(Nc, 1, n - 2))
    return Nc, psd, Xhat, cum_trap / total if total > 0 else cum_trap


def sdi_from_gft(evecs: np.ndarray, Xhat: np.ndarray, Nc: int) -> np.ndarray:
    X_low = evecs[:, :Nc] @ Xhat[:Nc, :]
    X_high = evecs[:, Nc:] @ Xhat[Nc:, :]
    low_norm = np.linalg.norm(X_low, axis=1)
    high_norm = np.linalg.norm(X_high, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(high_norm / low_norm)


def expand_to_full(values: np.ndarray, keep_idx: np.ndarray, n_full: int, fill=np.nan) -> np.ndarray:
    out = np.full(n_full, fill, dtype=float)
    out[keep_idx] = values
    return out


# --------------------------------------------------------------------------
# Tethering: primary (regional multilinear R^2) + secondary (gradient distance)
# --------------------------------------------------------------------------

def zscore(v: np.ndarray) -> np.ndarray:
    v = v.astype(float)
    return (v - np.nanmean(v)) / np.nanstd(v)


def ols_adj_rsq(y: np.ndarray, X: np.ndarray) -> float:
    Xd = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    r2 = 1 - ss_res / ss_tot
    n, p = len(y), X.shape[1]
    if n <= p + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def tethering_rsq(FC: np.ndarray, dist: np.ndarray, sc_binary: np.ndarray) -> np.ndarray:
    n = FC.shape[0]
    sp = shortest_path(csgraph=sparse.csr_matrix(sc_binary), method="D", unweighted=True)
    sp[np.isinf(sp)] = np.nan
    comm = expm(sc_binary)

    sp_z, comm_z, dist_z = zscore(sp), zscore(comm), zscore(dist)
    offdiag = ~np.eye(n, dtype=bool)

    rsq = np.full(n, np.nan)
    for j in range(n):
        rows = offdiag[:, j]
        y = FC[rows, j]
        X = np.column_stack([sp_z[rows, j], comm_z[rows, j], dist_z[rows, j]])
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() < 10:
            continue
        rsq[j] = ols_adj_rsq(y[valid], X[valid])
    return rsq


def gradient_tethering(SC: np.ndarray, FC: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gm = GradientMaps(n_components=3, approach="dm", kernel="normalized_angle", alignment="procrustes", random_state=0)
    gm.fit([SC, FC])
    sc_grad, fc_grad = gm.aligned_[0], gm.aligned_[1]
    from scipy.stats import spearmanr
    whole_map_corr = np.array([spearmanr(sc_grad[:, k], fc_grad[:, k]).statistic for k in range(3)])
    per_region_dist_g1 = np.abs(sc_grad[:, 0] - fc_grad[:, 0])
    return sc_grad, fc_grad, whole_map_corr, per_region_dist_g1


# --------------------------------------------------------------------------
# Per-subject processing
# --------------------------------------------------------------------------

def load_fc_variant(sub: str, atlas: str, variant: str, canonical: list[str]) -> dict:
    func_dir = FC_ROOTS[variant] / sub / "func"
    stem = f"{sub}_task-rest_space-MNI152NLin6Asym_atlas-{atlas}"
    relmat = pd.read_csv(func_dir / f"{stem}_stat-pearsoncorrelation_relmat.tsv", sep="\t", index_col=0)
    if set(relmat.index) != set(canonical) or set(relmat.columns) != set(canonical):
        raise ValueError(f"relmat ({variant}) label set mismatch")
    relmat = relmat.loc[canonical, canonical]

    ts = pd.read_csv(func_dir / f"{stem}_stat-mean_timeseries.tsv", sep="\t")
    if set(ts.columns) != set(canonical):
        raise ValueError(f"timeseries ({variant}) label set mismatch")
    ts = ts[list(canonical)]

    # Coverage is only written per-run (not for the concatenated series); a subject may have
    # only one usable run (e.g. sub-uaxk has run-2 only) -- combine whichever run(s) exist.
    cov_paths = sorted(func_dir.glob(f"{sub}_task-rest_run-*_space-MNI152NLin6Asym_atlas-{atlas}_stat-coverage_bold.tsv"))
    if not cov_paths:
        raise ValueError(f"no per-run coverage files found for ({variant})")
    cov_runs = []
    for p in cov_paths:
        c = pd.read_csv(p, sep="\t", index_col=0)["coverage"]
        if set(c.index) != set(canonical):
            raise ValueError(f"coverage ({variant}, {p.name}) label set mismatch")
        cov_runs.append(c.reindex(canonical).to_numpy())
    coverage_combined = np.minimum.reduce(cov_runs)

    return {"relmat": relmat.to_numpy(), "ts": ts.to_numpy(), "coverage": coverage_combined, "n_runs": len(cov_paths)}


def process_subject_atlas(sub: str, atlas: str, canonical: list[str], distmat: np.ndarray, consistency: dict[str, np.ndarray], out_dir: Path) -> dict:
    key = f"{sub}_atlas-{atlas}"
    update_manifest(key, "RUNNING")

    mat_path = SC_ROOT / sub / "dwi" / f"{sub}_space-ACPC_connectivity.mat"
    m = loadmat(mat_path)
    labels_sc = [str(x[0]) for x in m[f"atlas_{atlas}_region_labels"][0]]
    idx_sc = reindex_to_canonical(labels_sc, canonical)

    sc_full = {}
    for w, mat_key in SC_WEIGHT_KEYS.items():
        A = m[f"atlas_{atlas}_{mat_key}"].astype(float)
        np.fill_diagonal(A, 0)
        A = A[np.ix_(idx_sc, idx_sc)]
        A = A * consistency[w]  # group consistency-based edge thresholding
        sc_full[w] = A

    fc_data = {v: load_fc_variant(sub, atlas, v, canonical) for v in FC_VARIANTS}

    n_full = len(canonical)
    excluded_sc = sc_full[PRIMARY_WEIGHT].sum(axis=1) <= 0
    coverage_combined = np.minimum.reduce([d["coverage"] for d in fc_data.values()])
    excluded_fc = coverage_combined < COVERAGE_THRESHOLD
    excluded_combined = excluded_sc | excluded_fc
    keep = ~excluded_combined
    keep_idx = np.where(keep)[0]
    n_keep = int(keep.sum())
    excl_frac = 1 - n_keep / n_full

    pd.DataFrame({
        "region_label": canonical,
        "sc_isolated": excluded_sc,
        "fc_coverage_combined": coverage_combined,
        "excluded_fc": excluded_fc,
        "excluded_combined": excluded_combined,
    }).to_csv(out_dir / f"{sub}_atlas-{atlas}_desc-nodeexclusion.tsv", sep="\t", index=False)

    dist_sub = distmat[np.ix_(keep_idx, keep_idx)]

    # Per-weight consistency masks are computed independently (sift vs count streamline
    # weights have different CV distributions even for identical edge presence), so they can
    # occasionally diverge by a node or two after masking even though the raw tractogram's
    # edge-presence pattern is identical between weights. Handle defensively per weight rather
    # than failing the whole subject x atlas combination over 1-2 nodes: iteratively drop any
    # additionally-isolated nodes for that weight only, NaN-ing just that weight's SDI columns
    # at those nodes (node_included_combined / other weights' columns are unaffected).
    extra_notes: list[str] = []
    sdi_cols: dict[str, np.ndarray] = {}
    for w in SC_WEIGHT_KEYS:
        keep_idx_w = keep_idx
        A = sc_full[w][np.ix_(keep_idx_w, keep_idx_w)]
        while True:
            d = A.sum(axis=1)
            zero = d <= 0
            if not zero.any():
                break
            extra_notes.append(f"weight={w}: dropped {int(zero.sum())} node(s) additionally isolated after consistency masking")
            keep_idx_w = keep_idx_w[~zero]
            A = sc_full[w][np.ix_(keep_idx_w, keep_idx_w)]

        evals, evecs = build_laplacian(A)

        keep_w_mask = np.zeros(n_full, dtype=bool)
        keep_w_mask[keep_idx_w] = True
        pruned_labels = np.array([canonical[i] for i in keep_idx_w], dtype=object)
        np.savez(
            out_dir / f"{sub}_atlas-{atlas}_weight-{w}_desc-laplacian.npz",
            eigenvalues=evals, eigenvectors=evecs, included_node_mask=keep_w_mask, region_labels_pruned=pruned_labels,
        )

        for v in FC_VARIANTS:
            X = fc_data[v]["ts"][:, keep_idx_w]  # T x n_keep_w
            Nc, psd, Xhat, cum_frac = gft_and_cutoff(evecs, X)
            sdi = sdi_from_gft(evecs, Xhat, Nc)
            combo = f"weight-{w}_fc-{v}"
            sdi_cols[f"sdi_{combo}"] = expand_to_full(sdi, keep_idx_w, n_full)
            sdi_cols[f"nc_{combo}"] = np.full(n_full, Nc)

            pd.DataFrame({
                "harmonic_idx": np.arange(len(psd)), "eigenvalue": evals, "psd": psd, "cumulative_frac": cum_frac,
            }).to_csv(out_dir / f"{sub}_atlas-{atlas}_weight-{w}_fc-{v}_desc-harmonicspectrum.tsv", sep="\t", index=False)

    pd.DataFrame({
        "region_label": canonical,
        "node_included_sc": ~excluded_sc,
        "node_included_fc": ~excluded_fc,
        "node_included_combined": keep,
        **sdi_cols,
    }).to_csv(out_dir / f"{sub}_atlas-{atlas}_desc-sdi.tsv", sep="\t", index=False)

    # Gradients + tethering: primary SC weight x primary FC variant only
    A_primary = sc_full[PRIMARY_WEIGHT][np.ix_(keep_idx, keep_idx)]
    FC_primary = fc_data[PRIMARY_FC]["relmat"][np.ix_(keep_idx, keep_idx)]
    sc_bin = (A_primary > 0).astype(float)

    rsq = tethering_rsq(FC_primary, dist_sub, sc_bin)
    pd.DataFrame({
        "region_label": [canonical[i] for i in keep_idx],
        "tethering_adj_rsq": rsq,
    }).to_csv(out_dir / f"{sub}_atlas-{atlas}_desc-tethering-rsq.tsv", sep="\t", index=False)

    sc_grad, fc_grad, whole_map_corr, region_dist_g1 = gradient_tethering(A_primary, FC_primary)
    grad_df = pd.DataFrame({
        "region_label": [canonical[i] for i in keep_idx],
        "sc_gradient_1": sc_grad[:, 0], "sc_gradient_2": sc_grad[:, 1], "sc_gradient_3": sc_grad[:, 2],
        "fc_gradient_1": fc_grad[:, 0], "fc_gradient_2": fc_grad[:, 1], "fc_gradient_3": fc_grad[:, 2],
        "gradient_distance_g1": region_dist_g1,
    })
    grad_df.attrs["whole_map_spearman_g1_g2_g3"] = whole_map_corr.tolist()
    grad_df.to_csv(out_dir / f"{sub}_atlas-{atlas}_desc-gradients.tsv", sep="\t", index=False)

    status = "DONE_FLAGGED" if (excl_frac > FLAGGED_EXCLUSION_FRACTION or extra_notes) else "DONE"
    detail = f"excluded_fraction={excl_frac:.3f} n_keep={n_keep}/{n_full}"
    if extra_notes:
        detail += " | " + "; ".join(extra_notes)
    update_manifest(key, status, detail=detail)
    return {"status": status, "excluded_fraction": excl_frac, "n_keep": n_keep, "n_full": n_full}


def process_subject(sub: str) -> str:
    assert _CANONICAL is not None and _DISTMATS is not None and _CONSISTENCY is not None
    marker = SFC_DIR / f"{sub}_sfc-complete.json"

    if marker.exists() and not FORCE_RECOMPUTE:
        newest_input = (SC_ROOT / sub / "dwi" / f"{sub}_space-ACPC_connectivity.mat").stat().st_mtime
        for v, root in FC_ROOTS.items():
            p = root / sub / "func" / f"{sub}_task-rest_space-MNI152NLin6Asym_atlas-4S456Parcels_stat-pearsoncorrelation_relmat.tsv"
            if p.exists():
                newest_input = max(newest_input, p.stat().st_mtime)
        if marker.stat().st_mtime >= newest_input:
            return f"[skip] {sub} already complete"

    results = {}
    try:
        for atlas in ATLASES:
            out_dir = SFC_DIR / sub / f"atlas-{atlas}"
            out_dir.mkdir(parents=True, exist_ok=True)
            consistency = {w: _CONSISTENCY[(atlas, w)] for w in SC_WEIGHT_KEYS}
            results[atlas] = process_subject_atlas(sub, atlas, _CANONICAL[atlas], _DISTMATS[atlas], consistency, out_dir)
    except Exception as e:
        update_manifest(f"{sub}_atlas-{atlas}", "FAILED", detail=f"{type(e).__name__}: {e}")
        return f"[FAILED] {sub} ({atlas}): {e}\n{traceback.format_exc(limit=3)}"

    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w") as f:
        json.dump({"subject": sub, "completed_at": now_iso(), "atlases": results, "primary_combo": f"weight-{PRIMARY_WEIGHT}_fc-{PRIMARY_FC}"}, f, indent=2)
    return f"[done] {sub}: " + ", ".join(f"{a}={r['status']}" for a, r in results.items())


def main() -> None:
    global _CANONICAL, _DISTMATS, _CONSISTENCY

    argv_subjects = sys.argv[1:]
    subjects = argv_subjects if argv_subjects else ready_subjects()
    print(f"SFC: {len(subjects)} subjects")

    centroids = {a: compute_centroids(a) for a in ATLASES}
    _CANONICAL = {a: centroids[a].sort_values("index")["name"].tolist() for a in ATLASES}
    _DISTMATS = {a: compute_distmat(a, centroids[a]) for a in ATLASES}

    all_ready = ready_subjects()  # consistency mask always built from the full ready cohort, not a CLI subset
    _CONSISTENCY = {}
    for atlas in ATLASES:
        for w in SC_WEIGHT_KEYS:
            _CONSISTENCY[(atlas, w)] = compute_consistency_mask(atlas, w, all_ready, _CANONICAL[atlas])

    with mp.Pool(processes=N_JOBS) as pool:
        for msg in pool.imap_unordered(process_subject, subjects):
            print(msg)

    print("\nDone.")


if __name__ == "__main__":
    main()
