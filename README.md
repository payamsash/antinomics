# Antinomics

## Subcortical Connectivity in Tinnitus

This repository contains the analysis pipeline for:

> **"Multimodal Characterisation of Subcortical Networks in Tinnitus"**

The study investigates structural and functional connectivity in tinnitus by comparing
the cortical and subcortical organization of tinnitus patients and healthy controls,
combining structural MRI, diffusion MRI, and resting-state fMRI to characterize both
the brain's structural wiring and its functional networks.

The pipeline is BIDS-native and container-based (Apptainer), built around the
[QSIPrep](https://qsiprep.readthedocs.io)/[QSIRecon](https://qsirecon.readthedocs.io),
[fMRIPrep](https://fmriprep.org), [XCP-D](https://xcp-d.readthedocs.io), and
[MRIQC](https://mriqc.readthedocs.io) BIDS-Apps, with FreeSurfer providing the
structural segmentation that anchors both the diffusion (ACT/HSVS-based tractography)
and functional (surface-based) processing streams.

---

## Pipeline

Scripts are numbered in execution order and are idempotent, per-subject, and safe to
re-run: each maintains a status manifest under `derivatives/logs/<step>/` and skips
subjects that already have a completed output.

| Step | Script | Purpose | Status |
|------|--------|---------|--------|
| 00 | `00_convert_to_bids.py` | Convert raw Philips acquisitions into a valid BIDS dataset | Done |
| 01 | `01_check_freesurfer_status.py` | Read-only audit of FreeSurfer completeness per subject | Done |
| 02 | `02_run_freesurfer_pipeline.sh` | Fill in missing FreeSurfer segmentation steps (recon-all, subcortical/cortical parcellations) | Done |
| 03 | `03_prepare_bids_scaffold.py` | BIDS-validate the dataset; materialize FreeSurfer derivative symlinks; patch required sidecar fields | Done |
| 04 | `04_setup_containers.sh` | Install Apptainer and pull all pipeline container images | Done |
| 05 | `05_run_mriqc.sh` / `05b_mriqc_group_report.py` | Structural + functional image quality metrics, group-level QC report | Done — 4 subjects excluded on QC grounds ([excluded_subjects.tsv](../../volume/antinomics/derivatives/excluded_subjects.tsv)) |
| 06 | `06_run_qsiprep.sh` | DWI preprocessing: denoising, motion/eddy-current correction, bias correction | Done — 76/76 subjects |
| 07 | `07_run_qsirecon.sh` | CSD, ACT/HSVS-anchored probabilistic tractography (iFOD2, 10M streamlines), SIFT2 filtering, structural connectomes | Done — 76/76 subjects |
| 08 | `08_run_fmriprep.sh` | Anatomical + resting-state BOLD minimal preprocessing (motion correction, coregistration, spatial normalization, confound estimation) | In progress |
| 09 | `09_run_xcpd.sh` | Denoising and functional connectome construction (no-GSR primary, GSR sensitivity) | Planned |
| 10 | `10_auditory_qc_and_status.py` | Cross-pipeline status aggregation and auditory-ROI QC | Planned |

No fieldmaps are available in this dataset, so susceptibility distortion correction is
not performed at any stage — an explicit, documented limitation rather than a silent
gap (see `derivatives/README` in the BIDS root).

## Compute

Steps 00-06 and 08-10 run on a shared lab workstation (30 cores, 117GB RAM) under
Apptainer. Step 07 (QSIRecon) is CPU-tractography-bound at roughly 20-30 hours per
subject; the full-batch run was migrated to the [UZH ScienceCluster](https://www.zi.uzh.ch/en/teaching-and-research/science-it/computing/sciencecluster.html)
(Slurm) to parallelize across subjects rather than being limited to the workstation's
4-way concurrency budget.

## Repository layout

```
antinomics/
  00_convert_to_bids.py ... 10_auditory_qc_and_status.py   # pipeline scripts, execution order
  tinnitus_SFC_analysis_plan-4.md                          # full study analysis plan
```

Data and derivatives live outside this repository, on the workstation's data volume:

```
/home/ubuntu/volume/antinomics/            # BIDS root
  sub-*/                                    # raw BIDS data
  derivatives/
    freesurfer/                             # sub-<id> symlinks into the shared FreeSurfer store
    mriqc/, qsiprep/, qsirecon/, fmriprep/, xcp_d/
    logs/<step>/<step>_status.tsv           # per-step, per-subject status manifests
    excluded_subjects.tsv                   # QC-based exclusions, with rationale
```

---

*For questions or collaboration inquiries, please contact Payam.*
