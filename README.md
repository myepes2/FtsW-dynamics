# FtsW-dynamics
Analysis Pipeline from Yepes Yehya 2026:
"Substrate binding and activation mechanism of the essential bacterial septal cell wall synthase FtsW"
https://www.biorxiv.org/content/10.64898/2026.02.13.705525v1.full

This repository contains analysis notebooks and reusable Python helpers for molecular dynamics (MD) simulations focused on **FtsW**, a bacterial membrane protein involved in septal peptidoglycan synthesis and cell division.

<img width="771" height="400" alt="image" src="https://github.com/user-attachments/assets/6f388cd1-bf81-40e5-bcc0-18109aa0e58f" />

Because we run **many simulations** (different constructs/conditions/replicates, often with different trajectory lengths and file locations), the analysis must be **automated and index-driven**. The goal is to make it easy to:

- calculate the same observable across many trajectories,
- cache per-simulation results (CSVs) to avoid recomputation,
- and regenerate example/paper figures in a repeatable way.

## Setup

This repo is primarily driven by Jupyter notebooks in `notebooks/` which import helper modules from `src/`.

- Create the conda environment:

  ```bash
  conda env create -f environment.yml
  conda activate ftsw-dynamics
  ```

## Running analyses

### 1) Residue distances (recommended starting point)

Most downstream plots (including stacked histograms) assume you already have per-simulation CSVs of the quantity of interest.

Start with:

- `notebooks/Compare_Residue_Distances_v2.ipynb`

Key idea / take-home message:

- You can compute **any distance** (or related geometric metric) as long as you can define the atom groups using **MDAnalysis selection strings**.

Example (outer gate distance, L198–L236 Cα–Cα):

```text
res1 = (segid PROD PAG1 PAU1) and resid 198 and name CA
res2 = (segid PROD PAG1 PAU1) and resid 236 and name CA
```

This notebook will:

- read trajectories using an **index CSV** (see below),
- compute (or reuse cached) per-simulation CSVs like `{sim}_{short_var_name}.csv`,
- and optionally generate per-simulation trace + histogram plots and summary tables.

### Stacked histograms (trace + histogram)

- Open `notebooks/Stack_Histograms_v2.ipynb`.
- Edit the **User inputs** cell (e.g. `STACK_LIST`, `PLOT_TYPE`, `NUM_BINS`).
- Run all cells.

This notebook expects a folder that already contains per-simulation CSVs (typically produced by the residue-distance workflow above), plus optional metadata.

Outputs are written under:

- `outputs/stacked_histograms/<variable>/...`

The merged trace + sideways histogram figure is written as:

- `<STACK_LIST>_<short_var_name>_trace_plus_hist.png`

<img width="837" height="781" alt="image" src="https://github.com/user-attachments/assets/cc169350-93c3-4f9a-a865-7a5aab9807ba" />

## Repo hygiene

Generated outputs (e.g. `outputs/`), caches (e.g. `__pycache__/`), and notebook checkpoints are ignored via `.gitignore`.

## Trajectory index CSV

Several notebooks use a single **trajectory index CSV** to describe where each simulation’s topology/trajectory files live and how to interpret time.

At minimum, the index CSV should contain columns:

- `sim_number`
- `sim_description`
- `psf_path` (or a topology path used by MDAnalysis)
- `dcd_path` (trajectory path)
- `time_factor` (to convert frames to time units)

This indirection is what lets the same analysis code run across many simulations without hardcoding file paths inside each notebook.
