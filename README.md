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

## Data layout (FTSW_DATA)

Notebooks resolve all simulation data through `src/ftsw_data.py`. Set one
environment variable per machine:

```bash
# Windows (PowerShell)
$env:FTSW_DATA = "E:\Xiao Lab Dropbox\Martin Yepes\FtsW_MD_dryrun"
# macOS/Linux
export FTSW_DATA=/path/to/FtsW_MD
```

The data root is a manifest-driven tree:

```text
$FTSW_DATA/
  manifest.yaml            # sim ids -> folder, variants, top/traj, time_factor
  sims.csv                 # optional drop-in export (used if manifest missing)
  sims/<id>_<name>/
    proc/<variant>/        # topology + trajectory kept together
    analysis/              # per-sim cached analysis CSVs live here
  external/                # sims tracked elsewhere (e.g. Anton archive)
  outputs/
    dynamics/<observable>/           # cross-sim outputs + provenance.json
    dynamics/<observable>/<stack>/   # stacked-histogram outputs
```

`ftsw_data.load_index()` builds the trajectory index (`sim_number`,
`sim_description`, `psf_path`, `dcd_path`, `time_factor`) from `manifest.yaml`
(or `sims.csv`), resolving relative paths to absolute at load time. No absolute
paths are stored in the repo.

Optional env vars:

- `MDFOLDER` — root of the old MDfolder tree; used to find pre-migration caches
  under `FtsW Manuscript/` (copied forward on first use, never modified) and
  external sims' index paths. Falls back to walking up to a dir named
  `MDfolder`.
- `FTSW_INDEX_CSV` — explicit `sims.csv` path; bypasses the manifest.

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

- read trajectories via `ftsw_data.load_index()` (manifest-driven),
- compute (or reuse cached) per-simulation CSVs like `{sim}_{short_var_name}.csv`
  in `sims/<id>_<name>/analysis/`,
- write cross-sim outputs to `outputs/dynamics/<observable>/` with a
  `provenance.json` (sims, params, git sha),
- and optionally generate per-simulation trace + histogram plots and summary tables.

### Stacked histograms (trace + histogram)

- Open `notebooks/Stack_Histograms_v2.ipynb`.
- Edit the **User inputs** cell: `STACK_LIST`, `SHORT_VAR_NAME`, `PLOT_TYPE`, `NUM_BINS`.
- Run all cells.

Per-sim CSVs are located automatically from `sims/<id>/analysis/` (plus the
observable's `outputs/dynamics/<var>/` dir and legacy caches). Outputs go to
`$FTSW_DATA/outputs/dynamics/<var>/<stack>/`.

The merged trace + sideways histogram figure is written as:

- `<STACK_LIST>_<short_var_name>_trace_plus_hist.png`

<img width="837" height="781" alt="image" src="https://github.com/user-attachments/assets/cc169350-93c3-4f9a-a865-7a5aab9807ba" />

## Repo hygiene

Generated outputs (e.g. `outputs/`), caches (e.g. `__pycache__/`), and notebook checkpoints are ignored via `.gitignore`.

## Trajectory index

The index is built by `ftsw_data.load_index()` from `$FTSW_DATA/manifest.yaml`
(or its `sims.csv` export). Columns:

- `sim_number` — string id (`14`, `14b`, `17ext`, `45_noW`, `62-gmx`, `A`–`F`)
- `sim_description`
- `psf_path` / `dcd_path` — resolved absolute paths (manifest stores them relative)
- `time_factor` — ns per frame
- `variant`, `folder`, `engine` — extra manifest fields (ignored by older code)

`traj_utils.validate_traj_index` accepts either this DataFrame or a legacy CSV
path, so older index files still work via `FTSW_INDEX_CSV`.
