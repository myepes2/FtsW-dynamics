# nascentPG_charmm agent notes

CHARMM input scripts + setup helpers for building the nascent-PG (NPG)
lipid/divisome models: `build_NPG.inp`, `pg_vars.str`, `divisome_vars.str`,
`setup_NPG.py`, `setup_assembly.py`, `setup_sim.py`, `Example/`.

## Repository boundary

This folder lives **inside the FtsW-dynamics repo**
(`C:\Users\12404\Documents\GitHub\FtsW-dynamics`). Commits here share that
repo's history and remote (`myepes2/FtsW-dynamics`, branch `main`) — pull
--rebase before starting, keep NPG commits logically separate from
analysis-pipeline commits, and coordinate with the FtsW-dynamics agent on
shared files (`src/`, `environment.yml`).

## Related material outside this repo

- `Code_Resources\` root (Dropbox, read-only snapshots + loose files):
  `gen_NPG_charmm_inp.ipynb`, `convert_rtf_to_viparr.ipynb`,
  `LIG_NPG_conversion.pml`, `pg_subset.prm`, `toppar_files.csv`
- `Cell_Division_Projects\Residue_Build_Test\` and
  `NascentPG_Divisome_Build\` — the data-side build sandboxes
- `MDfolder\Parameter Generation\`

If you consolidate these, copy (don't move) and note it — the Dropbox
copies are shared lab state.

## Interface with the data tree

- Parameter/topology outputs for a specific sim belong in
  `${FTSW_DATA}\sims\<id>_<name>\params\`; force-field pieces shared across
  sims belong in `${FTSW_DATA}\refs\`. Invoke the `ftsw-data` skill before
  touching data paths — no drive letters, ever.
- CHARMM `.inp`/`.str` files are run on a cluster/HPC: keep them portable,
  parameterise paths, don't embed local machine locations.
- Do not edit `manifest.*` — new sims/variants are registered by the
  data-tree coordinator.
