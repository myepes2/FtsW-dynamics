from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def read_curated_sim_list(path: str | Path, *, required: bool = True) -> List[str]:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(str(p))
        return []

    sims: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        sims.append(str(line))

    return sims


def write_curated_sim_list(path: str | Path, sims: List[str], *, overwrite: bool = False) -> str:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(str(s) for s in sims) + "\n", encoding="utf-8")
    return str(p)


def curated_list_with_optional_override(
    default_sims: List[str],
    override_path: str | Path,
    *,
    required: bool = False,
) -> List[str]:
    """Return `default_sims`, overridden by a file if it exists.

    This supports the "one preset in the notebook" workflow while still letting
    you persist a curated list as a plain-text file when desired.

    Parameters
    ----------
    default_sims
        The default curated list (order preserved).
    override_path
        If this file exists, it's read and returned.
    required
        If True, require the file to exist.
    """

    p = Path(override_path)
    if p.exists():
        return read_curated_sim_list(p, required=True)
    if required:
        raise FileNotFoundError(str(p))
    return list(default_sims)


def persist_per_variable_override_if_changed(
    *,
    default_sims: List[str],
    used_sims: List[str],
    curated_dir: str | Path,
    short_var_name: str,
    preset_name: str = "paper_main",
    overwrite: bool = True,
) -> Optional[str]:
    """Persist a per-variable curated list if it differs from the default.

    This supports:
    - a notebook-defined default list
    - optional overrides stored on disk *per variable* when needed

    The file name is:
        `{preset_name}_{short_var_name}.txt`

    Returns
    -------
    Optional[str]
        Path written, or None if no write was needed.
    """

    if list(used_sims) == list(default_sims):
        return None

    p = Path(curated_dir) / f"{preset_name}_{short_var_name}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)

    existing = None
    if p.exists():
        existing = read_curated_sim_list(p, required=True)
        if existing == list(used_sims):
            return None

    write_curated_sim_list(p, list(used_sims), overwrite=overwrite)
    return str(p)
