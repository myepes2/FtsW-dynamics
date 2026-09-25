"""Resolve the FTSW_DATA simulation tree.

Single source of truth for data paths in FtsW-dynamics. Written to graduate
into the shared ``ftsw-data`` repo once it exists (see the ``ftsw-data`` skill
and MDfolder ``AGENTS.md``).

Environment variables
---------------------
FTSW_DATA
    Required. Data root containing ``manifest.yaml`` / ``sims.csv`` and the
    ``sims/``, ``external/``, ``outputs/`` tree. Set per machine.
MDFOLDER
    Optional. Root of the old MDfolder tree. Used to resolve the external
    sims' ``index_csv_pair`` provenance paths and to discover legacy analysis
    caches under ``FtsW Manuscript/``. Falls back to walking up from this
    repo to a directory named ``MDfolder``.
FTSW_INDEX_CSV
    Optional. Explicit path to a ``sims.csv`` export; bypasses the manifest.

Everything downstream consumes *resolved absolute paths*; nothing is stored
with a drive letter.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import yaml
except ImportError:  # manifest reading degrades to sims.csv
    yaml = None


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

def data_root() -> Path:
    """Return the FTSW_DATA root (required env var)."""
    raw = os.environ.get("FTSW_DATA", "").strip()
    if not raw:
        raise EnvironmentError(
            "FTSW_DATA is not set. Point it at the data tree root "
            "(the folder containing manifest.yaml / sims.csv)."
        )
    return Path(raw).expanduser().resolve()


def mdfolder_root() -> Optional[Path]:
    """Return the MDfolder root: $MDFOLDER, else walk up to a dir named 'MDfolder'."""
    raw = os.environ.get("MDFOLDER", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    d = Path(__file__).resolve()
    for parent in [d, *d.parents]:
        if parent.name == "MDfolder":
            return parent
    return None


# ---------------------------------------------------------------------------
# Manifest / index loading
# ---------------------------------------------------------------------------

def manifest_path(root: Optional[Path] = None) -> Path:
    return (root or data_root()) / "manifest.yaml"


def load_manifest(root: Optional[Path] = None) -> dict:
    """Load manifest.yaml. Raises if absent or pyyaml missing."""
    p = manifest_path(root)
    if yaml is None:
        raise ImportError("pyyaml is required to read manifest.yaml")
    if not p.is_file():
        raise FileNotFoundError(str(p))
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest.yaml at {p} did not parse to a mapping")
    return data


def _resolve(root: Path, p: str) -> str:
    """Resolve a possibly-relative path against root; absolute paths pass through."""
    p = str(p).strip()
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def _external_index_row(sim_id: str, entry: dict, md: Optional[Path]) -> Optional[dict]:
    """Build an index row for an external sim from its legacy index_csv_pair."""
    pair = entry.get("index_csv_pair") or {}
    top, traj = pair.get("top"), pair.get("traj")
    if not (top and traj):
        print(f"[ftsw_data] external sim '{sim_id}' has no index_csv_pair; skipped")
        return None
    if md is None:
        print(f"[ftsw_data] external sim '{sim_id}': MDfolder root unknown; skipped")
        return None
    return {
        "sim_number": sim_id,
        "sim_description": str(entry.get("description", "")),
        "psf_path": _resolve(md, top),
        "dcd_path": _resolve(md, traj),
        "time_factor": float(entry.get("time_factor", 1.0)),
        "variant": "",
        "folder": str(entry.get("folder", "")),
        "engine": str(entry.get("engine", "external")),
    }


def _index_from_manifest(manifest: dict, root: Path) -> pd.DataFrame:
    rows: List[dict] = []

    for raw_id, entry in (manifest.get("sims") or {}).items():
        sim_id = str(raw_id)
        if not isinstance(entry, dict):
            continue
        default = str(entry.get("default", "")).strip()
        variants = entry.get("variants") or {}
        var = variants.get(default) or {}
        top, traj = var.get("top"), var.get("traj")
        if not (top and traj):
            print(f"[ftsw_data] sim '{sim_id}': default variant '{default}' "
                  f"has no top/traj; skipped")
            continue
        time_factor = var.get("time_factor_ns", entry.get("time_factor", 1.0))
        rows.append({
            "sim_number": sim_id,
            "sim_description": str(entry.get("description", "")),
            "psf_path": _resolve(root, top),
            "dcd_path": _resolve(root, traj),
            "time_factor": float(time_factor),
            "variant": default,
            "folder": str(entry.get("folder", "")),
            "engine": str(entry.get("engine", "")),
        })

    md = mdfolder_root()
    for raw_id, entry in (manifest.get("external") or {}).items():
        row = _external_index_row(str(raw_id), entry or {}, md)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["sim_number"] = df["sim_number"].astype(str)
    return df


def _index_from_sims_csv(csv_path: Path, root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"sim_number": str})
    for col in ("psf_path", "dcd_path"):
        if col in df.columns:
            df[col] = df[col].map(lambda p: _resolve(root, p))
    if "folder" not in df.columns:
        df["folder"] = df["psf_path"].map(_folder_from_path)
    if "variant" not in df.columns:
        df["variant"] = df["psf_path"].map(_variant_from_path)
    if "engine" not in df.columns:
        df["engine"] = ""
    return df


def _folder_from_path(psf_path: str) -> str:
    """sims/<folder>/proc/<variant>/top -> 'sims/<folder>' when possible."""
    try:
        parts = Path(psf_path).parts
        if "proc" in parts:
            i = parts.index("proc")
            return str(Path(*parts[:i])) if i else ""
    except Exception:
        pass
    return ""


def _variant_from_path(psf_path: str) -> str:
    try:
        parts = Path(psf_path).parts
        if "proc" in parts:
            i = parts.index("proc")
            if i + 1 < len(parts):
                return parts[i + 1]
    except Exception:
        pass
    return ""


def load_index(root: Optional[Path] = None) -> pd.DataFrame:
    """Return the trajectory index as a DataFrame.

    Columns: ``sim_number, sim_description, psf_path, dcd_path, time_factor``
    plus ``variant, folder, engine`` (extra columns are ignored by older code).

    Source order: ``$FTSW_INDEX_CSV`` → ``manifest.yaml`` → ``sims.csv``.
    """
    root = Path(root).resolve() if root is not None else data_root()

    override = os.environ.get("FTSW_INDEX_CSV", "").strip()
    if override:
        return _index_from_sims_csv(Path(override).expanduser().resolve(), root)

    try:
        manifest = load_manifest(root)
    except (FileNotFoundError, ImportError) as e:
        print(f"[ftsw_data] manifest unavailable ({e}); falling back to sims.csv")
    else:
        df = _index_from_manifest(manifest, root)
        if not df.empty:
            return df

    csv_path = root / "sims.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Neither manifest.yaml nor sims.csv usable under {root}"
        )
    return _index_from_sims_csv(csv_path, root)


# ---------------------------------------------------------------------------
# Per-sim and cross-sim locations
# ---------------------------------------------------------------------------

def sim_folder(sim_id: str, root: Optional[Path] = None) -> Path:
    """Absolute path to a sim's folder (``sims/<id>_<name>/`` or external/).

    Resolution order: manifest ``folder`` field → glob ``sims/<id>_*/`` →
    ``external/<id>_*/``. Raises if nothing matches.
    """
    root = Path(root).resolve() if root is not None else data_root()
    sim_id = str(sim_id)

    try:
        manifest = load_manifest(root)
    except (FileNotFoundError, ImportError):
        manifest = {}

    for section in ("sims", "external"):
        entry = (manifest.get(section) or {}).get(sim_id)
        if isinstance(entry, dict) and entry.get("folder"):
            return (root / str(entry["folder"])).resolve()

    for base in (root / "sims", root / "external"):
        matches = sorted(base.glob(f"{sim_id}_*"))
        if matches:
            return matches[0].resolve()
        direct = base / sim_id
        if direct.is_dir():
            return direct.resolve()

    raise FileNotFoundError(
        f"sim '{sim_id}' has no folder under {root} (manifest + sims/ glob)"
    )


def analysis_dir(sim_id: str, root: Optional[Path] = None, *, create: bool = True) -> Path:
    """``sims/<id>_<name>/analysis/`` — per-sim cached analysis CSVs/PNGs."""
    d = sim_folder(sim_id, root) / "analysis"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def sim_analysis_dirs(
    sim_ids: Iterable[str],
    root: Optional[Path] = None,
    *,
    strict: bool = False,
) -> Dict[str, str]:
    """``{sim_id: analysis_dir}`` for configs' ``sim_analysis_dirs`` field.

    Sims with no folder in the data tree (e.g. synthetic ids like ``17full``)
    are skipped unless ``strict`` — their caches fall back to ``out_dir``.
    """
    out: Dict[str, str] = {}
    for s in sim_ids:
        try:
            out[str(s)] = str(analysis_dir(s, root, create=False))
        except FileNotFoundError:
            if strict:
                raise
    return out


def dynamics_dir(
    observable: str,
    stack: Optional[str] = None,
    root: Optional[Path] = None,
    *,
    create: bool = True,
) -> Path:
    """``outputs/dynamics/<observable>/[stack]/`` — cross-sim outputs."""
    root = Path(root).resolve() if root is not None else data_root()
    d = root / "outputs" / "dynamics" / str(observable)
    if stack:
        d = d / str(stack)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Legacy caches (old layout: loose per-variable dirs under FtsW Manuscript/)
# ---------------------------------------------------------------------------

def legacy_cache_dirs(max_depth: int = 3) -> List[Path]:
    """Candidate dirs holding pre-migration cached CSVs.

    ``<MDfolder>/FtsW Manuscript`` plus subdirectories up to ``max_depth``
    levels deep — old caches live at
    ``FtsW Manuscript/Analysis_Visualization/<var_dir>/`` (depth 2) and some
    nested deeper (e.g. ``acetyl_contacts/GlcNAc/``, depth 3).
    """
    md = mdfolder_root()
    if md is None:
        return []
    root = md / "FtsW Manuscript"
    if not root.is_dir():
        return []
    dirs = [root]
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            children = [p for p in d.iterdir() if p.is_dir()]
        except OSError:
            continue
        dirs.extend(children)
        stack.extend((p, depth + 1) for p in children)
    return sorted(dirs)


def find_cached_csv(
    short_var_name: str,
    sim_id: str,
    *,
    root: Optional[Path] = None,
    legacy_dirs: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    """Locate a cached per-sim CSV: new analysis dir first, then legacy dirs.

    Legacy matching: exact ``{sim}_{var}.csv`` filename, plus — when the dir
    itself is named after the observable — ``{sim}_*.csv`` (legacy dirs like
    ``FtsW_L198_L236_Ca_dist/`` hold files named ``8_FtsW_L198-L236.csv``).
    """
    sim_id = str(sim_id)
    name = f"{sim_id}_{short_var_name}.csv"
    try:
        p = analysis_dir(sim_id, root, create=False) / name
        if p.is_file():
            return p
    except FileNotFoundError:
        pass
    for d in (legacy_dirs if legacy_dirs is not None else legacy_cache_dirs()):
        d = Path(d)
        p = d / name
        if p.is_file():
            return p
        if d.name == str(short_var_name):
            hits = sorted(
                h for h in d.glob(f"{sim_id}_*.csv")
                if not h.stem.endswith("_byres")
            )
            if hits:
                return hits[0]
    return None


def analysis_csv_map(
    short_var_name: str,
    sim_ids: Iterable[str],
    *,
    root: Optional[Path] = None,
    include_legacy: bool = True,
) -> Dict[str, str]:
    """``{sim_id: csv_path}`` for stacked-histogram inputs across per-sim dirs."""
    out: Dict[str, str] = {}
    legacy = legacy_cache_dirs() if include_legacy else []
    for sim_id in sim_ids:
        p = find_cached_csv(short_var_name, str(sim_id), root=root, legacy_dirs=legacy)
        if p is not None:
            out[str(sim_id)] = str(p)
    return out


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def repo_root() -> Optional[Path]:
    """Git repo root containing this module (walk up looking for .git)."""
    d = Path(__file__).resolve()
    for parent in [d.parent, *d.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def git_sha() -> str:
    """Current commit of the FtsW-dynamics repo, or 'unknown'."""
    root = repo_root()
    if root is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_file = root / ".git" / ref
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip()
        elif head:
            return head
    except OSError:
        pass
    return "unknown"


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_provenance(
    out_dir: str | Path,
    *,
    sims: Iterable[str],
    params: dict,
    extra: Optional[dict] = None,
) -> Path:
    """Write ``provenance.json`` into ``out_dir`` (sims, params, git sha).

    Written atomically-ish (temp file + rename) to be kind to Dropbox sync.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        root = str(data_root())
    except EnvironmentError:
        root = ""

    manifest_info = ""
    try:
        mpath = manifest_path()
        if mpath.is_file() and yaml is not None:
            manifest_info = str(mpath)
    except Exception:
        pass

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "repo": str(repo_root() or ""),
        "ftsw_data": root,
        "manifest": manifest_info,
        "sims": [str(s) for s in sims],
        "params": _jsonable(params),
    }
    if extra:
        payload.update(_jsonable(extra))

    target = out_dir / "provenance.json"
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp", prefix="provenance_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target
