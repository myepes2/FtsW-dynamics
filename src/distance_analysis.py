import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from distance_core import SelectionEmptyError, calc_res_distance, save_var_to_file, save_vars_to_file


@dataclass
class PlotStyle:
    share_lim: bool = True
    x_min_shared: float = 0.0
    x_max_shared: float = 20.0
    max_hist: float = 0.4
    scale: float = 2.6
    var_name_plot: Optional[str] = None


@dataclass
class DistanceAnalysisConfig:
    """Configuration for one distance analysis (one 'variable') across many simulations.

    This replaces the old `set_manual` / ad-hoc metadata JSON workflow.

    Notes
    -----
    - Intermediate per-simulation CSVs are treated as a cache.
    - Plotting should always read from cached CSVs unless `force_recompute=True`.
    """

    # Identity / labeling
    var_name: str
    short_var_name: str

    # MDAnalysis selections and how to compute distance
    res1: str
    res2: str
    dist_type: str  # e.g., 'any' (min atom-atom), 'COG', 'COM'

    # Simulations to process
    plot_list: List[str]

    # Output
    out_dir: str

    # Analysis parameters
    x_cutoff: Optional[float] = None
    partition: bool = False

    # Plotting
    plot_style: PlotStyle = field(default_factory=PlotStyle)

    # Metadata
    version: int = 1

    def csv_path_for_sim(self, sim_number: str) -> str:
        if self.partition:
            out_csv = f"{sim_number}_{self.short_var_name}_byres.csv"
        else:
            out_csv = f"{sim_number}_{self.short_var_name}.csv"
        return os.path.join(self.out_dir, out_csv)

    def config_path(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"{self.short_var_name}_config.json"
        return os.path.join(self.out_dir, filename)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        # dataclasses -> nested dict already, but keep explicit in case of future changes
        return d

    def save(self, path: Optional[str] = None) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        if path is None:
            path = self.config_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "DistanceAnalysisConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        plot_style = PlotStyle(**d.get("plot_style", {}))
        d["plot_style"] = plot_style
        return DistanceAnalysisConfig(**d)


def build_data_dict_from_cache(cfg: DistanceAnalysisConfig) -> Dict[str, str]:
    """Return mapping sim_number -> csv_path for cached CSVs that exist."""
    data_dict: Dict[str, str] = {}
    for sim_number in cfg.plot_list:
        csv_path = cfg.csv_path_for_sim(sim_number)
        if os.path.exists(csv_path):
            data_dict[sim_number] = csv_path
    return data_dict


def compute_distance_csvs(
    cfg: DistanceAnalysisConfig,
    sim_list: List[str],
    u_list: List[object],
    tf_list: List[float],
    force_recompute: bool = False,
) -> Dict[str, str]:
    """Compute per-simulation distance CSVs if missing, otherwise reuse cached CSVs.

    Parameters
    ----------
    cfg
        Analysis configuration.
    sim_list
        Master list of simulation ids, aligned with u_list/tf_list.
    u_list
        List of MDAnalysis Universes.
    tf_list
        Time factor per simulation (ns per frame).
    force_recompute
        If True, overwrite CSVs even if they exist.

    Returns
    -------
    Dict[str, str]
        Mapping sim_number -> CSV path.
    """

    os.makedirs(cfg.out_dir, exist_ok=True)

    data_dict: Dict[str, str] = {}

    for sim_number in cfg.plot_list:
        out_path = cfg.csv_path_for_sim(sim_number)

        if os.path.exists(out_path) and not force_recompute:
            data_dict[sim_number] = out_path
            continue

        if sim_number not in sim_list:
            raise ValueError(f"Simulation '{sim_number}' not found in sim_list.")

        m = sim_list.index(sim_number)
        traj = u_list[m]
        time_factor = tf_list[m]

        if cfg.partition:
            # Kept for compatibility, but partitioning requires helper functions from MYT_functions.
            # We intentionally do not import them here to keep this module focused.
            raise NotImplementedError(
                "partition=True is not yet supported in distance_analysis.compute_distance_csvs. "
                "Keep using the old notebook for partitioning, or we can port that workflow next."
            )

        try:
            x = calc_res_distance(traj, cfg.dist_type, cfg.res1, cfg.res2)
        except SelectionEmptyError as e:
            print(f"Skipping sim {sim_number}: {e}")
            continue

        csv_path = save_var_to_file(x, time_factor, out_path)
        data_dict[sim_number] = csv_path

    return data_dict


def make_summary_table(data_dict: Dict[str, str], out_path: str, x_cutoff: Optional[float] = None) -> pd.DataFrame:
    """Same idea as the notebook helper: aggregate across all value columns in each CSV."""

    rows = []
    for sim_id, csv_file in data_dict.items():
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            rows.append(
                {
                    "sim_id": sim_id,
                    "csv_path": csv_file,
                    "n_points": 0,
                    "min": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "stderr": np.nan,
                    "max": np.nan,
                    "fraction_below_cutoff": np.nan,
                    "error": str(e),
                }
            )
            continue

        if "Time" in df.columns:
            value_cols = [c for c in df.columns if c != "Time"]
        else:
            value_cols = list(df.columns)

        if len(value_cols) == 0:
            rows.append(
                {
                    "sim_id": sim_id,
                    "csv_path": csv_file,
                    "n_points": 0,
                    "min": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "stderr": np.nan,
                    "max": np.nan,
                    "fraction_below_cutoff": np.nan,
                    "error": "no value columns",
                }
            )
            continue

        try:
            all_vals = np.concatenate([df[c].dropna().values for c in value_cols])
        except Exception as e:
            rows.append(
                {
                    "sim_id": sim_id,
                    "csv_path": csv_file,
                    "n_points": 0,
                    "min": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "stderr": np.nan,
                    "max": np.nan,
                    "fraction_below_cutoff": np.nan,
                    "error": f"concat_error: {e}",
                }
            )
            continue

        n = all_vals.size
        if n == 0:
            rows.append(
                {
                    "sim_id": sim_id,
                    "csv_path": csv_file,
                    "n_points": 0,
                    "min": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "stderr": np.nan,
                    "max": np.nan,
                    "fraction_below_cutoff": np.nan,
                    "error": "no numeric data",
                }
            )
            continue

        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        vmean = float(np.nanmean(all_vals))
        vstd = float(np.nanstd(all_vals, ddof=1)) if n > 1 else 0.0
        vstderr = float(vstd / np.sqrt(n)) if n > 0 else np.nan

        if x_cutoff is not None:
            frac_below = float((all_vals <= x_cutoff).mean())
        else:
            frac_below = np.nan

        rows.append(
            {
                "sim_id": sim_id,
                "csv_path": csv_file,
                "n_points": int(n),
                "min": vmin,
                "mean": vmean,
                "std": vstd,
                "stderr": vstderr,
                "max": vmax,
                "fraction_below_cutoff": frac_below,
                "error": "",
            }
        )

    df_summary = pd.DataFrame(rows)
    cols = [
        "sim_id",
        "csv_path",
        "n_points",
        "min",
        "mean",
        "std",
        "stderr",
        "max",
        "fraction_below_cutoff",
        "error",
    ]
    df_summary = df_summary[cols]

    try:
        df_summary.to_csv(out_path, index=False)
    except Exception:
        # keep silent: summary is convenient but not critical
        pass

    return df_summary


def load_or_create_config(
    *,
    out_dir: str,
    short_var_name: str,
    default: Optional[DistanceAnalysisConfig] = None,
    filename: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[DistanceAnalysisConfig, str]:
    """Load `{short_var_name}_config.json` from `out_dir`, or create it.

    This is meant to be used from notebooks where you want a single pattern:
    - define `default` once for a new analysis variable
    - re-run later and automatically load the existing config

    Parameters
    ----------
    out_dir
        Directory that also contains cached per-simulation CSVs.
    short_var_name
        Used to derive the default config filename.
    default
        If the config file does not exist, this object is saved and returned.
        If `default` is provided, its `out_dir` / `short_var_name` are forced to
        match the function arguments.
    filename
        Optional explicit config filename (within `out_dir`). If not provided,
        uses `{short_var_name}_config.json`.
    overwrite
        If True, write `default` even if a config file already exists.

    Returns
    -------
    (cfg, path)
        The loaded/created config and the JSON path.
    """

    if filename is None:
        filename = f"{short_var_name}_config.json"
    path = os.path.join(out_dir, filename)

    if os.path.exists(path) and not overwrite:
        try:
            return DistanceAnalysisConfig.load(path), path
        except Exception as e:
            if default is None:
                raise
            print(f"Warning: failed to load config '{path}' ({type(e).__name__}: {e}). Rewriting from default.")
            cfg = default
            cfg.out_dir = out_dir
            cfg.short_var_name = short_var_name
            cfg.save(path)
            return cfg, path

    if default is None:
        raise FileNotFoundError(
            f"Config not found at '{path}'. Provide `default=DistanceAnalysisConfig(...)` to create it."
        )

    cfg = default
    cfg.out_dir = out_dir
    cfg.short_var_name = short_var_name
    cfg.save(path)
    return cfg, path


def _safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def resolve_out_dir(
    *,
    out_root: str,
    short_var_name: str,
    out_dir_override: Optional[str] = None,
    prefer_existing: bool = True,
) -> str:
    """Resolve which output directory to use for a given analysis.

    This is a non-GUI alternative to "pick output folder" that still supports
    your legacy folder naming patterns where the folder name doesn't match
    `short_var_name`.

    Resolution order
    ----------------
    - If `out_dir_override` is provided: return it.
    - If `prefer_existing=True`: scan `out_root` and its immediate subfolders
      for evidence of an existing analysis for `short_var_name`:
        - a JSON file with `short_var_name` in its content (legacy `*_metadata.json`)
        - a JSON file named `{short_var_name}_config.json`
        - cached CSVs matching `*_{short_var_name}.csv`
      If found, return that directory.
    - Otherwise default to `{out_root}/{short_var_name}`.
    """

    out_root = os.path.abspath(os.path.expanduser(out_root))
    default_dir = os.path.join(out_root, short_var_name)

    if out_dir_override is not None and str(out_dir_override).strip() != "":
        return os.path.abspath(os.path.expanduser(str(out_dir_override)))

    if not prefer_existing or not os.path.isdir(out_root):
        return default_dir

    candidates = [out_root]
    try:
        for name in os.listdir(out_root):
            p = os.path.join(out_root, name)
            if os.path.isdir(p):
                candidates.append(p)
    except Exception:
        return default_dir

    # 1) Prefer explicit config filename match
    config_name = f"{short_var_name}_config.json"
    for d in candidates:
        p = os.path.join(d, config_name)
        if os.path.isfile(p):
            return d

    # 2) Look for legacy metadata json that declares short_var_name
    for d in candidates:
        try:
            for fn in os.listdir(d):
                if not fn.lower().endswith(".json"):
                    continue
                meta = _safe_load_json(os.path.join(d, fn))
                if not isinstance(meta, dict):
                    continue
                if str(meta.get("short_var_name", "")) == short_var_name:
                    return d
        except Exception:
            continue

    # 3) Finally, check for cached CSV naming pattern
    suffix = f"_{short_var_name}.csv"
    for d in candidates:
        try:
            if any(fn.endswith(suffix) for fn in os.listdir(d)):
                return d
        except Exception:
            continue

    return default_dir
