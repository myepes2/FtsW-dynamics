from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MDAnalysis.analysis import dihedrals

from distance_core import save_vars_to_file


def unwrap_angles(angle_series_deg: np.ndarray) -> np.ndarray:
    """Unwrap angles (degrees) to avoid periodic jumps."""
    unwrapped = np.unwrap(np.deg2rad(np.asarray(angle_series_deg, float)))
    return np.rad2deg(unwrapped)


def compute_single_residue_dihedral(
    u,
    *,
    resi: int,
    angle_type: str,
    ftsw_sel: str = "segid PROD",
    ftsw_truncation: int = 46,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute wrapped + unwrapped dihedral series for one residue.

    Parameters
    ----------
    u
        MDAnalysis Universe.
    resi
        Residue id in your canonical numbering (pre-truncation).
    angle_type
        'Chi1', 'Phi', or 'Psi'.
    ftsw_sel
        Selection string for the FtsW chain for that simulation.
    ftsw_truncation
        The v1 notebook used `start = resi - 46` to account for truncation.

    Returns
    -------
    wrapped_deg, unwrapped_deg
    """

    angle_type_norm = str(angle_type).strip().lower()

    ag = u.select_atoms(ftsw_sel)
    if len(ag.residues) == 0:
        raise ValueError(f"Selection has no residues: {ftsw_sel}")

    start = int(resi) - int(ftsw_truncation)
    idx = start
    if idx < 0 or idx >= len(ag.residues):
        raise ValueError(
            f"Residue index out of range after truncation: resi={resi}, trunc={ftsw_truncation}, "
            f"idx={idx}, n_res={len(ag.residues)}"
        )

    res = ag.residues[idx]

    if angle_type_norm == "chi1":
        sel = res.chi1_selection()
    elif angle_type_norm == "phi":
        sel = res.phi_selection()
    elif angle_type_norm == "psi":
        sel = res.psi_selection()
    else:
        raise ValueError(f"angle_type must be one of Chi1/Phi/Psi, got '{angle_type}'")

    if sel is None:
        raise ValueError(f"No dihedral selection available for {angle_type} at residue {res.resname}{res.resid}")

    dih = dihedrals.Dihedral([sel]).run()
    wrapped = np.asarray(dih.results.angles).T[0]
    unwrapped = unwrap_angles(wrapped)
    return wrapped, unwrapped


def write_dihedral_csv(
    *,
    time_factor: float,
    out_path: str | Path,
    var_name: str,
    wrapped: np.ndarray,
    unwrapped: np.ndarray,
) -> str:
    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    series = {
        f"{var_name} (degrees)": np.asarray(wrapped),
        f"Unwrapped {var_name} (degrees)": np.asarray(unwrapped),
    }
    return save_vars_to_file(float(time_factor), out_path, **series)


def build_17full_dihedral_csv(
    *,
    csv_17: str | Path,
    csv_17ext: str | Path,
    out_path: str | Path,
) -> str:
    """Concatenate 17 + 17ext into a continuous-time CSV (like v1)."""

    csv_17 = Path(csv_17)
    csv_17ext = Path(csv_17ext)
    out_path = Path(out_path)

    df1 = pd.read_csv(csv_17)
    df2 = pd.read_csv(csv_17ext)

    if "Time" not in df1.columns or "Time" not in df2.columns:
        raise ValueError("Both CSVs must include a 'Time' column")

    # shift df2 times so the combined series is continuous
    t1_max = float(df1["Time"].max())

    # estimate dt for segment 2 from its own time axis
    t2 = df2["Time"].to_numpy()
    if t2.size >= 2:
        dt2 = float(np.nanmedian(np.diff(t2)))
        if not np.isfinite(dt2) or dt2 <= 0:
            dt2 = 0.0
    else:
        dt2 = 0.0

    df2_shifted = df2.copy()
    df2_shifted["Time"] = (t1_max + dt2) + df2_shifted["Time"].to_numpy()

    df = pd.concat([df1, df2_shifted], ignore_index=True).sort_values("Time").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


@dataclass
class DihedralPlotStyle:
    scale: float = 2.6
    max_hist: float = 0.4
    x_lim: Optional[Tuple[float, float]] = (-420, 420)
    bin_step_deg: float = 10.0
    figsize: Tuple[float, float] = (12.8, 5.6)
    dpi: int = 300


def plot_dihedral_trace_hist(
    *,
    csv_path: str | Path,
    var_name: str,
    out_png_path: str | Path,
    title: Optional[str] = None,
    style: Optional[DihedralPlotStyle] = None,
) -> str:
    if style is None:
        style = DihedralPlotStyle()

    df = pd.read_csv(csv_path)
    if "Time" not in df.columns:
        raise ValueError("CSV must contain a 'Time' column")

    col = f"Unwrapped {var_name} (degrees)"
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in {csv_path}")

    t = df["Time"].to_numpy()
    x = df[col].to_numpy()

    if style.x_lim is None:
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
    else:
        x_min, x_max = style.x_lim

    bins = np.arange(x_min, x_max + style.bin_step_deg, style.bin_step_deg)

    rc_params = {
        "axes.titlesize": 12 * style.scale,
        "axes.labelsize": 8 * style.scale,
        "xtick.labelsize": 8 * style.scale,
        "ytick.labelsize": 8 * style.scale,
        "legend.fontsize": 5 * style.scale,
        "figure.titlesize": 12 * style.scale,
        "font.size": 8 * style.scale,
    }

    out_png_path = str(Path(out_png_path))
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context(rc=rc_params):
        fig, axs = plt.subplots(1, 2, figsize=style.figsize)

        if title:
            axs[0].set_title(title, pad=10)
            axs[1].set_title(title, pad=10)

        plot_label = f"{var_name} (°)\n(unwrapped)"

        axs[0].plot(t, x, color="gray")
        axs[0].set_ylabel(plot_label)
        axs[0].set_xlabel("time (ns)")
        axs[0].set_ylim(x_min, x_max)

        w = np.ones_like(x, dtype=float) / float(len(x)) if len(x) else np.array([])
        axs[1].hist(
            x,
            bins=bins,
            orientation="vertical",
            color="gray",
            alpha=0.7,
            weights=w,
            edgecolor="black",
            linewidth=2,
        )
        axs[1].set_xlabel(plot_label)
        axs[1].set_xlim(x_min, x_max)
        axs[1].set_ylim(0, style.max_hist)
        axs[1].set_yticks(np.arange(0, style.max_hist + 0.1, step=0.1))

        fig.tight_layout()
        fig.savefig(out_png_path, format="png", dpi=style.dpi, bbox_inches="tight")
        plt.show()

    return out_png_path


@dataclass
class AngleRangeSummaryStyle:
    scale: float = 4.0
    long_end: bool = True
    fig_w: float = 15.0
    row_height: float = 0.6
    extra_height: float = 2.0
    bar_alpha: float = 0.7
    text_offset_frac: float = 0.01


def plot_angle_range_summary(
    *,
    data_dict: Dict[str, str],
    out_png_path: str,
    summary_list: list[str],
    sim_list: list[str],
    label_list: list[str],
    var_name: str,
    angle_range: Tuple[float, float],
    value_column: Optional[str] = None,
    num_to_label: Optional[Dict[str, str]] = None,
    style: Optional[AngleRangeSummaryStyle] = None,
) -> str:
    import contact_summary as _cs

    if style is None:
        style = AngleRangeSummaryStyle()

    out_png_path = str(Path(out_png_path))
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)

    end_times = {}
    for sim_number in summary_list:
        if sim_number not in data_dict:
            continue
        df_tmp = pd.read_csv(data_dict[sim_number])
        end_times[sim_number] = float(df_tmp["Time"].max())

    if not end_times:
        raise RuntimeError("No valid CSVs found in data_dict")

    common_end = max(end_times.values()) if style.long_end else min(end_times.values())

    ordered_sims = [s for s in reversed(summary_list) if s in data_dict]
    n_sims = len(ordered_sims)
    fig_h = max(3, style.row_height * n_sims + style.extra_height)

    rc_params = {
        "axes.titlesize": 12 * style.scale,
        "axes.labelsize": 8 * style.scale,
        "xtick.labelsize": 8 * style.scale,
        "ytick.labelsize": 8 * style.scale,
        "legend.fontsize": 5 * style.scale,
        "figure.titlesize": 12 * style.scale,
        "font.size": 8 * style.scale,
    }

    lo, hi = float(angle_range[0]), float(angle_range[1])

    with mpl.rc_context(rc=rc_params):
        fig, ax = plt.subplots(figsize=(style.fig_w, fig_h))

        yticks = []
        ylabels = []

        for i, sim_number in enumerate(ordered_sims):
            df_sim = pd.read_csv(data_dict[sim_number])
            df_sim = df_sim[df_sim["Time"] <= common_end]

            if value_column is None:
                candidates = [
                    f"Unwrapped {var_name} (degrees)",
                    f"{var_name} (degrees)",
                ]
                col = next((c for c in candidates if c in df_sim.columns), None)
            else:
                col = value_column

            if col is None or col not in df_sim.columns:
                print(f"Skipping {sim_number}: value column not found")
                continue

            times = df_sim["Time"].to_numpy()
            xvals = df_sim[col].to_numpy()
            if times.size == 0:
                continue

            in_range = (xvals > lo) & (xvals < hi)
            segments = _cs.mask_to_segments_exact(times, in_range)
            ax.broken_barh(segments, (i - 0.35, 0.7), facecolors="k", edgecolors="none", alpha=style.bar_alpha)

            frac = _cs.time_weighted_fraction(times, in_range)
            x_text = common_end + style.text_offset_frac * common_end
            ax.text(x_text, i, f"{frac:.2f}", va="center", ha="left", color="black")

            yticks.append(i)
            if num_to_label and sim_number in num_to_label:
                ylabels.append(num_to_label[sim_number])
            else:
                try:
                    m = sim_list.index(sim_number)
                    ylabels.append(label_list[m])
                except ValueError:
                    ylabels.append(sim_number)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_ylim(-0.5, len(yticks) - 0.5)
        ax.set_xlabel("Time (ns)")
        ax.set_title(f"{var_name}  ({lo}° < angle < {hi}°)")
        ax.set_xlim(0, common_end)

        plt.tight_layout()
        fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
        plt.show()

    return out_png_path
