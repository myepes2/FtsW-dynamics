import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _fallback_color_for_sim(sim_number: str, *, stack_list: list[str]) -> str:
    """Deterministic fallback color when not specified in style CSV.

    Uses matplotlib's default color cycle, indexed by the sim's position in
    `stack_list`.
    """

    try:
        i = stack_list.index(sim_number)
    except ValueError:
        i = 0

    cycle = plt.rcParams.get("axes.prop_cycle", None)
    if cycle is None:
        return "C0"

    colors = [d.get("color") for d in cycle]
    colors = [c for c in colors if c]
    if not colors:
        return "C0"

    return colors[i % len(colors)]


def detect_repo_root(cwd: str | Path | None = None) -> Path:
    p = Path.cwd().resolve() if cwd is None else Path(cwd).resolve()
    if (p / "notebooks").is_dir() and (p / "src").is_dir():
        return p
    if (p.parent / "notebooks").is_dir() and (p.parent / "src").is_dir():
        return p.parent.resolve()
    env = os.getenv("MD_REPO_ROOT", "")
    if env.strip():
        return Path(env).expanduser().resolve()
    return p


def load_style_map(style_csv: str | Path) -> dict[str, dict[str, str]]:
    style_csv = Path(style_csv).expanduser().resolve()
    if not style_csv.exists():
        return {}
    df = pd.read_csv(style_csv, dtype=str).fillna("")
    if "sim_number" not in df.columns:
        return {}
    out: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        sim = str(r["sim_number"]).strip()
        if not sim:
            continue

        # Pandas will read backslashes literally from CSV. Our curated labels are
        # typically written with doubled backslashes so the CSV stays readable.
        # Matplotlib mathtext expects single backslashes (e.g. '\\mathrm').
        label_default = str(r.get("label_default", "")).strip().replace("\\\\", "\\")
        label_explicit = str(r.get("label_explicit", "")).strip().replace("\\\\", "\\")

        out[sim] = {
            "group": str(r.get("group", "")).strip(),
            "color": str(r.get("color", "")).strip(),
            "label_default": label_default,
            "label_explicit": label_explicit,
            "linestyle": str(r.get("linestyle", "")).strip(),
        }
    return out


def resolve_style(style_map: dict[str, dict[str, str]], sim_number: str, *, mixed_apo: bool) -> tuple[str | None, str, str | None]:
    sim_number = str(sim_number)
    s = style_map.get(sim_number, {})

    color = s.get("color", "").strip() or None

    if mixed_apo and s.get("group", "") == "apo":
        label = s.get("label_explicit", "").strip() or s.get("label_default", "").strip() or sim_number
    else:
        label = s.get("label_default", "").strip() or sim_number

    linestyle = s.get("linestyle", "").strip() or None

    return color, label, linestyle


def choose_input_dir_tk(title: str) -> Path:
    import tkinter as tk
    from tkinter.filedialog import askdirectory

    root = tk.Tk()
    root.withdraw()
    p = Path(askdirectory(title=title)).resolve()
    if not str(p):
        raise RuntimeError("No directory selected")
    return p


def load_metadata_json(out_dir: str | Path) -> tuple[dict, Path | None]:
    out_dir = Path(out_dir).expanduser().resolve()
    json_files = sorted(out_dir.glob("*.json"))

    meta_file = None
    for p in json_files:
        if re.search(r"metadata", p.name, re.IGNORECASE):
            meta_file = p
            break
    if meta_file is None and json_files:
        meta_file = json_files[0]

    if meta_file is None or not meta_file.exists():
        return {}, None

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            return meta, meta_file
    except Exception:
        pass

    return {}, meta_file


def build_data_dict(out_dir: str | Path, meta: dict, short_var_name: str) -> dict[str, str]:
    out_dir = Path(out_dir).expanduser().resolve()
    data_dict: dict[str, str] = {}

    if isinstance(meta.get("data_dict", None), dict) and meta.get("data_dict"):
        for k, v in meta["data_dict"].items():
            if v:
                data_dict[str(k)] = str(Path(v))

    if not data_dict and short_var_name:
        inferred = sorted(out_dir.glob(f"*_{short_var_name}.csv"))
        for p in inferred:
            sim = p.name[: -len(f"_{short_var_name}.csv")]
            data_dict[sim] = str(p)

    if not data_dict:
        inferred = sorted(out_dir.glob("*.csv"))
        for p in inferred:
            data_dict[p.stem] = str(p)

    return data_dict


def load_series(data_dict: dict[str, str], stack_list: list[str], plot_type: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series_by_sim: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sim_number in stack_list:
        if sim_number not in data_dict:
            raise KeyError(f"Simulation '{sim_number}' not found in data_dict")

        df_x = pd.read_csv(data_dict[sim_number])
        if "Time" not in df_x.columns:
            raise ValueError(f"CSV for {sim_number} is missing a 'Time' column")

        if plot_type == "bsa":
            for col in ["BSA (Ų)", "BSA (Å²)", "BSA"]:
                if col in df_x.columns:
                    x = df_x[col]
                    break
            else:
                raise ValueError(f"CSV for {sim_number} missing BSA column")
        elif plot_type == "distance":
            x = df_x["X"] if "X" in df_x.columns else df_x.iloc[:, 1]
        elif plot_type == "angle":
            x = df_x.iloc[:, 1]
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        t = df_x["Time"].to_numpy()
        series_by_sim[sim_number] = (t, np.asarray(x))

    return series_by_sim


def compute_mixed_apo(style_map: dict[str, dict[str, str]], stack_list: list[str]) -> bool:
    groups = [style_map.get(s, {}).get("group", "") for s in stack_list]
    has_apo = any(g == "apo" for g in groups)
    has_non_apo = any((g != "apo") and (g != "") for g in groups)
    return bool(has_apo and has_non_apo)


def plot_histogram(
    *,
    series_by_sim: dict[str, tuple[np.ndarray, np.ndarray]],
    stack_list: list[str],
    style_map: dict[str, dict[str, str]],
    mixed_apo: bool,
    out_path: str | Path,
    title: str,
    xlabel: str,
    x_lim: tuple[float, float] | None,
    num_bins: int = 30,
    scale: float = 4.0,
    step: int = 60,
    plot_type: str,
) -> tuple[float, float]:
    plt.rcParams.update({
        "axes.titlesize": 10 * scale,
        "axes.labelsize": 9 * scale,
        "xtick.labelsize": 8 * scale,
        "ytick.labelsize": 8 * scale,
        "legend.fontsize": 6 * scale,
        "figure.titlesize": 10 * scale,
        "font.size": 8 * scale,
    })

    xs = [series_by_sim[s][1] for s in stack_list]
    x_min_guess = float(min(np.nanmin(x) for x in xs))
    x_max_guess = float(max(np.nanmax(x) for x in xs))

    if x_lim is None:
        plot_min, plot_max = x_min_guess, x_max_guess
    else:
        plot_min, plot_max = float(x_lim[0]), float(x_lim[1])

    plot_bins = np.linspace(plot_min, plot_max + 1, int(num_bins))

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.set_title(title, pad=30)
    axs.set_xlim(plot_min, plot_max)
    axs.set_xlabel(xlabel)
    axs.set_ylabel("Probability")

    count_max = 0.0

    for sim_number in stack_list:
        t, x = series_by_sim[sim_number]

        if plot_type == "angle":
            axs.set_xticks(np.arange(plot_min, plot_max + step, step=step))

        color, label, linestyle = resolve_style(style_map, sim_number, mixed_apo=mixed_apo)
        if color is None:
            color = _fallback_color_for_sim(sim_number, stack_list=stack_list)

        ls = linestyle if linestyle is not None else ("--" if "b" in sim_number else None)

        if "c" in sim_number:
            counts, _, _ = axs.hist(
                x,
                bins=plot_bins,
                orientation="vertical",
                color="white",
                alpha=0.4,
                weights=(1 / len(x)) * np.ones_like(x),
                edgecolor=color,
                linewidth=2,
                label=label,
            )
        else:
            hist_kwargs = {
                "bins": plot_bins,
                "orientation": "vertical",
                "color": color,
                "alpha": 0.4,
                "weights": (1 / len(x)) * np.ones_like(x),
                "edgecolor": "black",
                "linewidth": 2,
                "label": label,
            }
            if ls is not None:
                hist_kwargs["linestyle"] = ls

            counts, _, _ = axs.hist(x, **hist_kwargs)

        if len(counts):
            count_max = max(count_max, float(np.max(counts)))

    axs.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
    )

    legend = axs.get_legend()
    fig.canvas.draw()
    bbox_display = legend.get_window_extent(fig.canvas.get_renderer())
    bbox_data = bbox_display.transformed(axs.transData.inverted())
    legend_height = bbox_data.height

    max_hist = (count_max + legend_height) * 1.2
    axs.set_ylim(0, max_hist)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.show()

    return plot_min, plot_max


def running_mean_rolling(x: np.ndarray, window_frames: int) -> np.ndarray:
    x = np.asarray(x, float)
    w = int(max(1, window_frames))
    if w == 1:
        return x
    return (
        pd.Series(x)
        .rolling(window=w, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def plot_traces(
    *,
    series_by_sim: dict[str, tuple[np.ndarray, np.ndarray]],
    stack_list: list[str],
    style_map: dict[str, dict[str, str]],
    mixed_apo: bool,
    out_path: str | Path,
    title: str,
    ylabel: str,
    y_lim: tuple[float, float],
    scale: float = 4.0,
    smooth_window_frames: int = 10,
) -> None:
    fig, axs = plt.subplots(1, 1, figsize=((scale / 2) * 6, (scale / 2) * 4))
    axs.set_title(title, pad=scale * 10)
    axs.set_ylim(float(y_lim[0]), float(y_lim[1]))
    axs.set_ylabel(ylabel)
    axs.set_xlabel("time (ns)")

    for sim_number in stack_list:
        t, x = series_by_sim[sim_number]
        color, label, linestyle = resolve_style(style_map, sim_number, mixed_apo=mixed_apo)
        if color is None:
            color = _fallback_color_for_sim(sim_number, stack_list=stack_list)

        axs.plot(t, x, color=color, alpha=0.5)

        y = running_mean_rolling(x, window_frames=smooth_window_frames)
        ls = linestyle if linestyle is not None else ("--" if "b" in sim_number else "-")
        axs.plot(t, y, color=color, alpha=1.0, lw=2.2, ls=ls, label=label)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.show()


def write_plot_metadata(out_path: str | Path, meta: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
