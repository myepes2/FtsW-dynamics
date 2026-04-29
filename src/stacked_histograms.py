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

    # Prefer config JSONs if present (single-source-of-truth style):
    # e.g. '{short_var_name}_config.json'
    for p in json_files:
        if re.search(r"_config\.json$", p.name, re.IGNORECASE):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    return meta, p
            except Exception:
                pass

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

    # Augment with any CSVs present in the out_dir. This is important because
    # older metadata JSONs can have a partial/outdated data_dict.
    inferred: list[Path] = []
    if short_var_name:
        inferred.extend(sorted(out_dir.glob(f"*_{short_var_name}.csv")))
    inferred.extend(sorted(out_dir.glob("*.csv")))

    for p in inferred:
        # Prefer the sim_number prefix (before the first underscore) so both
        # '53_Foo.csv' and '53_Foo_Bar.csv' map to '53'.
        prefix = p.name.split("_", 1)[0]
        if prefix and prefix not in data_dict:
            data_dict[prefix] = str(p)

        # Also keep a fallback key for unusual filenames if it doesn't collide.
        stem = p.stem
        if stem and stem not in data_dict:
            data_dict[stem] = str(p)

    return data_dict


def infer_axis_label(*, meta: dict, var_name: str, unit_tag: str) -> str:
    res1 = str(meta.get("res1", ""))
    res2 = str(meta.get("res2", ""))

    is_ca = bool(re.search(r"\bname\s+CA\b", res1)) and bool(re.search(r"\bname\s+CA\b", res2))
    base = str(var_name).strip() or "X"

    plot_tag = str(meta.get("plot_tag", "")).strip()
    if not plot_tag and is_ca:
        plot_tag = "Cα - Cα"

    unit_tag = str(unit_tag).strip()
    if not unit_tag:
        return f"{base} {plot_tag}".strip() if plot_tag else base

    # Insert the tag between var_name and the quantity (e.g. 'distance') so the
    # tag can be swapped independently (e.g. Cα - Cα vs. H-bond).
    if plot_tag:
        label = f"{base} {plot_tag} {unit_tag}".strip()
    else:
        label = f"{base} {unit_tag}".strip()
    if len(label) > 28:
        if plot_tag:
            return f"{base}\n{plot_tag} {unit_tag}".strip()
        return f"{base}\n{unit_tag}".strip()
    return label


def load_series(data_dict: dict[str, str], stack_list: list[str], plot_type: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series_by_sim: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sim_number in stack_list:
        sim_number = str(sim_number)
        if sim_number not in data_dict:
            keys_preview = ", ".join(list(data_dict.keys())[:25])
            raise KeyError(
                f"Simulation '{sim_number}' not found in data_dict. "
                f"Available keys (first 25): [{keys_preview}]"
            )

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

    fig.tight_layout(pad=0.9, w_pad=0.8, h_pad=0.8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.show()

    return plot_min, plot_max


def plot_traces_plus_side_hist(
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
    num_bins: int = 30,
    width_ratios: tuple[float, float] = (2.75, 1.25),
    trace_alpha: float = 0.5,
    hist_alpha: float = 0.5,
    divider_lw: float = 3.0,
    legend_ncol: int = 1,
    y_pad_frac: float | tuple[float, float] = 0.08,
) -> None:
    # The standalone histogram/traces plots in this module intentionally use
    # large text (scaled for manuscript figures). For the merged trace+hist
    # layout, those same sizes become unreadable due to crowding/clipping.
    # Use a gentler font scaling here.
    plt.rcParams.update({
        "axes.titlesize": 6 * scale,
        "axes.labelsize": 5 * scale,
        "xtick.labelsize": 5 * scale,
        "ytick.labelsize": 5 * scale,
        "legend.fontsize": 4 * scale,
        "figure.titlesize": 6 * scale,
        "font.size": 5 * scale,
    })

    # If the caller passes a full axis label like:
    #   'FtsW L198 - L236\nCα - Cα distance (Å)'
    # it is too large when rotated vertically. Keep the *units* on the y-axis
    # and rely on the title/legend for context.
    if isinstance(ylabel, str) and "\n" in ylabel:
        ylabel = ylabel.split("\n")[-1].strip() or ylabel

    y_min, y_max = float(y_lim[0]), float(y_lim[1])
    if isinstance(y_pad_frac, (tuple, list)) and len(y_pad_frac) == 2:
        y_pad_low = float(y_pad_frac[0])
        y_pad_high = float(y_pad_frac[1])
    else:
        y_pad_low = float(y_pad_frac)
        y_pad_high = float(y_pad_frac)

    y_range = max(1e-12, y_max - y_min)
    y_lim_plot = (y_min - y_pad_low * y_range, y_max + y_pad_high * y_range)
    bins = np.linspace(y_min, y_max, int(num_bins) + 1)

    fig, (ax_t, ax_h) = plt.subplots(
        ncols=2,
        figsize=(9.0, 4.6),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.0},
        sharey=True,
    )

    # time traces
    line_handles = []
    line_labels = []
    for sim_number in stack_list:
        t, x = series_by_sim[sim_number]
        color, label, linestyle = resolve_style(style_map, sim_number, mixed_apo=mixed_apo)
        if color is None:
            color = _fallback_color_for_sim(sim_number, stack_list=stack_list)

        ax_t.plot(t, x, color=color, alpha=trace_alpha)

        y_sm = running_mean_rolling(x, window_frames=smooth_window_frames)
        ls = linestyle if linestyle is not None else ("--" if "b" in sim_number else "-")
        (ln,) = ax_t.plot(
            t,
            y_sm,
            color=color,
            alpha=1.0,
            lw=2.2,
            ls=ls,
            label=label,
        )
        line_handles.append(ln)
        line_labels.append(label)

    ax_t.set_title(title, pad=scale * 2.5)
    ax_t.set_xlabel("time (ns)", labelpad=scale * 0.8)
    ax_t.set_ylabel(ylabel, labelpad=scale * 0.8)
    ax_t.set_ylim(*y_lim_plot)

    # sideways histogram
    max_prob = 0.0
    for sim_number in stack_list:
        _t, x = series_by_sim[sim_number]
        x = np.asarray(x)
        if x.size == 0:
            continue

        w = np.ones_like(x, dtype=float) / float(len(x))
        h, _ = np.histogram(x, bins=bins, weights=w)
        if h.size:
            max_prob = max(max_prob, float(np.nanmax(h)))

        color, label, linestyle = resolve_style(style_map, sim_number, mixed_apo=mixed_apo)
        if color is None:
            color = _fallback_color_for_sim(sim_number, stack_list=stack_list)

        hist_kwargs = {
            "bins": bins,
            "orientation": "horizontal",
            "weights": w,
            "color": color,
            "alpha": hist_alpha,
            "edgecolor": "black",
            "linewidth": 2,
        }
        ls = linestyle if linestyle is not None else ("--" if "b" in sim_number else None)
        if ls is not None:
            hist_kwargs["linestyle"] = ls
        ax_h.hist(x, **hist_kwargs)

    ax_h.set_xlabel("Probability", labelpad=scale * 0.8)
    ax_h.set_xlim(0, max(0.05, 1.1 * max_prob))

    # legend on histogram panel (upper right)
    ax_h.legend(
        line_handles,
        line_labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
        ncol=legend_ncol,
    )

    # divider styling
    ax_t.spines["right"].set_color("black")
    ax_t.spines["right"].set_linewidth(divider_lw)
    ax_h.spines["left"].set_visible(False)

    fig.tight_layout(pad=0.9, w_pad=0.8, h_pad=0.8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.show()

    return None


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
