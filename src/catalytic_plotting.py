from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def running_mean_rolling(x: np.ndarray, window_frames: int = 42) -> np.ndarray:
    import pandas as pd

    x = np.asarray(x, float)
    w = int(max(1, window_frames))
    if w == 1:
        return x
    return (
        pd.Series(x).rolling(window=w, center=True, min_periods=1).mean().to_numpy()
    )


@dataclass
class CatalyticFigureStyle:
    scale: float = 3.2

    # smoothing
    window_ns: float = 10.0

    # figure geometry
    figsize: tuple = (9.75, 5.25)
    width_ratios: tuple = (2.75, 1.25)
    dpi: int = 300

    # y-axis
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    # histogram
    num_bins: int = 30
    hist_alpha: float = 0.5
    hist_lw: float = 2.0
    prob_xmax: Optional[float] = None

    # traces
    trace_alpha: float = 0.5
    smooth_lw: float = 2.2

    # histogram axis formatting
    hist_xticks: tuple = (0.1, 0.2, 0.3)
    hist_xlim: Optional[tuple] = None

    # divider
    divider_lw: float = 3.0

    legend_mode: str = "above"  # 'above' or 'inside'
    legend_ncol: int = 1
    legend_framealpha: float = 0.85
    legend_facecolor: str = "white"


DEFAULT_COLORS = {
    "Acc-Don": "gray",
    "Acc-D297": "#00838F",
    "Don-D297": "green",
}

DEFAULT_LABELS = {
    "Acc-Don": "Acceptor - Donor",
    "Acc-D297": "Acceptor - D297",
    "Don-D297": "Donor - D297",
}


def plot_catalytic_trace_plus_hist(
    *,
    time_ns: np.ndarray,
    distance_series: Dict[str, np.ndarray],
    out_path: str,
    style: Optional[CatalyticFigureStyle] = None,
    colors_map: Optional[Dict[str, str]] = None,
    labels_map: Optional[Dict[str, str]] = None,
    title: str = "Catalytic Distances",
) -> str:
    if style is None:
        style = CatalyticFigureStyle()
    if colors_map is None:
        colors_map = dict(DEFAULT_COLORS)
    if labels_map is None:
        labels_map = dict(DEFAULT_LABELS)

    # derive dt (ns/frame) for smoothing window in frames
    if time_ns.size < 2:
        dt = 1.0
    else:
        dt = float(np.nanmedian(np.diff(time_ns)))
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0

    window_frames = int(round(style.window_ns / dt))

    # bins based on all data
    all_vals = np.concatenate([np.asarray(v) for v in distance_series.values()])
    plot_min = float(np.floor(all_vals.min() - 1.0))
    plot_max = float(np.ceil(all_vals.max() + 1.0))
    bins = np.linspace(plot_min, plot_max, style.num_bins + 1)

    y_min = style.y_min
    y_max = style.y_max
    if y_min is None or y_max is None:
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        pad = 0.05 * (vmax - vmin) if np.isfinite(vmax - vmin) and vmax > vmin else 1.0
        if y_min is None:
            y_min = vmin - pad
        if y_max is None:
            y_max = vmax + pad

    hist_xmax = style.prob_xmax
    if hist_xmax is None and style.hist_xlim is None:
        max_prob = 0.0
        for _k, x in distance_series.items():
            x = np.asarray(x)
            if x.size == 0:
                continue
            w = np.ones_like(x, dtype=float) / float(len(x))
            h, _ = np.histogram(x, bins=bins, weights=w)
            if h.size:
                max_prob = max(max_prob, float(np.nanmax(h)))
        hist_xmax = max(0.05, 1.1 * max_prob)

    rc_params = {
        "axes.titlesize": 10 * style.scale,
        "axes.labelsize": 8 * style.scale,
        "xtick.labelsize": 8 * style.scale,
        "ytick.labelsize": 8 * style.scale,
        "legend.fontsize": 5 * style.scale,
        "figure.titlesize": 10 * style.scale,
        "font.size": 8 * style.scale,
    }

    with mpl.rc_context(rc=rc_params):
        fig, (ax_t, ax_h) = plt.subplots(
            ncols=2,
            figsize=style.figsize,
            gridspec_kw={"width_ratios": style.width_ratios, "wspace": 0.0},
            sharey=True,
        )

        # time traces
        line_handles = []
        line_labels = []
        for key, x in distance_series.items():
            x = np.asarray(x)
            (ln,) = ax_t.plot(
                time_ns,
                x,
                color=colors_map.get(key, "gray"),
                alpha=style.trace_alpha,
                label=labels_map.get(key, key),
            )
            line_handles.append(ln)
            line_labels.append(labels_map.get(key, key))
            y_sm = running_mean_rolling(x, window_frames)
            ax_t.plot(
                time_ns,
                y_sm,
                color=colors_map.get(key, "gray"),
                alpha=1.0,
                lw=style.smooth_lw,
            )

        ax_t.set_title(title, pad=10)
        ax_t.set_xlabel("time (ns)")
        ax_t.set_ylabel("Distance (Å)")
        ax_t.set_ylim(y_min, y_max)

        if style.legend_mode == "inside":
            ax_t.legend(
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                borderaxespad=0.0,
                frameon=True,
                framealpha=style.legend_framealpha,
                facecolor=style.legend_facecolor,
                edgecolor="none",
                ncol=style.legend_ncol,
            )
        else:
            fig.legend(
                line_handles,
                line_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                frameon=True,
                framealpha=style.legend_framealpha,
                facecolor=style.legend_facecolor,
                edgecolor="none",
                ncol=style.legend_ncol,
            )
            fig.subplots_adjust(top=0.82)

        # sideways hist
        for key, x in distance_series.items():
            x = np.asarray(x)
            w = np.ones_like(x, dtype=float) / float(len(x))
            ax_h.hist(
                x,
                bins=bins,
                orientation="horizontal",
                weights=w,
                color=colors_map.get(key, "gray"),
                alpha=style.hist_alpha,
                edgecolor="black",
                linewidth=style.hist_lw,
            )

        ax_h.set_xlabel("Probability")
        if style.hist_xlim is not None:
            ax_h.set_xlim(*style.hist_xlim)
        else:
            ax_h.set_xlim(0, hist_xmax)

        ax_h.set_xticks(list(style.hist_xticks))
        ax_h.set_xticklabels([str(x) for x in style.hist_xticks])

        # divider styling
        ax_t.spines["right"].set_color("black")
        ax_t.spines["right"].set_linewidth(style.divider_lw)
        ax_h.spines["left"].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
        print(f"Wrote: {out_path}")
        plt.show()

    return out_path
