from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ContactSummaryStyle:
    scale: float = 4.0
    long_end: bool = True
    below_cutoff: bool = True
    fig_w: float = 12.0
    row_height: float = 0.6
    extra_height: float = 2.0
    bar_alpha: float = 0.7
    text_offset_frac: float = 0.01


def time_weighted_fraction(times: np.ndarray, mask: np.ndarray) -> float:
    times = np.asarray(times)
    mask = np.asarray(mask, dtype=bool)
    if times.size < 2:
        return float(mask.mean())

    dt = np.diff(times)
    in_contact = mask[:-1]
    contact_time = np.sum(dt[in_contact])
    total_time = times[-1] - times[0]
    return float(contact_time / total_time) if total_time > 0 else float(mask.mean())


def mask_to_segments_exact(times: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    times = np.asarray(times)
    mask = np.asarray(mask, dtype=bool)
    if times.size < 2:
        return []

    in_contact = mask[:-1]
    t0 = times[:-1]
    t1 = times[1:]

    segments: List[Tuple[float, float]] = []
    i = 0
    n = in_contact.size
    while i < n:
        if not in_contact[i]:
            i += 1
            continue
        start = float(t0[i])
        j = i + 1
        while j < n and in_contact[j]:
            j += 1
        end = float(t1[j - 1])
        segments.append((start, end - start))
        i = j

    return segments


def plot_contact_summary(
    *,
    data_dict: Dict[str, str],
    out_png_path: str,
    var_name: str,
    x_cutoff: float,
    summary_list: List[str],
    sim_list: List[str],
    label_list: List[str],
    num_to_label: Optional[Dict[str, str]] = None,
    style: Optional[ContactSummaryStyle] = None,
) -> str:
    if style is None:
        style = ContactSummaryStyle()

    end_times = {}
    for sim_number in summary_list:
        if sim_number not in data_dict:
            print(f"Skipping {sim_number}: not in data_dict")
            continue
        csv_path = data_dict[sim_number]
        try:
            df_tmp = pd.read_csv(csv_path)
            end_times[sim_number] = float(df_tmp["Time"].max())
        except Exception as e:
            print(f"Skipping {sim_number}: failed to read {csv_path}: {e}")

    if len(end_times) == 0:
        raise RuntimeError("No valid CSVs found in data_dict.")

    if style.long_end:
        common_end = max(end_times.values())
        print(f"Common end time (longest sim): {common_end:.2f} ns")
    else:
        common_end = min(end_times.values())
        print(f"Common end time (shortest sim): {common_end:.2f} ns")

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

    with mpl.rc_context(rc=rc_params):
        fig, ax = plt.subplots(figsize=(style.fig_w, fig_h))

        yticks = []
        ylabels = []

        for i, sim_number in enumerate(ordered_sims):
            csv_path = data_dict[sim_number]
            try:
                df_sim = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Failed to read {csv_path} for {sim_number}: {e}")
                continue

            mask_time = df_sim["Time"] <= common_end
            times = df_sim.loc[mask_time, "Time"].values
            xvals = df_sim.loc[mask_time, "X"].values

            if times.size == 0:
                print(f"No data within common window for {sim_number}, skipping.")
                continue

            if style.below_cutoff:
                contact_mask = xvals < x_cutoff
                plot_title = f"{var_name} Contact (X < {x_cutoff} Å)"
            else:
                contact_mask = xvals >= x_cutoff
                plot_title = f"{var_name} Non-Contact (X ≥ {x_cutoff} Å)"

            segments = mask_to_segments_exact(times, contact_mask)
            ax.broken_barh(
                segments,
                (i - 0.35, 0.7),
                facecolors="k",
                edgecolors="none",
                alpha=style.bar_alpha,
            )

            contact_frac = time_weighted_fraction(times, contact_mask)

            if style.long_end:
                ax.scatter([times[-1]], [i], marker="o", s=20, c="k")
                ax.scatter([times[-1]], [i], marker="|", s=120, c="k")

            x_text = common_end + style.text_offset_frac * common_end
            ax.text(x_text, i, f"{contact_frac:.2f}", va="center", ha="left", color="black")

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
        ax.set_title(plot_title)
        ax.set_xlim(0, common_end)

        plt.tight_layout()
        fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined contact summary to {out_png_path}")
        plt.show()

    return out_png_path
