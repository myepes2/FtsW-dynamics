import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dist_time_trace_hist(
    csv_file,
    var_name,
    sim_name,
    x_cutoff=None,
    x_lim=None,
    max_hist=0.4,
    split_title=True,
    scale=None,
):
    csv_parentdir = os.path.dirname(csv_file)
    csv_basename = os.path.basename(csv_file)
    no_ext = os.path.splitext(csv_basename)[0]
    sim_number = no_ext.split("_")[0]

    png_out_path = os.path.join(csv_parentdir, f"{no_ext}.png")

    df_x = pd.read_csv(csv_file)
    t = df_x["Time"]
    x = df_x["X"]

    min_text = (
        f"{sim_number}: Min = {x.min():.1f} Å | Frame {x.argmin()+1}/{x.size} | Time {t[x.argmin()]} ns"
    )
    print(min_text)

    if x_lim is not None:
        x_min, x_max = x_lim
    else:
        x_min = x.min()
        x_max = x.max()

    plot_min = min(x_min, x.min())
    plot_max = max(x_max, x.max())

    if scale is not None:
        rc_params = {
            "axes.titlesize": 12 * scale,
            "axes.labelsize": 8 * scale,
            "xtick.labelsize": 8 * scale,
            "ytick.labelsize": 8 * scale,
            "legend.fontsize": 5 * scale,
            "figure.titlesize": 12 * scale,
            "font.size": 8 * scale,
        }
    else:
        rc_params = {}

    with mpl.rc_context(rc=rc_params):
        fig, axs = plt.subplots(1, 2, figsize=(2 * 6.4, 6.4))

        if split_title:
            axs[0].set_title(sim_name, pad=10)
            axs[1].set_title(sim_name, pad=10)
        else:
            fig.suptitle(sim_name, y=0.92)

        axs[0].plot(t, x, color="gray")
        axs[0].set_ylabel(f"{var_name} distance (Å)")
        axs[0].set_xlabel("time (ns)")
        axs[0].set_ylim(plot_min, plot_max)

        axs[1].hist(
            x,
            bins=np.linspace(plot_min, plot_max + 1, 30),
            orientation="vertical",
            color="gray",
            alpha=0.7,
            weights=(1 / len(x)) * np.ones_like(x),
            edgecolor="black",
            linewidth=2,
        )
        axs[1].set_xlabel(f"{var_name} distance (Å)")
        axs[1].set_xlim(plot_min, plot_max)
        axs[1].set_ylim(0, max_hist)
        axs[1].set_yticks(np.arange(0, max_hist + 0.1, step=0.1))

        if x_cutoff is not None:
            x_below = x < x_cutoff
            x_below_frac = x_below.mean()
            label_text = (
                f"{sim_number}: Mean = {x.mean():.1f} Å | Fraction < {x_cutoff:.1f} Å = {x_below_frac:.2%}"
            )
            print(f"{label_text}")
            axs[0].hlines(x_cutoff, t.min(), t.max(), color="k", linewidth=2)
            axs[1].vlines(x_cutoff, 0, max_hist, color="k", linewidth=2)
            fig.text(0.5, -0.02, label_text, ha="center")

        fig.tight_layout()
        fig.savefig(png_out_path, format="png", dpi=300, bbox_inches="tight")
        print(f"Wrote to {png_out_path}")
        plt.show()
