import numpy as np
import pandas as pd

from MDAnalysis.analysis.distances import distance_array


class SelectionEmptyError(ValueError):
    pass


def calc_res_distance(u, dist_type: str, sel1: str, sel2: str) -> np.ndarray:
    """Calculate a per-frame minimum distance between two selections.

    Parameters
    ----------
    u
        MDAnalysis Universe.
    dist_type
        - 'any': minimum atom-atom distance between grp1 and grp2
        - 'COG': compute distance between centers of geometry (still uses min distance for API compatibility)
        - 'COM': compute distance between centers of mass (still uses min distance for API compatibility)

    Notes
    -----
    This is a thin extraction of the behavior in `MYT_functions.calc_res_distance`.
    """

    grp1 = u.select_atoms(sel1)
    grp2 = u.select_atoms(sel2)

    if len(grp1.atoms) == 0 or len(grp2.atoms) == 0:
        raise SelectionEmptyError(
            "Empty selection(s) for distance calculation: "
            f"len(sel1)={len(grp1.atoms)} len(sel2)={len(grp2.atoms)}; "
            f"sel1='{sel1}' sel2='{sel2}'"
        )

    print(f"grp1: {len(grp1.atoms)} atoms\ngrp2: {len(grp2.atoms)} atoms")

    dist_list = []
    for _ts in u.trajectory:
        if dist_type == "COG":
            grp1 = u.select_atoms(sel1).center_of_geometry()
            grp2 = u.select_atoms(sel2).center_of_geometry()
        elif dist_type == "COM":
            grp1 = u.select_atoms(sel1).center_of_mass()
            grp2 = u.select_atoms(sel2).center_of_mass()

        dist = distance_array(grp1, grp2)
        min_dist = np.amin(dist)
        dist_list.append(min_dist)

    return np.array(dist_list)


def save_var_to_file(x: np.ndarray, time_factor: float, out_path: str, var_label: str = "X") -> str:
    t = np.arange(0, len(x)) * time_factor

    df_x = pd.DataFrame({"Time": t, var_label: x})
    df_x.to_csv(out_path, index=False)
    print(f"Wrote to {out_path}")
    return out_path


def save_vars_to_file(time_factor: float, out_path: str, **series) -> str:
    first_key = next(iter(series))
    n = len(series[first_key])

    t = np.arange(n) * time_factor

    data = {"Time": t}
    for name, arr in series.items():
        if len(arr) != n:
            raise ValueError(f"Length mismatch: '{name}' has length {len(arr)}, expected {n}.")
        data[name] = arr

    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    print(f"Wrote to {out_path}")
    return out_path
