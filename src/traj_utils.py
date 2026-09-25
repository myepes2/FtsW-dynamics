import os
import warnings
from typing import List, Tuple

import pandas as pd
import MDAnalysis as mda


def validate_index_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return only index rows with existing PSF/PDB and trajectory files.

    Expected columns (minimum):
    - sim_number
    - sim_description
    - psf_path
    - dcd_path
    - time_factor
    """

    required_columns = {"sim_number", "sim_description", "psf_path", "dcd_path"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Index must contain the following columns: {required_columns}")

    valid_entries = []
    for _, row in df.iterrows():
        psf_exists = os.path.exists(row["psf_path"])
        dcd_exists = os.path.exists(row["dcd_path"])

        if psf_exists and dcd_exists:
            valid_entries.append(row)
        else:
            if not psf_exists:
                print(f"Missing system file (PSF): {row['psf_path']}")
            if not dcd_exists:
                print(f"Missing trajectory file (DCD): {row['dcd_path']}")

    valid_df = pd.DataFrame(valid_entries)

    if valid_df.empty:
        raise RuntimeError("No valid entries found. All files are missing.")

    return valid_df


def validate_traj_index(index) -> pd.DataFrame:
    """Validate an index source: a DataFrame from ``ftsw_data.load_index()``,
    or a path to a legacy index CSV."""

    if isinstance(index, pd.DataFrame):
        return validate_index_df(index)

    if index is None or str(index).strip() == "":
        raise ValueError("index must be provided (GUI selection is disabled).")

    df = pd.read_csv(index, dtype={"sim_number": str})
    print(index)
    return validate_index_df(df)


def read_trajectory(df: pd.DataFrame, sim_number: str, in_memory: bool = True):
    row = df[df["sim_number"].astype(str) == str(sim_number)]
    if row.empty:
        raise ValueError(f"Simulation '{sim_number}' not found in index dataframe")

    psf_path = row["psf_path"].values[0]
    dcd_path = row["dcd_path"].values[0]

    warnings.filterwarnings(
        "ignore",
        message=r"DCDReader currently makes independent timesteps.*",
        category=DeprecationWarning,
    )
    u = mda.Universe(psf_path, dcd_path, in_memory=in_memory)
    return u, row


def read_trajectories(
    df: pd.DataFrame,
    sims: List[str],
    in_memory: bool = False,
) -> Tuple[List[object], List[str], List[float]]:
    u_list = []
    label_list = []
    tf_list = []

    if in_memory:
        print("Reading into memory")

    for s in sims:
        print(f"reading: {s}")
        u, row = read_trajectory(df, s, in_memory=in_memory)
        u.sim_number = s
        u_list.append(u)

        label = row["sim_description"].values[0]
        label_list.append(label)

        if "time_factor" not in row.columns:
            raise ValueError("Index CSV must contain 'time_factor' column for v2 workflow")
        time_factor = float(row["time_factor"].values[0])
        tf_list.append(time_factor)

        num_frames = len(u.trajectory)
        sim_time = time_factor * num_frames
        print(f"{s}: {num_frames} frames, {sim_time} ns")

    return u_list, label_list, tf_list
