from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from distance_core import calc_res_distance


@dataclass
class CatalyticSelections:
    ftsw_297: str
    l2_acceptor: str
    l2_donor: str


def _first_nonempty_selection(u, candidates: Iterable[str]) -> Optional[str]:
    for sel in candidates:
        if len(u.select_atoms(sel)) > 0:
            return sel
    return None


def infer_l2_donor_selection(u, *, segid: str = "HETB") -> str:
    """Infer Lipid II donor atom name for a given Universe.

    Some sims use different donor atom names (e.g. C56, C58, C138).
    We avoid hardcoding per-sim rules by selecting the first matching known name.

    Returns
    -------
    selection_str
        A MDAnalysis selection string that selects exactly one atom.
    """

    donor_candidates = ["C56", "C58", "C138"]
    for name in donor_candidates:
        sel = f"segid {segid} and name {name}"
        if len(u.select_atoms(sel)) > 0:
            return sel

    # fallback: try 'C*' and take the first carbon atom in that segid
    sel = f"segid {segid} and name C*"
    ag = u.select_atoms(sel)
    if len(ag) == 0:
        raise ValueError(f"Could not infer donor atom in segid '{segid}'.")

    first = ag.atoms[0]
    return f"segid {first.segid} and resid {first.resid} and name {first.name}"


def default_catalytic_selections(u) -> CatalyticSelections:
    """Return the standard catalytic selection set used in the v1 notebook."""

    ftsw_297 = "segid PROD and resid 297 and (name OD1 OD2)"

    # These match the legacy metadata selections and are more robust across
    # segid conventions (HET* vs HAA1/HAB1 carbohydrate segids).
    l2_acceptor_candidates = [
        "(segid HETA and name O24) or (segid HAA1 and resname BNAG and name O4)",
        "segid HETA and name O24",
    ]
    l2_donor_candidates = [
        "(segid HETB and name C56) or (segid HAB1 and resname ANAM and name C1)",
        "segid HETB and name C56",
    ]

    l2_acceptor = _first_nonempty_selection(u, l2_acceptor_candidates)
    if l2_acceptor is None:
        raise ValueError(
            "Could not find acceptor atoms using known selections. "
            f"Tried: {l2_acceptor_candidates}"
        )

    l2_donor = _first_nonempty_selection(u, l2_donor_candidates)
    if l2_donor is None:
        # As a last resort, infer donor atom name from a known list.
        l2_donor = infer_l2_donor_selection(u, segid="HETB")

    # sanity checks (fail early with readable errors)
    if len(u.select_atoms(ftsw_297)) == 0:
        raise ValueError(f"Selection empty: {ftsw_297}")
    if len(u.select_atoms(l2_acceptor)) == 0:
        raise ValueError(f"Selection empty: {l2_acceptor}")
    if len(u.select_atoms(l2_donor)) == 0:
        raise ValueError(f"Selection empty: {l2_donor}")

    return CatalyticSelections(ftsw_297=ftsw_297, l2_acceptor=l2_acceptor, l2_donor=l2_donor)


def compute_catalytic_distance_series(
    u,
    *,
    selections: Optional[CatalyticSelections] = None,
) -> Tuple[Dict[str, np.ndarray], CatalyticSelections]:
    """Compute the three catalytic distance series used in the final figure.

    Series keys
    -----------
    - 'Acc-Don'
    - 'Acc-D297'
    - 'Don-D297'
    """

    if selections is None:
        selections = default_catalytic_selections(u)

    acc_don = calc_res_distance(u, "any", selections.l2_acceptor, selections.l2_donor)
    acc_d297 = calc_res_distance(u, "any", selections.l2_acceptor, selections.ftsw_297)
    don_d297 = calc_res_distance(u, "any", selections.l2_donor, selections.ftsw_297)

    series = {
        "Acc-Don": acc_don,
        "Acc-D297": acc_d297,
        "Don-D297": don_d297,
    }

    return series, selections
