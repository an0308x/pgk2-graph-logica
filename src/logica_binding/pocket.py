"""Structure-derived protein regions used to localize LogiCA scoring.

The default LogiCA protein score masks a random 15% of all residues, so the
score reflects roughly 62 scattered positions with no guarantee that any of
them lie in the orthosteric inhibitor pocket. The DREAM PGK2 endpoint is
active-site inhibition, so these region definitions let a run state exactly
which residues define the protein-side score.

Regions come from ``artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json``,
produced by ``scripts/derive_pgk2_inhibitor_pocket.py`` as the union of
residue neighborhoods around the crystallographic compound-21 and compound-47
ligands. They are coordinate-derived neighborhoods, not energetic interactions.
"""

from __future__ import annotations

import json
from pathlib import Path

REGION_CHOICES = (
    "random",
    "inhibitor_pocket",
    "pocket_context_shell",
    "pocket_plus_3pg",
)


def load_pocket_region(pocket_json: Path, region: str) -> tuple[int, ...]:
    """Return sorted zero-based residue indices for a named region.

    ``random`` returns an empty tuple, meaning "keep the seeded random
    mask_fraction behaviour".
    """
    if region not in REGION_CHOICES:
        raise ValueError(f"Unknown region {region!r}; expected one of {REGION_CHOICES}")
    if region == "random":
        return ()
    payload = json.loads(pocket_json.read_text(encoding="utf-8"))
    direct = payload["direct_ligand_edges"]["residue_indices_0_based"]
    if region == "inhibitor_pocket":
        indices = direct
    elif region == "pocket_context_shell":
        indices = payload["protein_context_shell"]["residue_indices_0_based"]
    else:
        indices = list(direct) + list(
            payload["auxiliary_3pg_subsite"]["residue_indices_0_based"]
        )
    ordered = tuple(sorted(set(int(index) for index in indices)))
    if not ordered:
        raise ValueError(f"Region {region!r} contains no residues in {pocket_json}")
    return ordered
