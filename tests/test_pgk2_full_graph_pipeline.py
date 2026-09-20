from __future__ import annotations

from scripts.build_pgk2_canonical_index import (
    canonicalize_structure,
    scaffold_split,
    stable_bucket,
)
from scripts.precompute_pgk2_graph_shard import build_graph, shard_bounds


def test_canonicalization_and_scaffold_split_are_deterministic() -> None:
    first = canonicalize_structure("C(C)O")
    second = canonicalize_structure("CCO")
    assert first.status == second.status == "ok"
    assert first.canonical_smiles == second.canonical_smiles == "CCO"
    assert first.molecule_hash == second.molecule_hash
    assert first.scaffold_split == second.scaffold_split
    assert 0 <= first.sample_bucket < 1000


def test_same_scaffold_cannot_cross_internal_splits() -> None:
    assert scaffold_split("c1ccccc1", "Cc1ccccc1") == scaffold_split(
        "c1ccccc1", "Oc1ccccc1"
    )
    assert 0 <= stable_bucket("example", 17) < 17


def test_shard_bounds_cover_without_gaps() -> None:
    bounds = [shard_bounds(101, shard, 8) for shard in range(8)]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 101
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    assert sum(stop - start for start, stop in bounds) == 101


def test_2d_graph_contains_no_coordinates_or_radius_edges() -> None:
    result = build_graph(("CC(=O)Nc1ccncc1", "2d", 2026, 4.5, 0, 2, 2048))
    assert result.status == "ok"
    assert result.quality == "topology_2d"
    assert result.coordinates is None
    assert result.node_features is not None
    assert result.edge_index is not None
    assert result.edge_features is not None
    assert result.fingerprint is not None
    # Every covalent bond is represented in both directions; no radius edges.
    assert result.edge_index.shape[1] == 20
    assert result.edge_features.shape == (20, 7)
    assert result.fingerprint.shape == (256,)


def test_unoptimized_etkdg_mode_is_explicit() -> None:
    result = build_graph(("CCO", "etkdg", 2026, 4.5, 0, 2, 2048))
    assert result.status == "ok"
    assert result.quality in {"etkdg_unoptimized", "etkdg_random_coords_unoptimized"}
    assert result.coordinates is not None
