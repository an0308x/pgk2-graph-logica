from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_pgk2_competition_pairs import building_block_keys, competition_log_ratio


def test_competition_log_ratio_shrinks_zero_counts() -> None:
    assert math.isfinite(competition_log_ratio(4, 0))
    assert competition_log_ratio(4, 0) > competition_log_ratio(4, 2)


def test_building_block_keys_require_open_del_identifier() -> None:
    assert building_block_keys("LIB-A-B-C") == (
        "LIB:A:B",
        "LIB:A:C",
        "LIB:B:C",
    )
    assert building_block_keys("bad-id") is None


def test_production_competition_pairs_obey_active_site_rules() -> None:
    path = ROOT / "artifacts/pgk2_competition_pairs/pgk2_competition_pairs.parquet"
    if not path.exists():
        return
    pairs = pl.read_parquet(path)
    assert pairs.height > 0
    assert pairs.filter(pl.col("positive_count_PGK2") < 4).is_empty()
    assert pairs.filter(pl.col("negative_count_PGK2") < 4).is_empty()
    assert pairs.filter(
        pl.col("positive_count_inhibitor") >= 0.1 * pl.col("positive_count_PGK2")
    ).is_empty()
    assert pairs.filter(
        pl.col("negative_count_inhibitor") < 0.5 * pl.col("negative_count_PGK2")
    ).is_empty()
    assert pairs.filter(pl.col("target_count_fold_difference") > 2.0).is_empty()


def test_inhibitor_pocket_has_nested_direct_and_context_regions() -> None:
    path = ROOT / "artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json"
    if not path.exists():
        return
    pocket = json.loads(path.read_text())
    direct = set(pocket["direct_ligand_edges"]["residue_indices_0_based"])
    context = set(pocket["protein_context_shell"]["residue_indices_0_based"])
    assert len(direct) == 26
    assert len(context) == 49
    assert direct < context
    assert 337 in direct  # Human residue 338 is contributed by compound 21 at 6 A.


def test_pocket_regions_are_nested_and_in_sequence_range() -> None:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from logica_binding.data import read_fasta
    from logica_binding.pocket import load_pocket_region

    root = Path(__file__).resolve().parents[1]
    pocket_json = root / "artifacts" / "pgk2_inhibitor_pocket" / "inhibitor_pocket.json"
    sequence = read_fasta(root / "data" / "proteins" / "PGK2_P07205.fasta")

    assert load_pocket_region(pocket_json, "random") == ()
    direct = load_pocket_region(pocket_json, "inhibitor_pocket")
    shell = load_pocket_region(pocket_json, "pocket_context_shell")
    with_3pg = load_pocket_region(pocket_json, "pocket_plus_3pg")

    assert len(direct) == 26
    assert set(direct).issubset(set(shell))
    assert set(direct).issubset(set(with_3pg))
    assert len(with_3pg) > len(direct)
    for region in (direct, shell, with_3pg):
        assert region == tuple(sorted(set(region)))
        assert min(region) >= 0 and max(region) < len(sequence)


def test_positional_mask_targets_requested_residues_only() -> None:
    from pathlib import Path
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from logica_binding.model import _make_positional_masked_inputs

    # Two special tokens (<cls> ... <eos>) around five residues.
    input_ids = torch.tensor([[0, 11, 12, 13, 14, 15, 2]])
    attention_mask = torch.ones_like(input_ids)
    special_tokens_mask = torch.tensor([[1, 0, 0, 0, 0, 0, 1]])

    masked, selected = _make_positional_masked_inputs(
        input_ids, attention_mask, special_tokens_mask, 32, [0, 3], sequence_length=5
    )

    # Residue i must land on token i + 1, and nothing else may change.
    assert selected[0].tolist() == [False, True, False, False, True, False, False]
    assert masked[0].tolist() == [0, 32, 12, 13, 32, 15, 2]

    for bad in ([5], [-1]):
        try:
            _make_positional_masked_inputs(
                input_ids, attention_mask, special_tokens_mask, 32, bad, sequence_length=5
            )
        except ValueError:
            continue
        raise AssertionError(f"expected out-of-range rejection for {bad}")


def test_building_block_split_yields_two_of_three_shared_keys() -> None:
    from pathlib import Path
    import sys

    import polars as pl

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_pgk2_unobserved_negative_pairs import with_building_blocks

    # Library names contain digits and underscores but never a hyphen, so a
    # 4-way split on "-" always isolates library and three building blocks.
    frame = pl.DataFrame(
        {"compound": ["qDOS18_3-172-977-552", "qDOS13-20-159-91"], "SMILES": ["CC", "CCC"]}
    )
    out = with_building_blocks(frame.lazy()).collect()

    assert out["lib"].to_list() == ["qDOS18_3", "qDOS13"]
    assert out["b1"].to_list() == ["172", "20"]
    assert out["b3"].to_list() == ["552", "91"]
    assert out["k12"].to_list() == ["qDOS18_3:172:977", "qDOS13:20:159"]
    assert out["k13"].to_list() == ["qDOS18_3:172:552", "qDOS13:20:91"]
    assert out["k23"].to_list() == ["qDOS18_3:977:552", "qDOS13:159:91"]


def test_unobserved_negative_pairs_hold_their_invariants() -> None:
    """Guards the properties that make these pairs usable as training signal."""
    from pathlib import Path

    import polars as pl

    root = Path(__file__).resolve().parents[1]
    pairs_path = (
        root / "artifacts" / "pgk2_unobserved_negative_pairs" / "pgk2_unobserved_negative_pairs.parquet"
    )
    if not pairs_path.exists():
        import pytest

        pytest.skip("run scripts/build_pgk2_unobserved_negative_pairs.py first")

    pairs = pl.read_parquet(pairs_path)
    observed = set(
        pl.scan_parquet(root / "PGK2_selection.parquet").select("SMILES").collect()["SMILES"]
    )
    negatives = set(pairs["negative_smiles"])
    positives = set(pairs["positive_smiles"])

    # A negative is only a negative if the sequencer never saw it.
    assert not (negatives & observed)
    # Anchors must come from the selection file.
    assert not (positives - observed)

    for anchor, negative in pairs.select("anchor_compound", "negative_compound").iter_rows():
        left, right = anchor.split("-"), negative.split("-")
        assert left[0] == right[0], "negative must come from the anchor's library"
        shared = sum(1 for i in (1, 2, 3) if left[i] == right[i])
        assert shared == 2, "negative must share exactly two of three building blocks"

    assert pairs["pair_weight"].min() > 0
    assert pairs["pair_weight"].max() <= 1.0
