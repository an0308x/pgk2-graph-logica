from pathlib import Path
import sys
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_pgk2_v3_baselines import ranking_schedule, pair_details
from audit_pgk2_v3_supervision import audit_counts


def test_ranking_schedule_exactly_replays_training_selection():
    frame = pl.DataFrame({"split":["train"]*600+["dev"]*40,
        **{f"confidence_{i}":[float((j+i)%3 == 0) for j in range(640)] for i in range(3)}})
    actual = list(ranking_schedule(frame, 17))
    # Literal recipe from the production trainer, including shard-dependent RNG.
    mask = frame["split"].to_numpy() == "train"
    pools = [np.flatnonzero(mask & (frame[f"confidence_{i}"].to_numpy()>0)) for i in range(3)]
    expected = []
    for batch_index, offset in enumerate(range(0,int(mask.sum()),64)):
        if batch_index%4 == 0:
            available = [g for g in pools if len(g)>1]
            pool = available[(batch_index//4)%len(available)]
            draw = np.random.default_rng(2026+17*10000+batch_index)
            expected.append(draw.choice(pool,min(32,len(pool)),replace=False))
    assert len(actual) == len(expected) == 3
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)
        assert (a<600).all()


def test_supervision_audit_counts_excluded_high_target_rows():
    frame = pl.DataFrame({"source_rows":[1,2,1],"count_PGK2":[100,40,1],
        "count_PGK2_with_inhibitor":[0,2,0],"count_NTC_selection":[0,0,0],
        "count_NTC_supplement":[0,0,0],"ntc_supplement_rows":[0,0,0],
        "historic_hits":[0,1,0],"confidence_0":[0.,.01,0.],
        "confidence_1":[0.,0.,0.],"confidence_2":[0.,0.,0.]})
    counts = audit_counts(frame)
    assert counts["competition_supervised"] == 1
    assert counts["target_ge_20"] == 2
    assert counts["target_ge_20_zero_competitor"] == 1
    assert counts["target_ge_20_zero_competitor_zero_ntc_no_history"] == 1
    assert counts["duplicate_source_molecules"] == 1


def test_weight_diagnostics_exclude_unknowns_and_small_differences():
    rows = pl.DataFrame({"proxy_0":[0.,0.,2.,-4.],"confidence_0":[.2,.4,.1,0.]})
    a,b,w = pair_details(None,rows,0)
    np.testing.assert_array_equal(a,[0,1])
    np.testing.assert_array_equal(b,[2,2])
    np.testing.assert_allclose(w,[.1,.1])
