import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_pgk2_graph_v2 import main


def make_reports(root):
    for mode in ("on", "off"):
        for seed in (2026, 2027, 2028):
            path = root / f"graph_{mode}_seed{seed}" / "report.json"
            path.parent.mkdir()
            path.write_text(json.dumps({
                "protocol": "test_protocol", "split_manifest_sha256": "same_split",
                "graph_conditioning": mode, "ligand_mode": "2d",
                "graphs": {"base_attention_interface_restricted": True},
                "smoke_only": False, "run_purpose": "diagnostic",
                "pairs": {"train": 20, "dev": 4, "heldout": 4},
                "holdout": {"pairs": 4, "win_rate": .75,
                            "weighted_win_rate": .7 if mode == "on" else .6},
                "selected_epoch": 1,
                "training": {"history": [
                    {"epoch": 1, "dev": {"weighted_win_rate": .6}},
                    {"epoch": 2, "dev": {"weighted_win_rate": .6}},
                ]},
            }))


def test_matched_summary_and_refuse_overwrite(tmp_path, monkeypatch):
    make_reports(tmp_path)
    monkeypatch.setattr(sys, "argv", ["summary", "--run-dir", str(tmp_path)])
    main()
    result = json.loads((tmp_path / "comparison_report.json").read_text())
    assert result["conditions"]["on"]["weighted_win_rate"]["mean"] == .7
    assert result["graph_minus_control_weighted_win_rate"] == pytest.approx([.1] * 3)
    with pytest.raises(FileExistsError):
        main()


@pytest.mark.parametrize("field,value", [
    ("split_manifest_sha256", "other_split"),
    ("graph_conditioning", "off"),
    ("selected_epoch", 2),
    ("smoke_only", True),
])
def test_summary_rejects_unmatched_reports(tmp_path, monkeypatch, field, value):
    make_reports(tmp_path)
    path = tmp_path / "graph_on_seed2026" / "report.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["summary", "--run-dir", str(tmp_path)])
    with pytest.raises(ValueError):
        main()
