from __future__ import annotations

from pathlib import Path

from benchmarks.pace2025_dominating_set import (
    SourceConfig,
    build_pace_dataset,
    domination_lower_bound,
    parse_pace_gr_text,
)
from dasbench.data import load_split
from dasbench.utils import load_json, public_instance


def test_parse_pace_gr_text_converts_one_based_edges() -> None:
    text = """
c tiny graph
p ds 4 3
1 2
2 3
4 3
"""
    instance = parse_pace_gr_text(text, instance_id="tiny", source_path="tiny.gr")

    assert instance["num_vertices"] == 4
    assert instance["edges"] == [[0, 1], [1, 2], [2, 3]]
    assert instance["pace_declared_edges"] == 3


def test_build_pace_dataset_uses_private_proxy_fields(tmp_path: Path) -> None:
    pace_root = tmp_path / "pace"
    (pace_root / "ds" / "exact").mkdir(parents=True)
    (pace_root / "private" / "ds" / "exact").mkdir(parents=True)
    graph_one = "p ds 4 3\n1 2\n2 3\n3 4\n"
    graph_two = "p ds 5 4\n1 2\n1 3\n1 4\n1 5\n"
    graph_private = "p ds 3 2\n1 2\n2 3\n"
    (pace_root / "ds" / "exact" / "exact_001.gr").write_text(graph_one, encoding="utf-8")
    (pace_root / "ds" / "exact" / "exact_002.gr").write_text(graph_two, encoding="utf-8")
    (pace_root / "private" / "ds" / "exact" / "private_exact_001.gr").write_text(
        graph_private,
        encoding="utf-8",
    )

    dataset_dir = tmp_path / "dataset"
    build_pace_dataset(
        dataset_dir=dataset_dir,
        output_root=tmp_path / "out",
        source_config=SourceConfig(pace_root=pace_root, cache_dir=tmp_path / "cache", github_ref="master"),
        track="exact",
        test_source="private",
        train_count=1,
        validation_count=1,
        test_count=1,
        public_start_index=1,
        test_start_index=1,
        reference_baselines=["marginal_gain_greedy"],
    )

    manifest = load_json(dataset_dir / "manifest.json")
    train = load_split(dataset_dir, "train")
    public_train = public_instance(train[0])

    assert manifest["problem"] == "mds"
    assert manifest["split_sizes"] == {"train": 1, "validation": 1, "test": 1}
    assert train[0]["optimum_objective"] == domination_lower_bound(train[0])
    assert "_pace_reference_solution" in train[0]
    assert "_pace_reference_solution" not in public_train
    assert "optimum_objective" not in public_train
