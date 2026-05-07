from __future__ import annotations

from dataclasses import asdict, dataclass, field


DEFAULT_SPLIT_SIZES = {"train": 256, "validation": 128, "test": 10_000}
DEFAULT_SEEDS = {
    "family": 17,
    "train": 101,
    "validation": 202,
    "test": 303,
}


@dataclass(frozen=True)
class BenchmarkSpec:
    problem: str
    family: str
    instance_params: dict[str, object] = field(default_factory=dict)
    family_params: dict[str, object] = field(default_factory=dict)
    split_sizes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SPLIT_SIZES))
    seeds: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SEEDS))
    compute_optima: bool = True

    def to_reproducibility_record(self) -> dict[str, object]:
        return asdict(self)
