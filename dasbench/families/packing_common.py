from __future__ import annotations

import random


def build_packing_instance(
    instance_id: str,
    *,
    values: list[float] | list[int],
    weights: list[list[float]] | list[list[int]],
    capacities: list[float] | list[int],
    private_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    instance: dict[str, object] = {
        "id": instance_id,
        "num_items": len(values),
        "num_resources": len(capacities),
        "values": values,
        "weights": weights,
        "capacities": capacities,
    }
    for key, value in (private_metadata or {}).items():
        instance[f"_{key.removeprefix('_')}"] = value
    return instance


def capacity_from_tightness(
    weights: list[list[int]] | list[list[float]],
    tightness_by_resource: list[float],
    *,
    integral: bool,
) -> list[int] | list[float]:
    capacities: list[int] | list[float] = []
    for resource, tightness in enumerate(tightness_by_resource):
        total = sum(float(row[resource]) for row in weights)
        minimum = max(float(row[resource]) for row in weights)
        capacity = max(minimum, total * tightness)
        capacities.append(max(1, int(round(capacity))) if integral else round(capacity, 6))
    return capacities


def balanced_classes(num_items: int, num_classes: int) -> list[int]:
    classes = [index % num_classes for index in range(num_items)]
    return classes


def shuffled_balanced_classes(rng: random.Random, num_items: int, num_classes: int) -> list[int]:
    classes = balanced_classes(num_items, num_classes)
    rng.shuffle(classes)
    return classes


def positive_int_jitter(rng: random.Random, center: float, *, spread: float = 0.22, minimum: int = 1) -> int:
    return max(minimum, int(round(center * rng.uniform(1.0 - spread, 1.0 + spread))))


def rounded_positive(value: float, *, minimum: int = 1) -> int:
    return max(minimum, int(round(value)))
