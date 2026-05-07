from __future__ import annotations

from collections import defaultdict

from dasbench.problems.graph_utils import (
    adjacency_sets,
    connected_component_sizes,
    dsatur_coloring,
    edge_density,
    greedy_coloring_largest_degree,
    greedy_coloring_random_order,
    greedy_dominating_set_max_gain,
    greedy_independent_set_by_score,
    greedy_independent_set_with_local_improvement,
    prune_redundant_dominating_vertices_fast,
    sample_independent_set_random,
    smallest_last_coloring,
)
from dasbench.problems.maxsat import (
    count_satisfied_clauses,
    greedy_flip_improve,
    literal_majority_assignment,
    literal_majority_counts,
)
from dasbench.problems.packing_utils import (
    FEASIBILITY_TOLERANCE,
    greedy_binary_solution,
    greedy_fractional_solution,
    redundancy_improved_binary_solution,
    solve_packing_lp_glop,
)
from dasbench.problems.tsp_utils import (
    farthest_insertion_tour,
    nearest_insertion_tour,
    nearest_neighbor_tour,
    two_opt_improve,
)


def _maxsat_signature(instance: dict[str, object]) -> str:
    first_clause = instance["clauses"][0]
    last_clause = instance["clauses"][-1]
    first_bits = "".join("1" if literal > 0 else "0" for literal in first_clause)
    last_bits = "".join("1" if literal > 0 else "0" for literal in last_clause)
    return f"{first_bits}|{last_bits}"


def _set_anchor_values_from_last_clause(assignment: list[bool], instance: dict[str, object]) -> list[bool]:
    updated = list(assignment)
    for index, literal in enumerate(instance["clauses"][-1][:3]):
        updated[index] = literal > 0
    return updated


def _counts_to_assignment(counts: list[dict[str, int]]) -> tuple[list[bool], list[float]]:
    assignment: list[bool] = []
    confidence: list[float] = []
    for bucket in counts:
        total = bucket["positive"] + bucket["negative"]
        assignment.append(bucket["positive"] >= bucket["negative"])
        confidence.append(0.0 if total == 0 else abs(bucket["positive"] - bucket["negative"]) / total)
    return assignment, confidence


def _analyze_maxsat(train_instances: list[dict[str, object]], config: dict[str, object]) -> dict[str, object]:
    num_variables = int(train_instances[0]["num_variables"])
    global_counts = [{"positive": 0, "negative": 0} for _ in range(num_variables)]
    signature_counts: dict[str, list[dict[str, int]]] = defaultdict(
        lambda: [{"positive": 0, "negative": 0} for _ in range(num_variables)]
    )
    for instance in train_instances:
        signature = _maxsat_signature(instance)
        counts = literal_majority_counts(instance, include_last_clause=False)
        for variable_index, bucket in enumerate(counts):
            global_counts[variable_index]["positive"] += bucket["positive"]
            global_counts[variable_index]["negative"] += bucket["negative"]
            signature_counts[signature][variable_index]["positive"] += bucket["positive"]
            signature_counts[signature][variable_index]["negative"] += bucket["negative"]
    global_assignment, global_confidence = _counts_to_assignment(global_counts)
    signature_assignments: dict[str, list[bool]] = {}
    signature_confidences: dict[str, list[float]] = {}
    for signature, counts in signature_counts.items():
        signature_assignments[signature], signature_confidences[signature] = _counts_to_assignment(counts)
    return {
        "global_assignment": global_assignment,
        "global_confidence": global_confidence,
        "signature_assignments": signature_assignments,
        "signature_confidences": signature_confidences,
    }


def _solve_maxsat(
    instance: dict[str, object],
    analysis: dict[str, object] | None,
    config: dict[str, object],
) -> list[bool]:
    strategy = str(config["solver_strategy"])
    signature = _maxsat_signature(instance)
    if strategy == "instance_polarity" or analysis is None:
        assignment = literal_majority_assignment(instance, include_last_clause=True)
    elif strategy == "signature_lookup":
        lookup = analysis.get("signature_assignments", {})
        assignment = list(lookup.get(signature, analysis["global_assignment"]))
    else:
        assignment = list(analysis["global_assignment"])

    if bool(config.get("mix_with_instance_polarity", False)) and analysis is not None:
        fallback = literal_majority_assignment(instance, include_last_clause=False)
        confidences = analysis.get("signature_confidences", {}).get(
            signature,
            analysis.get("global_confidence", [0.0] * len(assignment)),
        )
        threshold = float(config.get("pattern_confidence_threshold", 0.0))
        for index, confidence in enumerate(confidences):
            if confidence < threshold:
                assignment[index] = fallback[index]
    if bool(config.get("force_last_clause_anchor", True)):
        assignment = _set_anchor_values_from_last_clause(assignment, instance)
    local_search_flips = int(config.get("local_search_flips", 0))
    if local_search_flips > 0:
        assignment = greedy_flip_improve(instance, assignment, max_flips=local_search_flips)
        if bool(config.get("force_last_clause_anchor", True)):
            assignment = _set_anchor_values_from_last_clause(assignment, instance)
    return assignment


def _analyze_graph(train_instances: list[dict[str, object]]) -> dict[str, object]:
    densities: list[float] = []
    component_sizes: list[list[int]] = []
    average_degrees: list[float] = []
    for instance in train_instances:
        adjacency = adjacency_sets(int(instance["num_vertices"]), instance["edges"])
        densities.append(edge_density(int(instance["num_vertices"]), instance["edges"]))
        average_degrees.append(sum(len(neighbors) for neighbors in adjacency) / max(1, len(adjacency)))
        component_sizes.append(connected_component_sizes(int(instance["num_vertices"]), adjacency)[:5])
    return {
        "density_mean": sum(densities) / len(densities),
        "average_degree_mean": sum(average_degrees) / len(average_degrees),
        "component_examples": component_sizes[:4],
    }


def _solve_mis(instance: dict[str, object], analysis: dict[str, object] | None, config: dict[str, object]) -> list[int]:
    strategy = str(config["solver_strategy"])
    if strategy == "random_greedy":
        solution = sample_independent_set_random(instance, seed_label="template-mis")
    elif strategy == "min_degree":
        solution = greedy_independent_set_by_score(
            instance,
            score_fn=lambda vertex, adjacency, remaining: len(adjacency[vertex] & remaining),
        )
    elif strategy == "ratio_greedy":
        solution = greedy_independent_set_by_score(
            instance,
            score_fn=lambda vertex, adjacency, remaining: (
                len(adjacency[vertex] & remaining),
                -sum(len(adjacency[neighbor] & remaining) for neighbor in adjacency[vertex] & remaining),
            ),
        )
    elif strategy == "density_adaptive" and analysis is not None:
        if float(analysis.get("density_mean", 0.0)) > float(config.get("density_threshold", 0.22)):
            solution = greedy_independent_set_by_score(
                instance,
                score_fn=lambda vertex, adjacency, remaining: (
                    len(adjacency[vertex] & remaining),
                    vertex,
                ),
            )
        else:
            solution = greedy_independent_set_with_local_improvement(instance)
    else:
        solution = greedy_independent_set_with_local_improvement(instance)
    return sorted(solution)


def _solve_mds(instance: dict[str, object], analysis: dict[str, object] | None, config: dict[str, object]) -> list[int]:
    strategy = str(config["solver_strategy"])
    if strategy in {"high_degree", "marginal_gain"}:
        solution = greedy_dominating_set_max_gain(instance)
    elif strategy == "overlap_hybrid" and analysis is not None:
        solution = greedy_dominating_set_max_gain(instance, prefer_low_overlap=True)
    else:
        solution = greedy_dominating_set_max_gain(instance, prefer_low_overlap=True)
    if bool(config.get("prune_redundant", True)):
        solution = prune_redundant_dominating_vertices_fast(instance, solution)
    return sorted(solution)


def _solve_coloring(
    instance: dict[str, object],
    analysis: dict[str, object] | None,
    config: dict[str, object],
) -> list[int]:
    strategy = str(config["solver_strategy"])
    if strategy == "random_greedy":
        return greedy_coloring_random_order(instance, seed_label="template-coloring")
    if strategy == "largest_degree":
        return greedy_coloring_largest_degree(instance)
    if strategy == "smallest_last":
        return smallest_last_coloring(instance)
    if strategy == "density_adaptive" and analysis is not None:
        if float(analysis.get("density_mean", 0.0)) > float(config.get("density_threshold", 0.42)):
            return dsatur_coloring(instance)
        return smallest_last_coloring(instance)
    return dsatur_coloring(instance)


def _analyze_tsp(train_instances: list[dict[str, object]]) -> dict[str, object]:
    aspect_ratios: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    for instance in train_instances:
        points = [(float(x), float(y)) for x, y in instance["points"]]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        widths.append(width)
        heights.append(height)
        larger = max(width, height, 1e-9)
        smaller = max(min(width, height), 1e-9)
        aspect_ratios.append(larger / smaller)
    return {
        "aspect_ratio_mean": sum(aspect_ratios) / len(aspect_ratios),
        "bbox_width_mean": sum(widths) / len(widths),
        "bbox_height_mean": sum(heights) / len(heights),
    }


def _solve_tsp(instance: dict[str, object], analysis: dict[str, object] | None, config: dict[str, object]) -> list[int]:
    strategy = str(config["solver_strategy"])
    rounds = int(config.get("two_opt_rounds", 0))
    if strategy == "random":
        from dasbench.problems.tsp_utils import random_tour

        return random_tour(instance, seed_label="template-tsp")
    if strategy == "nearest_neighbor":
        return nearest_neighbor_tour(instance)
    if strategy == "nearest_insertion":
        return nearest_insertion_tour(instance)
    if strategy == "farthest_insertion":
        return farthest_insertion_tour(instance)
    if strategy == "two_opt_nearest":
        return two_opt_improve(instance, nearest_neighbor_tour(instance), max_rounds=max(2, rounds))
    if strategy == "two_opt_farthest":
        return two_opt_improve(instance, farthest_insertion_tour(instance), max_rounds=max(2, rounds))
    if strategy == "structure_adaptive" and analysis is not None:
        aspect_ratio = float(analysis.get("aspect_ratio_mean", 1.0))
        threshold = float(config.get("aspect_ratio_threshold", 1.35))
        if aspect_ratio > threshold:
            base = nearest_insertion_tour(instance)
        else:
            base = nearest_neighbor_tour(instance)
        if rounds > 0:
            return two_opt_improve(instance, base, max_rounds=rounds)
        return base
    return two_opt_improve(instance, nearest_neighbor_tour(instance), max_rounds=max(2, rounds))


def _analyze_packing(train_instances: list[dict[str, object]]) -> dict[str, object]:
    num_resources = int(train_instances[0]["num_resources"])
    tightness_sums = [0.0] * num_resources
    density_sums = [0.0] * num_resources
    for instance in train_instances:
        weights = [[float(value) for value in row] for row in instance["weights"]]
        capacities = [float(value) for value in instance["capacities"]]
        values = [float(value) for value in instance["values"]]
        for resource in range(num_resources):
            total_weight = sum(row[resource] for row in weights)
            tightness_sums[resource] += capacities[resource] / max(total_weight, FEASIBILITY_TOLERANCE)
            density_sums[resource] += sum(
                values[item] / max(weights[item][resource], FEASIBILITY_TOLERANCE)
                for item in range(int(instance["num_items"]))
            ) / max(1, int(instance["num_items"]))
    count = max(1, len(train_instances))
    tightness = [value / count for value in tightness_sums]
    per_resource_density = [value / count for value in density_sums]
    tightest_resource = min(range(num_resources), key=lambda resource: tightness[resource])
    return {
        "capacity_tightness_by_resource": tightness,
        "per_resource_density_mean": per_resource_density,
        "tightest_resource": tightest_resource,
    }


def _resource_prices_from_analysis(analysis: dict[str, object] | None, instance: dict[str, object]) -> list[float] | None:
    if analysis is None:
        return None
    tightness = analysis.get("capacity_tightness_by_resource")
    if not isinstance(tightness, list):
        return None
    return [1.0 / max(float(value), FEASIBILITY_TOLERANCE) for value in tightness]


def _solve_packing_lp(
    instance: dict[str, object],
    analysis: dict[str, object] | None,
    config: dict[str, object],
) -> list[float]:
    strategy = str(config["solver_strategy"])
    if strategy == "uniform_fraction":
        capacities = [float(value) for value in instance["capacities"]]
        weights = [[float(value) for value in row] for row in instance["weights"]]
        fractions = []
        for resource, capacity in enumerate(capacities):
            total = sum(row[resource] for row in weights)
            fractions.append(1.0 if total <= FEASIBILITY_TOLERANCE else capacity / total)
        fraction = max(0.0, min(1.0, min(fractions)))
        return [fraction for _ in range(int(instance["num_items"]))]
    if strategy == "tight_resource_density":
        return greedy_fractional_solution(instance, resource_prices=_resource_prices_from_analysis(analysis, instance))
    if strategy == "lp_relaxation":
        return list(solve_packing_lp_glop(instance).solution)
    return greedy_fractional_solution(instance)


def _solve_mdkp(instance: dict[str, object], analysis: dict[str, object] | None, config: dict[str, object]) -> list[int]:
    strategy = str(config["solver_strategy"])
    if strategy == "tight_resource_greedy":
        return greedy_binary_solution(instance, resource_prices=_resource_prices_from_analysis(analysis, instance))
    if strategy == "redundancy_improved":
        return redundancy_improved_binary_solution(instance)
    if strategy == "lp_rounding":
        relaxation = solve_packing_lp_glop(instance).solution
        order = sorted(
            range(int(instance["num_items"])),
            key=lambda item: (float(relaxation[item]), float(instance["values"][item])),
            reverse=True,
        )
        selected: list[int] = []
        usage = [0.0] * int(instance["num_resources"])
        capacities = [float(value) for value in instance["capacities"]]
        weights = [[float(value) for value in row] for row in instance["weights"]]
        for item in order:
            if all(
                usage[resource] + weights[item][resource] <= capacities[resource] + FEASIBILITY_TOLERANCE
                for resource in range(int(instance["num_resources"]))
            ):
                selected.append(item)
                for resource in range(int(instance["num_resources"])):
                    usage[resource] += weights[item][resource]
        return sorted(selected)
    return greedy_binary_solution(instance)


def analyze_with_config(
    train_instances: list[dict[str, object]],
    manifest: dict[str, object] | None,
    config: dict[str, object],
) -> dict[str, object]:
    problem = str(config["problem"])
    if problem == "maxsat":
        return _analyze_maxsat(train_instances, config)
    if problem == "tsp":
        return _analyze_tsp(train_instances)
    if problem in {"packing_lp", "mdkp"}:
        return _analyze_packing(train_instances)
    return _analyze_graph(train_instances)


def solve_with_config(
    instance: dict[str, object],
    analysis: dict[str, object] | None,
    manifest: dict[str, object] | None,
    config: dict[str, object],
):
    problem = str(config["problem"])
    if problem == "maxsat":
        return _solve_maxsat(instance, analysis, config)
    if problem == "mis":
        return _solve_mis(instance, analysis, config)
    if problem == "mds":
        return _solve_mds(instance, analysis, config)
    if problem == "coloring":
        return _solve_coloring(instance, analysis, config)
    if problem == "tsp":
        return _solve_tsp(instance, analysis, config)
    if problem == "packing_lp":
        return _solve_packing_lp(instance, analysis, config)
    if problem == "mdkp":
        return _solve_mdkp(instance, analysis, config)
    raise ValueError(f"Unsupported template runtime problem: {problem}")
