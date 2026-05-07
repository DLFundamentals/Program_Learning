from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from collections.abc import Sequence

from dasbench.problems.base import ExactSolveResult, ProblemDefinition, ScoreResult
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def validate_instance(instance: dict[str, object]) -> None:
    num_variables = int(instance["num_variables"])
    clauses = instance["clauses"]
    if num_variables < 3:
        raise ValueError("MAXSAT instances require at least 3 variables.")
    if not isinstance(clauses, list) or not clauses:
        raise ValueError("MAXSAT instances require a non-empty clause list.")
    for clause in clauses:
        if len(clause) != 3:
            raise ValueError(f"Clause {clause!r} does not have exactly 3 literals.")
        seen_variables: set[int] = set()
        for literal in clause:
            variable = abs(int(literal))
            if not 1 <= variable <= num_variables:
                raise ValueError(f"Literal {literal!r} is outside 1..{num_variables}.")
            if variable in seen_variables:
                raise ValueError(f"Clause {clause!r} repeats variable x{variable}.")
            seen_variables.add(variable)


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[bool]:
    if not isinstance(raw_solution, Sequence) or isinstance(raw_solution, (str, bytes)):
        raise TypeError("MAXSAT solver output must be a sequence of booleans.")
    return [bool(value) for value in raw_solution]


def validate_solution(solution: list[bool], instance: dict[str, object]) -> tuple[bool, str | None]:
    expected = int(instance["num_variables"])
    if len(solution) != expected:
        return False, f"Returned {len(solution)} values for {expected} variables."
    return True, None


def literal_is_satisfied(literal: int, assignment: Sequence[bool]) -> bool:
    value = assignment[abs(int(literal)) - 1]
    return value if literal > 0 else not value


def clause_is_satisfied(clause: Sequence[int], assignment: Sequence[bool]) -> bool:
    return any(literal_is_satisfied(literal, assignment) for literal in clause)


def count_satisfied_clauses(instance: dict[str, object], assignment: Sequence[bool]) -> int:
    return sum(1 for clause in instance["clauses"] if clause_is_satisfied(clause, assignment))


def literal_majority_counts(
    instance: dict[str, object],
    *,
    include_last_clause: bool = True,
) -> list[dict[str, int]]:
    num_variables = int(instance["num_variables"])
    counts = [{"positive": 0, "negative": 0} for _ in range(num_variables)]
    clauses = instance["clauses"] if include_last_clause else instance["clauses"][:-1]
    for clause in clauses:
        for literal in clause:
            bucket = "positive" if literal > 0 else "negative"
            counts[abs(int(literal)) - 1][bucket] += 1
    return counts


def literal_majority_assignment(
    instance: dict[str, object],
    *,
    include_last_clause: bool = True,
    ties_default_to: bool = False,
) -> list[bool]:
    assignment: list[bool] = []
    for bucket in literal_majority_counts(instance, include_last_clause=include_last_clause):
        if bucket["positive"] == bucket["negative"]:
            assignment.append(ties_default_to)
        else:
            assignment.append(bucket["positive"] > bucket["negative"])
    return assignment


def greedy_flip_improve(
    instance: dict[str, object],
    start_assignment: Sequence[bool],
    *,
    max_flips: int,
) -> list[bool]:
    assignment = list(start_assignment)
    if max_flips <= 0:
        return assignment
    best_score = count_satisfied_clauses(instance, assignment)
    num_variables = len(assignment)
    for _ in range(max_flips):
        best_variable: int | None = None
        best_neighbor = best_score
        for variable_index in range(num_variables):
            assignment[variable_index] = not assignment[variable_index]
            candidate_score = count_satisfied_clauses(instance, assignment)
            assignment[variable_index] = not assignment[variable_index]
            if candidate_score > best_neighbor:
                best_neighbor = candidate_score
                best_variable = variable_index
        if best_variable is None:
            break
        assignment[best_variable] = not assignment[best_variable]
        best_score = best_neighbor
    return assignment


def random_assignment(instance: dict[str, object]) -> list[bool]:
    rng = random.Random(f"maxsat:{instance['id']}:random")
    return [bool(rng.getrandbits(1)) for _ in range(int(instance["num_variables"]))]


def format_clause(clause: Sequence[int]) -> str:
    parts = []
    for literal in clause:
        prefix = "" if literal > 0 else "~"
        parts.append(f"{prefix}x{abs(int(literal))}")
    return "(" + " v ".join(parts) + ")"


def score_solution(instance: dict[str, object], solution: list[bool]) -> ScoreResult:
    valid, error = validate_solution(solution, instance)
    if not valid:
        return ScoreResult(
            is_valid=False,
            is_feasible=False,
            objective_value=0.0,
            normalized_quality=0.0,
            is_optimal=False,
            error=error,
        )
    objective_value = float(count_satisfied_clauses(instance, solution))
    optimum = float(instance.get("optimum_objective", len(instance["clauses"])))
    normalized = 0.0 if optimum <= 0 else objective_value / optimum
    return ScoreResult(
        is_valid=True,
        is_feasible=True,
        objective_value=objective_value,
        normalized_quality=min(1.0, normalized),
        is_optimal=abs(objective_value - optimum) < 1e-9,
    )


def summarize_training_data(
    train_instances: list[dict[str, object]],
    manifest: dict[str, object],
) -> dict[str, object]:
    num_variables = int(train_instances[0]["num_variables"])
    clause_counts = Counter()
    sign_pattern_counts = Counter()
    pair_weights: dict[tuple[int, int], int] = defaultdict(int)
    literal_bias = [{"positive": 0, "negative": 0} for _ in range(num_variables)]
    for instance in train_instances:
        for clause_index, clause in enumerate(instance["clauses"]):
            if clause_index in {0, len(instance["clauses"]) - 1}:
                sign_pattern_counts["".join("1" if literal > 0 else "0" for literal in clause)] += 1
            rendered = tuple(sorted(abs(int(literal)) for literal in clause))
            clause_counts[rendered] += 1
            variables = [abs(int(literal)) for literal in clause]
            for index, variable in enumerate(variables):
                bucket = "positive" if clause[index] > 0 else "negative"
                literal_bias[variable - 1][bucket] += 1
            for left_index in range(3):
                for right_index in range(left_index + 1, 3):
                    pair = tuple(sorted((variables[left_index], variables[right_index])))
                    pair_weights[pair] += 1
    top_pairs = sorted(pair_weights.items(), key=lambda item: (item[1], item[0]), reverse=True)[:12]
    top_variables = sorted(
        range(1, num_variables + 1),
        key=lambda variable: (
            abs(literal_bias[variable - 1]["positive"] - literal_bias[variable - 1]["negative"]),
            variable,
        ),
        reverse=True,
    )[:12]
    return {
        "problem": manifest["problem"],
        "family": manifest["family"],
        "num_instances": len(train_instances),
        "num_variables": num_variables,
        "num_clauses": len(train_instances[0]["clauses"]),
        "top_clause_signatures": dict(sign_pattern_counts.most_common(8)),
        "top_variable_bias": {
            f"x{variable}": literal_bias[variable - 1]
            for variable in top_variables
        },
        "top_variable_pairs": [
            {"variables": [f"x{left}", f"x{right}"], "cooccurrence_count": weight}
            for (left, right), weight in top_pairs
        ],
        "sample_instances": [
            {
                "id": instance["id"],
                "first_clauses": [format_clause(clause) for clause in instance["clauses"][:4]],
                "last_clause": format_clause(instance["clauses"][-1]),
            }
            for instance in train_instances[:3]
        ],
    }


def failure_case(
    instance: dict[str, object],
    solution: list[bool],
    score: ScoreResult,
    runtime_seconds: float,
) -> dict[str, object]:
    unsatisfied = [
        format_clause(clause)
        for clause in instance["clauses"]
        if not clause_is_satisfied(clause, solution)
    ][:3]
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "last_clause": format_clause(instance["clauses"][-1]),
        "unsatisfied_clause_examples": unsatisfied,
        "error": score.error,
    }


def solve_rc2_exact(instance: dict[str, object], *, solver_name: str = "g3", source: str | None = None) -> ExactSolveResult:
    start = time.perf_counter()
    wcnf = WCNF()
    for clause in instance["clauses"]:
        wcnf.append(list(clause), weight=1)
    with RC2(wcnf, solver=solver_name) as optimizer:
        model = optimizer.compute()
    if model is None:
        raise RuntimeError("RC2 failed to produce a model.")
    assignment = [False] * int(instance["num_variables"])
    for literal in model:
        variable = abs(int(literal))
        if 1 <= variable <= int(instance["num_variables"]):
            assignment[variable - 1] = int(literal) > 0
    objective_value = float(count_satisfied_clauses(instance, assignment))
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return ExactSolveResult(
        solution=assignment,
        objective_value=objective_value,
        runtime_ms=runtime_ms,
        source=source or f"pysat-rc2-{solver_name}",
    )


def solve_exact(instance: dict[str, object]) -> ExactSolveResult:
    return solve_rc2_exact(instance, solver_name="g3", source="pysat-rc2")


def baseline_registry() -> dict[str, object]:
    return {
        "random": random_assignment,
        "polarity": lambda instance: literal_majority_assignment(instance, include_last_clause=True),
        "local_search": lambda instance: greedy_flip_improve(
            instance,
            literal_majority_assignment(instance, include_last_clause=True),
            max_flips=max(12, int(instance["num_variables"]) // 2),
        ),
        "rc2_exact": lambda instance: list(solve_exact(instance).solution),
        "rc2_glucose4": lambda instance: list(
            solve_rc2_exact(instance, solver_name="g4", source="pysat-rc2-glucose4").solution
        ),
        "rc2_minisat22": lambda instance: list(
            solve_rc2_exact(instance, solver_name="m22", source="pysat-rc2-minisat22").solution
        ),
        "rc2_cadical195": lambda instance: list(
            solve_rc2_exact(instance, solver_name="cd19", source="pysat-rc2-cadical195").solution
        ),
    }


PROBLEM = ProblemDefinition(
    name="maxsat",
    description="Distribution-aware synthesis benchmark for unweighted MAX-3SAT / MaxSAT-style instances.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is satisfied_clauses / optimum_satisfied_clauses",
    },
    instance_schema_version="maxsat.v1",
    default_instance_params={
        "num_variables": 28,
        "num_clauses": 72,
        "literal_agreement_probability": 0.64,
        "repair_probability": 0.25,
    },
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_exact,
)
