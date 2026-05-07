from __future__ import annotations

from dasbench.families.base import FamilyDefinition
from dasbench.families.coloring import FAMILIES as COLORING_FAMILIES
from dasbench.families.mdkp import FAMILIES as MDKP_FAMILIES
from dasbench.families.mds import FAMILIES as MDS_FAMILIES
from dasbench.families.maxsat import FAMILIES as MAXSAT_FAMILIES
from dasbench.families.mis import FAMILIES as MIS_FAMILIES
from dasbench.families.packing_lp import FAMILIES as PACKING_LP_FAMILIES
from dasbench.families.tsp import FAMILIES as TSP_FAMILIES

FAMILY_REGISTRY: dict[str, dict[str, FamilyDefinition]] = {
    "coloring": dict(COLORING_FAMILIES),
    "mdkp": dict(MDKP_FAMILIES),
    "maxsat": dict(MAXSAT_FAMILIES),
    "mis": dict(MIS_FAMILIES),
    "mds": dict(MDS_FAMILIES),
    "packing_lp": dict(PACKING_LP_FAMILIES),
    "tsp": dict(TSP_FAMILIES),
}


def register_problem_families(problem: str, families: dict[str, FamilyDefinition]) -> None:
    FAMILY_REGISTRY[problem] = dict(families)


def available_family_names(problem: str | None = None) -> list[str] | dict[str, list[str]]:
    if problem is None:
        return {name: sorted(families) for name, families in FAMILY_REGISTRY.items()}
    try:
        return sorted(FAMILY_REGISTRY[problem])
    except KeyError as exc:
        raise ValueError(
            f"Unknown problem `{problem}`. Available problems: {', '.join(sorted(FAMILY_REGISTRY))}"
        ) from exc


def get_family_definition(problem: str, family: str) -> FamilyDefinition:
    try:
        return FAMILY_REGISTRY[problem][family]
    except KeyError as exc:
        raise ValueError(
            f"Unknown family `{family}` for problem `{problem}`. "
            f"Available families: {', '.join(sorted(FAMILY_REGISTRY.get(problem, {})))}"
        ) from exc
