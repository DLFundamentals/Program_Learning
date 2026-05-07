from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.graph_common import (
    add_biclique,
    add_clique,
    add_crown,
    add_cycle,
    add_edge,
    add_path,
    add_random_noise_edges,
    build_graph_instance,
    partition_vertices,
    sample_block_sizes,
)


@dataclass(frozen=True)
class FamilyState:
    blocks: list[list[int]]
    regime_motifs: dict[int, list[str]]


def _apply_motif(edges: set[tuple[int, int]], motif: str, block: list[int]) -> None:
    if len(block) < 3:
        add_path(edges, block)
        return
    if motif == "clique":
        add_clique(edges, block)
        return
    if motif == "cycle":
        add_cycle(edges, block)
        return
    midpoint = len(block) // 2
    left = block[:midpoint]
    right = block[midpoint:]
    if len(left) < 2 or len(right) < 2:
        add_path(edges, block)
        return
    if motif == "biclique":
        add_biclique(edges, left, right)
        return
    if len(left) != len(right):
        add_cycle(edges, block)
        return
    add_crown(edges, left, right)


def build_state(context: dict[str, object]) -> FamilyState:
    params = context["instance_params"]
    seeds = context["seeds"]
    num_vertices = int(params["num_vertices"])
    rng = random.Random(int(seeds["family"]))
    block_sizes = sample_block_sizes(rng, num_vertices, min_size=5, max_size=8)
    blocks = partition_vertices(num_vertices, block_sizes)
    motif_sets = [
        ["clique", "cycle", "biclique", "crown"],
        ["cycle", "biclique", "clique", "crown"],
        ["biclique", "crown", "cycle", "clique"],
    ]
    regime_motifs = {
        regime: [motif_sets[regime][index % len(motif_sets[regime])] for index in range(len(blocks))]
        for regime in range(3)
    }
    return FamilyState(blocks=blocks, regime_motifs=regime_motifs)


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    regime = rng.randrange(3)
    edges: set[tuple[int, int]] = set()
    for block, motif in zip(state.blocks, state.regime_motifs[regime], strict=True):
        _apply_motif(edges, motif, block)
    for block_index, left_block in enumerate(state.blocks[:-1]):
        right_block = state.blocks[block_index + 1]
        if not left_block or not right_block:
            continue
        add_edge(edges, left_block[0], right_block[-1])
        if rng.random() < 0.55:
            add_edge(edges, left_block[-1], right_block[0])
        if block_index + 2 < len(state.blocks) and rng.random() < 0.4:
            skip_block = state.blocks[block_index + 2]
            add_edge(edges, left_block[len(left_block) // 2], skip_block[len(skip_block) // 2])
    add_random_noise_edges(
        rng,
        edges,
        list(range(int(context["instance_params"]["num_vertices"]))),
        probability=0.025 + 0.01 * regime,
    )
    return build_graph_instance(
        instance_id,
        int(context["instance_params"]["num_vertices"]),
        list(edges),
    )


FAMILY = FamilyDefinition(
    problem="mis",
    name="motif_bridge_mixture_v1",
    description=(
        "Paper-grade MIS family built from latent motif libraries such as cliques, cycles, bicliques, and crown-style gadgets with sparse bridge patterns."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Each instance samples one of three regimes that assigns a motif sequence of cliques, cycles, bicliques, and crown gadgets to vertex blocks.",
        "signals": ["latent motif sequence", "sparse adjacent bridges", "skip bridges", "regime-dependent noise"],
        "solver_hint": "Decompose into block motifs, solve each motif with MIS-specific rules, and account for bridge conflicts.",
    },
)
