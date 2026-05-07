from __future__ import annotations

from dasbench.families.coloring.cluster_ring_mix_v1 import FAMILY as CLUSTER_RING_MIX_V1
from dasbench.families.coloring.planted_palette_overlap_v1 import FAMILY as PLANTED_PALETTE_OVERLAP_V1
from dasbench.families.coloring.separator_palette_trap_v1 import FAMILY as SEPARATOR_PALETTE_TRAP_V1

FAMILIES = {
    CLUSTER_RING_MIX_V1.name: CLUSTER_RING_MIX_V1,
    PLANTED_PALETTE_OVERLAP_V1.name: PLANTED_PALETTE_OVERLAP_V1,
    SEPARATOR_PALETTE_TRAP_V1.name: SEPARATOR_PALETTE_TRAP_V1,
}
