from __future__ import annotations

from dasbench.families.mis.clique_path_mix_v1 import FAMILY as CLIQUE_PATH_MIX_V1
from dasbench.families.mis.core_fringe_trap_v1 import FAMILY as CORE_FRINGE_TRAP_V1
from dasbench.families.mis.motif_bridge_mixture_v1 import FAMILY as MOTIF_BRIDGE_MIXTURE_V1

FAMILIES = {
    CLIQUE_PATH_MIX_V1.name: CLIQUE_PATH_MIX_V1,
    MOTIF_BRIDGE_MIXTURE_V1.name: MOTIF_BRIDGE_MIXTURE_V1,
    CORE_FRINGE_TRAP_V1.name: CORE_FRINGE_TRAP_V1,
}
