from __future__ import annotations

from dasbench.families.maxsat.community_parity_overlay_v1 import FAMILY as COMMUNITY_PARITY_OVERLAY_V1
from dasbench.families.maxsat.last_clause_signal_v1 import FAMILY as LAST_CLAUSE_SIGNAL_V1
from dasbench.families.maxsat.latent_backdoor_mixture_v1 import FAMILY as LATENT_BACKDOOR_MIXTURE_V1

FAMILIES = {
    LAST_CLAUSE_SIGNAL_V1.name: LAST_CLAUSE_SIGNAL_V1,
    LATENT_BACKDOOR_MIXTURE_V1.name: LATENT_BACKDOOR_MIXTURE_V1,
    COMMUNITY_PARITY_OVERLAY_V1.name: COMMUNITY_PARITY_OVERLAY_V1,
}
