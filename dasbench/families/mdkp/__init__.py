from __future__ import annotations

from dasbench.families.mdkp.decoy_complement_mixture_v1 import FAMILY as DECOY_COMPLEMENT
from dasbench.families.mdkp.latent_class_knapsack_v1 import FAMILY as LATENT_CLASS
from dasbench.families.mdkp.single_resource_density_v1 import FAMILY as SINGLE_RESOURCE

FAMILIES = {
    SINGLE_RESOURCE.name: SINGLE_RESOURCE,
    LATENT_CLASS.name: LATENT_CLASS,
    DECOY_COMPLEMENT.name: DECOY_COMPLEMENT,
}
