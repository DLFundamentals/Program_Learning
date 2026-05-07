from __future__ import annotations

from dasbench.families.packing_lp.block_coupled_resource_v1 import FAMILY as BLOCK_COUPLED_RESOURCE
from dasbench.families.packing_lp.latent_active_basis_v1 import FAMILY as LATENT_ACTIVE_BASIS
from dasbench.families.packing_lp.single_bottleneck_fractional_v1 import FAMILY as SINGLE_BOTTLENECK

FAMILIES = {
    SINGLE_BOTTLENECK.name: SINGLE_BOTTLENECK,
    LATENT_ACTIVE_BASIS.name: LATENT_ACTIVE_BASIS,
    BLOCK_COUPLED_RESOURCE.name: BLOCK_COUPLED_RESOURCE,
}
