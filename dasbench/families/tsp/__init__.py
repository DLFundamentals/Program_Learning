from __future__ import annotations

from dasbench.families.tsp.clustered_euclidean_v1 import FAMILY as CLUSTERED_EUCLIDEAN_V1
from dasbench.families.tsp.latent_metric_mixture_v1 import FAMILY as LATENT_METRIC_MIXTURE_V1
from dasbench.families.tsp.paired_ribbon_zigzag_v1 import FAMILY as PAIRED_RIBBON_ZIGZAG_V1

FAMILIES = {
    CLUSTERED_EUCLIDEAN_V1.name: CLUSTERED_EUCLIDEAN_V1,
    PAIRED_RIBBON_ZIGZAG_V1.name: PAIRED_RIBBON_ZIGZAG_V1,
    LATENT_METRIC_MIXTURE_V1.name: LATENT_METRIC_MIXTURE_V1,
}
