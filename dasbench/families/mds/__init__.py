from __future__ import annotations

from dasbench.families.mds.gateway_overlap_cover_v1 import FAMILY as GATEWAY_OVERLAP_COVER_V1
from dasbench.families.mds.geometric_cluster_cover_v1 import FAMILY as GEOMETRIC_CLUSTER_COVER_V1
from dasbench.families.mds.star_cluster_cover_v1 import FAMILY as STAR_CLUSTER_COVER_V1

FAMILIES = {
    STAR_CLUSTER_COVER_V1.name: STAR_CLUSTER_COVER_V1,
    GATEWAY_OVERLAP_COVER_V1.name: GATEWAY_OVERLAP_COVER_V1,
    GEOMETRIC_CLUSTER_COVER_V1.name: GEOMETRIC_CLUSTER_COVER_V1,
}
