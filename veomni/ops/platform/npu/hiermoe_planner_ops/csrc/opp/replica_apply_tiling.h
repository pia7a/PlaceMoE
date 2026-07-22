#ifndef HIERMOE_REPLICA_APPLY_TILING_H
#define HIERMOE_REPLICA_APPLY_TILING_H

#include <cstdint>

struct HiermoeReplicaApplyTilingData {
    uint32_t tokenWidth;
    uint32_t numExperts;
    uint32_t numRoutes;
    uint32_t numTokens;
    uint32_t epSize;
    uint32_t numLevels;
    uint32_t totalGroups;
    uint32_t topK;
    uint32_t levelSize0;
    uint32_t levelSize1;
    uint32_t levelSize2;
    uint32_t levelGroups0;
    uint32_t levelGroups1;
    uint32_t levelGroups2;
    uint32_t levelOffset0;
    uint32_t levelOffset1;
    uint32_t levelOffset2;
};

#endif
