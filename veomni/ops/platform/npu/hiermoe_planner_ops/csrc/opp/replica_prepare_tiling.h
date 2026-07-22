#ifndef HIERMOE_REPLICA_PREPARE_TILING_H
#define HIERMOE_REPLICA_PREPARE_TILING_H

#include <cstdint>

struct HiermoeReplicaPrepareTilingData {
    uint32_t numTokens;
    uint32_t tokenWidth;
    uint32_t topK;
    uint32_t numExperts;
};

#endif
