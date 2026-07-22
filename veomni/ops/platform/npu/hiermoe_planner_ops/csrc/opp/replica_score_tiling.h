#ifndef HIERMOE_REPLICA_SCORE_TILING_H
#define HIERMOE_REPLICA_SCORE_TILING_H

#include <cstdint>

struct HiermoeReplicaScoreTilingData {
    uint32_t numCandidateExperts;
    uint32_t tokenWidth;
    uint32_t epSize;
    uint32_t numLevels;
    uint32_t totalGroups;
    uint32_t outputWidth;
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
