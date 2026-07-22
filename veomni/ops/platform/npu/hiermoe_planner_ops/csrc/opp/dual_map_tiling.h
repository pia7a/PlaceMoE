#ifndef VEOMNI_HIERMOE_DUAL_MAP_TILING_H
#define VEOMNI_HIERMOE_DUAL_MAP_TILING_H

#include <cstdint>

struct HiermoeDualMapTilingData {
    uint32_t numTokens;
    uint32_t tokensPerBlock;
    uint32_t topK;
    uint32_t numExperts;
    uint32_t maxCopies;
    uint32_t slotsPerRank;
    uint32_t sourceRank;
    uint32_t epSize;
    uint32_t numLevels;
    uint32_t levelSize0;
    uint32_t levelSize1;
    int64_t step;
    int64_t layerSeed;
};

#endif
