#ifndef VEOMNI_HIERMOE_QUOTA_MAP_TILING_H
#define VEOMNI_HIERMOE_QUOTA_MAP_TILING_H

#include <cstdint>

struct HiermoeQuotaMapTilingData {
    uint32_t numTokens;
    uint32_t tokensPerBlock;
    uint32_t runCapacity;
    uint32_t sortStride;
    uint32_t topK;
    uint32_t numExperts;
    uint32_t maxCopies;
    uint32_t maskCount;
    uint32_t bucketStride;
    uint32_t slotsPerRank;
    uint32_t sourceRank;
    uint32_t epSize;
    uint32_t numLevels;
    uint32_t levelSize0;
    uint32_t levelSize1;
    uint32_t groupWidth;
    uint32_t rankStride;
    uint32_t statsStride;
    uint32_t blockCount;
    int64_t step;
    int64_t layerSeed;
};

#endif
