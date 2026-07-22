#ifndef VEOMNI_HIERMOE_QUOTA_POLICY_TILING_H
#define VEOMNI_HIERMOE_QUOTA_POLICY_TILING_H

#include <cstdint>

struct HiermoeQuotaPolicyTilingData {
    uint32_t numSamples;
    uint32_t topK;
    uint32_t epSize;
    uint32_t numExperts;
    uint32_t numSlots;
    uint32_t maxCopies;
    uint32_t maskCount;
    uint32_t samplesPerSource;
    uint32_t rowCapacity;
    uint32_t rowWidth;
    uint32_t slotsPerRank;
    uint32_t sourceRank;
    uint32_t numLevels;
    uint32_t levelSize0;
    uint32_t levelSize1;
    uint32_t blockCount;
};

#endif
