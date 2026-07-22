#ifndef VEOMNI_HIERMOE_SWAP_SELECT_TILING_H
#define VEOMNI_HIERMOE_SWAP_SELECT_TILING_H

#include <cstdint>

struct HiermoeSwapSelectTilingData {
    uint32_t numExperts;
    uint32_t numSlots;
    uint32_t maxSwaps;
    uint32_t slotsPerRank;
    uint32_t epSize;
    uint32_t localWorldSize;
    uint32_t numLevels;
    uint32_t levelSize0;
    uint32_t levelSize1;
    uint32_t levelSize2;
    uint32_t levelGroups0;
    uint32_t levelGroups1;
    uint32_t levelGroups2;
    uint32_t levelOffset0;
    uint32_t levelOffset1;
    uint32_t levelOffset2;
    uint32_t totalGroups;
    uint32_t numSamples;
    uint32_t topK;
    uint32_t tokenWidth;
    uint32_t blockCount;
    uint32_t payloadBytes;
    float flatPayloadFactor;
    float interPayloadFactor0;
    float interPayloadFactor1;
    float intraPayloadFactor0;
    float intraPayloadFactor1;
    float communicationScale;
    float computePerAssignment;
    float a2aAlpha;
    float a2aBeta;
    float interAlpha0;
    float interBeta0;
    float interAlpha1;
    float interBeta1;
    float intraAlpha;
    float intraBeta;
    float stateIntraAlpha;
    float stateIntraBeta;
    float stateInterAlpha;
    float stateInterBeta;
    float gatherIntraAlpha;
    float gatherIntraBeta;
    float gatherInterAlpha;
    float gatherInterBeta;
    float scatterIntraAlpha;
    float scatterIntraBeta;
    float scatterInterAlpha;
    float scatterInterBeta;
    float runtimeCostScale;
    uint32_t chooseMinDimension;
};

#endif
