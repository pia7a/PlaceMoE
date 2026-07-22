#ifndef VEOMNI_HIERMOE_REPLICA_PROJECT_TILING_H
#define VEOMNI_HIERMOE_REPLICA_PROJECT_TILING_H

#include <cstdint>

struct HiermoeReplicaProjectTilingData {
    uint32_t numSamples;
    uint32_t topK;
    uint32_t numExperts;
    uint32_t numSlots;
    uint32_t redundantSlotsPerRank;
    uint32_t slotsPerRank;
    uint32_t epSize;
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
    uint32_t maxRecords;
    uint32_t sharedRecordCapacity;
    uint32_t directRawRecordCapacity;
    uint32_t hashCapacity;
    uint32_t distributionSize;
    uint32_t actionMaxRecords;
    uint32_t actionHashCapacity;
    uint32_t actionDistributionSize;
    uint32_t blockCount;
    uint32_t groupOutputStride;
    uint32_t assignmentOutputStride;
    uint32_t baselineIntWorkspaceStride;
    uint32_t baselineFloatWorkspaceStride;
    uint32_t actionIntWorkspaceStride;
    uint32_t actionFloatWorkspaceStride;
    uint32_t directBlockCountStride;
    uint32_t directBlockLoadStride;
    uint64_t actionIntWorkspaceOffset;
    uint64_t actionFloatWorkspaceOffset;
    uint64_t directBlockCountOffset;
    uint64_t directBlockLoadOffset;
    uint64_t sharedSummaryOffset;
    uint64_t sharedSummaryElements;
    int64_t step;
    int64_t layerSeed;
};

#endif
