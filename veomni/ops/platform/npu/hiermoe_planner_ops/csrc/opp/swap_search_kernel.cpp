#include "kernel_operator.h"

struct HiermoeSwapSearchTilingData {
    uint32_t numSamples;
    uint32_t tokenWidth;
    uint32_t topK;
    uint32_t numExperts;
    uint32_t numSlots;
    uint32_t maxSwaps;
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
    uint32_t statsWidth;
    uint32_t statsStride;
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
    uint32_t chooseMinDimension;
};

class KernelHiermoeSwapSearch {
public:
    __aicore__ inline void Init(
        GM_ADDR sampleRoutes,
        GM_ADDR sampleWeights,
        GM_ADDR assignmentCounts,
        GM_ADDR slotToLogical,
        GM_ADDR ownerSlots,
        GM_ADDR updatedLayout,
        GM_ADDR updatedOwners,
        GM_ADDR actions,
        GM_ADDR metadata,
        GM_ADDR floatWorkspace,
        GM_ADDR intWorkspace,
        const HiermoeSwapSearchTilingData &tiling)
    {
        this->tiling = tiling;
        sampleRoutesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleRoutes));
        sampleWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sampleWeights));
        assignmentCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(assignmentCounts));
        slotToLogicalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slotToLogical));
        ownerSlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ownerSlots));
        updatedLayoutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(updatedLayout));
        updatedOwnersGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(updatedOwners));
        actionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actions));
        metadataGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(metadata));
        floatWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(floatWorkspace));
        intWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(intWorkspace));

        globalStatsOffset = tiling.blockCount * tiling.statsStride;
        auxOffset = globalStatsOffset + tiling.statsStride;
        assignmentLoadOffset = auxOffset + 16U;
        expertAssignmentOffset = assignmentLoadOffset + tiling.epSize;
        bestCostOffset = Align32(expertAssignmentOffset + tiling.numExperts);
        usedOffset = 0;
        blockExpertCountOffset = Align16(tiling.numExperts);
        blockExpertCountStride = Align16(tiling.numExperts);
        bestPairOffset = blockExpertCountOffset + tiling.blockCount * blockExpertCountStride;
        controlOffset = bestPairOffset + tiling.blockCount * 16U;
        packedRouteOffset = Align16(controlOffset + 16U);
        packedCountOffset = packedRouteOffset
            + tiling.numExperts * tiling.blockCount * tiling.tokenWidth;
    }

    __aicore__ inline void Process()
    {
        InitializeOutputs();
        FlushGlobalWrites();
        AscendC::SyncAll<true>();

        PackRoutes();
        FlushGlobalWrites();
        AscendC::SyncAll<true>();
        BuildStats(true);
        PrepareCurrentCost();
        FlushGlobalWrites();
        AscendC::SyncAll<true>();

        for (uint32_t round = 0; round < tiling.maxSwaps; ++round) {
            ScoreCandidates();
            FlushGlobalWrites();
            AscendC::SyncAll<true>();
            SelectAndApply(round);
            FlushGlobalWrites();
            AscendC::SyncAll<true>();

            const bool accepted = intWorkspaceGm.GetValue(controlOffset + 1U) != 0;
            if (round + 1U < tiling.maxSwaps) {
                if (accepted) {
                    UpdateStatsAfterSwap(round);
                }
                FlushGlobalWrites();
                AscendC::SyncAll<true>();
            }
        }

        if (AscendC::GetBlockIdx() == 0) {
            metadataGm.SetValue(0, static_cast<int32_t>(intWorkspaceGm.GetValue(controlOffset)));
            metadataGm.SetValue(1, static_cast<int32_t>(intWorkspaceGm.GetValue(controlOffset + 2U)));
            metadataGm.SetValue(2, static_cast<int32_t>(intWorkspaceGm.GetValue(controlOffset + 3U)));
            FlushGlobalWrites();
        }
    }

private:
    __aicore__ inline void FlushGlobalWrites()
    {
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
    }

    __aicore__ inline uint32_t Align16(uint32_t value) const
    {
        return (value + 15U) / 16U * 16U;
    }

    __aicore__ inline uint32_t Align32(uint32_t value) const
    {
        return (value + 31U) / 32U * 32U;
    }

    __aicore__ inline uint32_t ExpertTokenOffset() const
    {
        return tiling.totalGroups * 32U;
    }

    __aicore__ inline uint32_t ExpertGroupStride() const
    {
        return Align32(tiling.totalGroups);
    }

    __aicore__ inline uint32_t ExpertGroupOffset() const
    {
        return ExpertTokenOffset() + tiling.numExperts * 32U;
    }

    __aicore__ inline uint32_t SoleExpertOffset() const
    {
        return ExpertGroupOffset() + tiling.numExperts * ExpertGroupStride();
    }

    __aicore__ inline uint32_t SolePairStride() const
    {
        return Align32(tiling.numExperts);
    }

    __aicore__ inline uint32_t SolePairOffset() const
    {
        return SoleExpertOffset() + tiling.numExperts * 32U;
    }

    __aicore__ inline uint32_t LevelSize(uint32_t level) const
    {
        return level == 0 ? tiling.levelSize0 : (level == 1 ? tiling.levelSize1 : tiling.levelSize2);
    }

    __aicore__ inline uint32_t LevelGroups(uint32_t level) const
    {
        return level == 0 ? tiling.levelGroups0 : (level == 1 ? tiling.levelGroups1 : tiling.levelGroups2);
    }

    __aicore__ inline uint32_t LevelOffset(uint32_t level) const
    {
        return level == 0 ? tiling.levelOffset0 : (level == 1 ? tiling.levelOffset1 : tiling.levelOffset2);
    }

    __aicore__ inline void InitializeOutputs()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        for (uint32_t slot = 0; slot < tiling.numSlots; ++slot) {
            updatedLayoutGm.SetValue(slot, slotToLogicalGm.GetValue(slot));
        }
        for (uint32_t expert = 0; expert < tiling.numExperts; ++expert) {
            updatedOwnersGm.SetValue(expert, ownerSlotsGm.GetValue(expert));
            intWorkspaceGm.SetValue(usedOffset + expert, 0);
        }
        for (uint32_t row = 0; row < (tiling.maxSwaps > 0 ? tiling.maxSwaps : 1U); ++row) {
            for (uint32_t column = 0; column < 5; ++column) {
                actionsGm.SetValue(static_cast<uint64_t>(row) * 5U + column, -1);
            }
        }
        for (uint32_t index = 0; index < 8; ++index) {
            metadataGm.SetValue(index, 0);
        }
        intWorkspaceGm.SetValue(controlOffset, 0);
        intWorkspaceGm.SetValue(controlOffset + 1U, 0);
        intWorkspaceGm.SetValue(controlOffset + 2U, 0);
        intWorkspaceGm.SetValue(controlOffset + 3U, 0);
    }

    __aicore__ inline uint32_t UniqueExperts(uint32_t token, int64_t (&experts)[16])
    {
        uint32_t count = 0;
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        for (uint32_t position = 0; position < tiling.topK; ++position) {
            const int64_t logical = sampleRoutesGm.GetValue(routeOffset + position);
            bool seen = false;
            for (uint32_t index = 0; index < count; ++index) {
                seen = seen || experts[index] == logical;
            }
            if (!seen) {
                experts[count++] = logical;
            }
        }
        return count;
    }

    __aicore__ inline void PackRoutes()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        uint32_t localCounts[256];
        for (uint32_t expert = 0; expert < tiling.numExperts; ++expert) {
            localCounts[expert] = 0;
        }
        for (uint32_t token = block; token < tiling.numSamples; token += tiling.blockCount) {
            int64_t experts[16];
            const uint32_t expertCount = UniqueExperts(token, experts);
            for (uint32_t index = 0; index < expertCount; ++index) {
                const uint32_t expert = static_cast<uint32_t>(experts[index]);
                const uint64_t row = packedRouteOffset
                    + (static_cast<uint64_t>(expert) * tiling.blockCount + block) * tiling.tokenWidth;
                intWorkspaceGm.SetValue(row + localCounts[expert], token);
                ++localCounts[expert];
            }
        }
        const uint64_t countRow = blockExpertCountOffset
            + static_cast<uint64_t>(block) * blockExpertCountStride;
        for (uint32_t expert = 0; expert < tiling.numExperts; ++expert) {
            intWorkspaceGm.SetValue(countRow + expert, localCounts[expert]);
        }
    }

    __aicore__ inline void BuildStats(bool initial)
    {
        if (initial) {
            for (uint32_t task = AscendC::GetBlockIdx(); task < tiling.totalGroups; task += tiling.blockCount) {
                uint32_t level = 0;
                while (level + 1U < tiling.numLevels && task >= LevelOffset(level + 1U)) {
                    ++level;
                }
                const uint32_t group = task - LevelOffset(level);
                const uint32_t levelSize = LevelSize(level);
                float count = 0.0F;
                for (uint32_t token = 0; token < tiling.numSamples; ++token) {
                    const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
                    bool hit = false;
                    for (uint32_t position = 0; position < tiling.topK; ++position) {
                        const uint32_t logical = static_cast<uint32_t>(sampleRoutesGm.GetValue(routeOffset + position));
                        const uint32_t rank = static_cast<uint32_t>(updatedOwnersGm.GetValue(logical))
                            / tiling.slotsPerRank;
                        hit = hit || rank / levelSize == group;
                    }
                    if (hit) {
                        count += sampleWeightsGm.GetValue(token);
                    }
                }
                floatWorkspaceGm.SetValue(globalStatsOffset + task * 32U, count);
            }
        }

        for (uint32_t logical = AscendC::GetBlockIdx(); logical < tiling.numExperts; logical += tiling.blockCount) {
            float expertGroups[256];
            float soleExperts[3];
            float solePairs[768];
            for (uint32_t index = 0; index < tiling.totalGroups; ++index) {
                expertGroups[index] = 0.0F;
            }
            for (uint32_t level = 0; level < tiling.numLevels; ++level) {
                soleExperts[level] = 0.0F;
                for (uint32_t other = 0; other < tiling.numExperts; ++other) {
                    solePairs[level * tiling.numExperts + other] = 0.0F;
                }
            }
            float expertTokenCount = 0.0F;
            for (uint32_t segment = 0; segment < tiling.blockCount; ++segment) {
                const uint32_t tokenCount = static_cast<uint32_t>(intWorkspaceGm.GetValue(
                    blockExpertCountOffset
                    + static_cast<uint64_t>(segment) * blockExpertCountStride + logical));
                const uint64_t packedRow = packedRouteOffset
                    + (static_cast<uint64_t>(logical) * tiling.blockCount + segment) * tiling.tokenWidth;
                for (uint32_t item = 0; item < tokenCount; ++item) {
                    const uint32_t token = static_cast<uint32_t>(intWorkspaceGm.GetValue(packedRow + item));
                const float weight = sampleWeightsGm.GetValue(token);
                expertTokenCount += weight;
                int64_t experts[16];
                const uint32_t expertCount = UniqueExperts(token, experts);
                for (uint32_t level = 0; level < tiling.numLevels; ++level) {
                    int32_t groups[16];
                    int32_t groupOccupancy[16];
                    uint32_t groupCount = 0;
                    const uint32_t levelSize = LevelSize(level);
                    const uint32_t levelOffset = LevelOffset(level);
                    for (uint32_t index = 0; index < expertCount; ++index) {
                        const uint32_t other = static_cast<uint32_t>(experts[index]);
                        const uint32_t rank = static_cast<uint32_t>(updatedOwnersGm.GetValue(other))
                            / tiling.slotsPerRank;
                        const int32_t group = static_cast<int32_t>(rank / levelSize);
                        uint32_t groupIndex = groupCount;
                        for (uint32_t previous = 0; previous < groupCount; ++previous) {
                            if (groups[previous] == group) {
                                groupIndex = previous;
                                break;
                            }
                        }
                        if (groupIndex == groupCount) {
                            groups[groupCount] = group;
                            groupOccupancy[groupCount] = 0;
                            ++groupCount;
                        }
                        ++groupOccupancy[groupIndex];
                    }
                    for (uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
                        expertGroups[levelOffset + static_cast<uint32_t>(groups[groupIndex])] += weight;
                    }
                    const uint32_t ownRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(logical))
                        / tiling.slotsPerRank;
                    const int32_t ownGroup = static_cast<int32_t>(ownRank / levelSize);
                    bool sole = false;
                    for (uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
                        sole = sole || (groups[groupIndex] == ownGroup && groupOccupancy[groupIndex] == 1);
                    }
                    if (!sole) {
                        continue;
                    }
                    soleExperts[level] += weight;
                    for (uint32_t other = 0; other < expertCount; ++other) {
                        solePairs[level * tiling.numExperts + static_cast<uint32_t>(experts[other])] += weight;
                    }
                }
            }
            }
            if (initial) {
                floatWorkspaceGm.SetValue(
                    globalStatsOffset + ExpertTokenOffset() + logical * 32U,
                    expertTokenCount);
            }
            const uint32_t expertGroupBase = globalStatsOffset + ExpertGroupOffset()
                + logical * ExpertGroupStride();
            for (uint32_t index = 0; index < tiling.totalGroups; ++index) {
                floatWorkspaceGm.SetValue(expertGroupBase + index, expertGroups[index]);
            }
            for (uint32_t level = 0; level < tiling.numLevels; ++level) {
                floatWorkspaceGm.SetValue(
                    globalStatsOffset + SoleExpertOffset() + logical * 32U + level,
                    soleExperts[level]);
                const uint32_t pairBase = globalStatsOffset + SolePairOffset()
                    + (logical * tiling.numLevels + level) * SolePairStride();
                for (uint32_t other = 0; other < tiling.numExperts; ++other) {
                    floatWorkspaceGm.SetValue(pairBase + other, solePairs[level * tiling.numExperts + other]);
                }
            }
        }
        FlushGlobalWrites();
        AscendC::SyncAll<true>();
    }

    __aicore__ inline uint32_t RankBeforeSwap(
        uint32_t logical,
        uint32_t lhs,
        uint32_t rhs,
        uint32_t lhsRank,
        uint32_t rhsRank) const
    {
        if (logical == lhs) {
            return lhsRank;
        }
        if (logical == rhs) {
            return rhsRank;
        }
        return static_cast<uint32_t>(updatedOwnersGm.GetValue(logical)) / tiling.slotsPerRank;
    }

    __aicore__ inline void UpdateStatsAfterSwap(uint32_t round)
    {
        const uint64_t actionOffset = static_cast<uint64_t>(round) * 5U;
        const uint32_t lhs = static_cast<uint32_t>(actionsGm.GetValue(actionOffset));
        const uint32_t rhs = static_cast<uint32_t>(actionsGm.GetValue(actionOffset + 1U));
        const uint32_t lhsRank = static_cast<uint32_t>(actionsGm.GetValue(actionOffset + 2U))
            / tiling.slotsPerRank;
        const uint32_t rhsRank = static_cast<uint32_t>(actionsGm.GetValue(actionOffset + 3U))
            / tiling.slotsPerRank;

        for (uint32_t logical = AscendC::GetBlockIdx(); logical < tiling.numExperts; logical += tiling.blockCount) {
            for (uint32_t segment = 0; segment < tiling.blockCount; ++segment) {
                const uint32_t count = static_cast<uint32_t>(intWorkspaceGm.GetValue(
                    blockExpertCountOffset
                    + static_cast<uint64_t>(segment) * blockExpertCountStride + logical));
                const uint64_t row = packedRouteOffset
                    + (static_cast<uint64_t>(logical) * tiling.blockCount + segment) * tiling.tokenWidth;
                for (uint32_t item = 0; item < count; ++item) {
                    const uint32_t token = static_cast<uint32_t>(intWorkspaceGm.GetValue(row + item));
                int64_t experts[16];
                const uint32_t expertCount = UniqueExperts(token, experts);
                bool containsLhs = false;
                bool containsRhs = false;
                for (uint32_t index = 0; index < expertCount; ++index) {
                    const uint32_t expert = static_cast<uint32_t>(experts[index]);
                    containsLhs = containsLhs || expert == lhs;
                    containsRhs = containsRhs || expert == rhs;
                }
                if (!containsLhs && !containsRhs) {
                    continue;
                }

                const float weight = sampleWeightsGm.GetValue(token);
                for (uint32_t level = 0; level < tiling.numLevels; ++level) {
                    const uint32_t levelSize = LevelSize(level);
                    const uint32_t lhsGroup = lhsRank / levelSize;
                    const uint32_t rhsGroup = rhsRank / levelSize;
                    if (lhsGroup == rhsGroup) {
                        continue;
                    }

                    uint32_t lhsOccupancy = 0;
                    uint32_t rhsOccupancy = 0;
                    for (uint32_t index = 0; index < expertCount; ++index) {
                        const uint32_t expert = static_cast<uint32_t>(experts[index]);
                        const uint32_t oldRank = RankBeforeSwap(expert, lhs, rhs, lhsRank, rhsRank);
                        const uint32_t oldGroup = oldRank / levelSize;
                        lhsOccupancy += oldGroup == lhsGroup;
                        rhsOccupancy += oldGroup == rhsGroup;
                    }
                    const uint32_t newLhsOccupancy = lhsOccupancy
                        - static_cast<uint32_t>(containsLhs) + static_cast<uint32_t>(containsRhs);
                    const uint32_t newRhsOccupancy = rhsOccupancy
                        - static_cast<uint32_t>(containsRhs) + static_cast<uint32_t>(containsLhs);
                    const uint32_t expertGroupBase = globalStatsOffset + ExpertGroupOffset()
                        + logical * ExpertGroupStride() + LevelOffset(level);
                    if ((lhsOccupancy == 0U) != (newLhsOccupancy == 0U)) {
                        const uint32_t offset = expertGroupBase + lhsGroup;
                        const float delta = newLhsOccupancy == 0U ? -weight : weight;
                        floatWorkspaceGm.SetValue(offset, floatWorkspaceGm.GetValue(offset) + delta);
                    }
                    if ((rhsOccupancy == 0U) != (newRhsOccupancy == 0U)) {
                        const uint32_t offset = expertGroupBase + rhsGroup;
                        const float delta = newRhsOccupancy == 0U ? -weight : weight;
                        floatWorkspaceGm.SetValue(offset, floatWorkspaceGm.GetValue(offset) + delta);
                    }

                    const uint32_t logicalOldRank = RankBeforeSwap(logical, lhs, rhs, lhsRank, rhsRank);
                    const uint32_t logicalNewRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(logical))
                        / tiling.slotsPerRank;
                    const uint32_t logicalOldGroup = logicalOldRank / levelSize;
                    const uint32_t logicalNewGroup = logicalNewRank / levelSize;
                    uint32_t oldOwnOccupancy = 2U;
                    uint32_t newOwnOccupancy = 2U;
                    if (logicalOldGroup == lhsGroup) {
                        oldOwnOccupancy = lhsOccupancy;
                    } else if (logicalOldGroup == rhsGroup) {
                        oldOwnOccupancy = rhsOccupancy;
                    }
                    if (logicalNewGroup == lhsGroup) {
                        newOwnOccupancy = newLhsOccupancy;
                    } else if (logicalNewGroup == rhsGroup) {
                        newOwnOccupancy = newRhsOccupancy;
                    }
                    const bool oldSole = oldOwnOccupancy == 1U;
                    const bool newSole = newOwnOccupancy == 1U;
                    if (oldSole == newSole) {
                        continue;
                    }
                    const float delta = newSole ? weight : -weight;
                    const uint32_t soleOffset = globalStatsOffset + SoleExpertOffset()
                        + logical * 32U + level;
                    floatWorkspaceGm.SetValue(
                        soleOffset, floatWorkspaceGm.GetValue(soleOffset) + delta);
                    const uint32_t pairBase = globalStatsOffset + SolePairOffset()
                        + (logical * tiling.numLevels + level) * SolePairStride();
                    for (uint32_t index = 0; index < expertCount; ++index) {
                        const uint32_t pairOffset = pairBase + static_cast<uint32_t>(experts[index]);
                        floatWorkspaceGm.SetValue(
                            pairOffset, floatWorkspaceGm.GetValue(pairOffset) + delta);
                    }
                }
            }
            }
        }
    }

    __aicore__ inline float GroupValue(
        uint32_t level,
        uint32_t group,
        int32_t lhs,
        int32_t rhs) const
    {
        float value = floatWorkspaceGm.GetValue(
            globalStatsOffset + (LevelOffset(level) + group) * 32U);
        if (lhs < 0 || rhs < 0) {
            return value;
        }
        const uint32_t lhsLogical = static_cast<uint32_t>(lhs);
        const uint32_t rhsLogical = static_cast<uint32_t>(rhs);
        const uint32_t levelSize = LevelSize(level);
        const uint32_t lhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(lhsLogical)) / tiling.slotsPerRank;
        const uint32_t rhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(rhsLogical)) / tiling.slotsPerRank;
        const uint32_t lhsGroup = lhsRank / levelSize;
        const uint32_t rhsGroup = rhsRank / levelSize;
        if (lhsGroup == rhsGroup) {
            return value;
        }
        const float lhsCount = floatWorkspaceGm.GetValue(
            globalStatsOffset + ExpertTokenOffset() + lhsLogical * 32U);
        const float rhsCount = floatWorkspaceGm.GetValue(
            globalStatsOffset + ExpertTokenOffset() + rhsLogical * 32U);
        const uint32_t expertGroup = ExpertGroupOffset();
        const uint32_t soleExpert = SoleExpertOffset();
        const uint32_t solePair = SolePairOffset();
        if (group == lhsGroup) {
            value += rhsCount
                - floatWorkspaceGm.GetValue(
                    globalStatsOffset + expertGroup + rhsLogical * ExpertGroupStride()
                    + LevelOffset(level) + lhsGroup)
                - floatWorkspaceGm.GetValue(globalStatsOffset + soleExpert + lhsLogical * 32U + level)
                + floatWorkspaceGm.GetValue(
                    globalStatsOffset + solePair
                    + (lhsLogical * tiling.numLevels + level) * SolePairStride() + rhsLogical);
        }
        if (group == rhsGroup) {
            value += lhsCount
                - floatWorkspaceGm.GetValue(
                    globalStatsOffset + expertGroup + lhsLogical * ExpertGroupStride()
                    + LevelOffset(level) + rhsGroup)
                - floatWorkspaceGm.GetValue(globalStatsOffset + soleExpert + rhsLogical * 32U + level)
                + floatWorkspaceGm.GetValue(
                    globalStatsOffset + solePair
                    + (rhsLogical * tiling.numLevels + level) * SolePairStride() + lhsLogical);
        }
        return value;
    }

    __aicore__ inline float CandidateCost(int32_t lhs, int32_t rhs, int32_t &commRank, int32_t &computeRank)
    {
        float levelMax[3] = {0.0F, 0.0F, 0.0F};
        for (uint32_t level = 0; level < tiling.numLevels; ++level) {
            const uint32_t groups = LevelGroups(level);
            for (uint32_t group = 0; group < groups; ++group) {
                const float value = GroupValue(level, group, lhs, rhs);
                if (value > levelMax[level]) {
                    levelMax[level] = value;
                }
            }
        }

        const uint32_t rankLevel = tiling.numLevels - 1U;
        float rankMaximum = -1.0F;
        commRank = 0;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            const float value = GroupValue(rankLevel, rank, lhs, rhs);
            if (value > rankMaximum) {
                rankMaximum = value;
                commRank = static_cast<int32_t>(rank);
            }
        }

        float oneWay = tiling.a2aAlpha
            + tiling.flatPayloadFactor * rankMaximum * tiling.a2aBeta;
        if (tiling.numLevels > 1U) {
            float running = 0.0F;
            float selected = oneWay;
            for (uint32_t level = 0; level + 1U < tiling.numLevels; ++level) {
                const float interAlpha = level == 0 ? tiling.interAlpha0 : tiling.interAlpha1;
                const float interBeta = level == 0 ? tiling.interBeta0 : tiling.interBeta1;
                const float interFactor = level == 0 ? tiling.interPayloadFactor0 : tiling.interPayloadFactor1;
                const float intraFactor = level == 0 ? tiling.intraPayloadFactor0 : tiling.intraPayloadFactor1;
                running += interAlpha + interFactor * levelMax[level] * interBeta;
                const float hierarchical = running + tiling.intraAlpha
                    + intraFactor * rankMaximum * tiling.intraBeta;
                if (tiling.chooseMinDimension != 0U) {
                    if (hierarchical < selected) {
                        selected = hierarchical;
                    }
                } else if (level == 0U) {
                    selected = hierarchical;
                }
            }
            oneWay = selected;
        }

        float peakAssignment = -1.0F;
        computeRank = 0;
        uint32_t lhsRank = 0;
        uint32_t rhsRank = 0;
        float lhsAssignments = 0.0F;
        float rhsAssignments = 0.0F;
        if (lhs >= 0 && rhs >= 0) {
            lhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(static_cast<uint32_t>(lhs))) / tiling.slotsPerRank;
            rhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(static_cast<uint32_t>(rhs))) / tiling.slotsPerRank;
            lhsAssignments = floatWorkspaceGm.GetValue(expertAssignmentOffset + static_cast<uint32_t>(lhs));
            rhsAssignments = floatWorkspaceGm.GetValue(expertAssignmentOffset + static_cast<uint32_t>(rhs));
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            float value = floatWorkspaceGm.GetValue(assignmentLoadOffset + rank);
            if (lhs >= 0 && rhs >= 0 && lhsRank != rhsRank) {
                if (rank == lhsRank) {
                    value += rhsAssignments - lhsAssignments;
                } else if (rank == rhsRank) {
                    value += lhsAssignments - rhsAssignments;
                }
            }
            if (value > peakAssignment) {
                peakAssignment = value;
                computeRank = static_cast<int32_t>(rank);
            }
        }
        return 4.0F * oneWay * tiling.communicationScale
            + tiling.computePerAssignment * peakAssignment;
    }

    __aicore__ inline void PrepareCurrentCost()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            floatWorkspaceGm.SetValue(assignmentLoadOffset + rank, 0.0F);
        }
        for (uint32_t expert = 0; expert < tiling.numExperts; ++expert) {
            int64_t total = 0;
            for (uint32_t source = 0; source < tiling.epSize; ++source) {
                total += assignmentCountsGm.GetValue(static_cast<uint64_t>(source) * tiling.numExperts + expert);
            }
            const float value = static_cast<float>(total);
            floatWorkspaceGm.SetValue(expertAssignmentOffset + expert, value);
            const uint32_t rank = static_cast<uint32_t>(updatedOwnersGm.GetValue(expert)) / tiling.slotsPerRank;
            floatWorkspaceGm.SetValue(
                assignmentLoadOffset + rank,
                floatWorkspaceGm.GetValue(assignmentLoadOffset + rank) + value);
        }
        RefreshCurrentCost();
    }

    __aicore__ inline void RefreshCurrentCost()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        int32_t commRank = 0;
        int32_t computeRank = 0;
        const float cost = CandidateCost(-1, -1, commRank, computeRank);
        floatWorkspaceGm.SetValue(auxOffset, cost);
        intWorkspaceGm.SetValue(controlOffset + 2U, commRank);
        intWorkspaceGm.SetValue(controlOffset + 3U, computeRank);
    }

    __aicore__ inline void ScoreCandidates()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        float bestCost = 3.402823466e38F;
        int64_t bestLhs = -1;
        int64_t bestRhs = -1;
        const int64_t commRank = intWorkspaceGm.GetValue(controlOffset + 2U);
        const int64_t computeRank = intWorkspaceGm.GetValue(controlOffset + 3U);
        uint32_t pairIndex = 0;
        for (uint32_t lhs = 0; lhs < tiling.numExperts; ++lhs) {
            const uint32_t lhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(lhs)) / tiling.slotsPerRank;
            const bool lhsUsed = intWorkspaceGm.GetValue(usedOffset + lhs) != 0;
            const bool lhsHot = lhsRank == static_cast<uint32_t>(commRank) || lhsRank == static_cast<uint32_t>(computeRank);
            for (uint32_t rhs = lhs + 1U; rhs < tiling.numExperts; ++rhs, ++pairIndex) {
                if (pairIndex % tiling.blockCount != block) {
                    continue;
                }
                const uint32_t rhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(rhs)) / tiling.slotsPerRank;
                const bool rhsUsed = intWorkspaceGm.GetValue(usedOffset + rhs) != 0;
                const bool rhsHot = rhsRank == static_cast<uint32_t>(commRank)
                    || rhsRank == static_cast<uint32_t>(computeRank);
                if (lhsUsed || rhsUsed || lhsRank == rhsRank || lhsHot == rhsHot) {
                    continue;
                }
                int32_t candidateCommRank = 0;
                int32_t candidateComputeRank = 0;
                const float cost = CandidateCost(
                    static_cast<int32_t>(lhs), static_cast<int32_t>(rhs), candidateCommRank, candidateComputeRank);
                if (cost < bestCost
                    || (cost == bestCost
                        && (bestLhs < 0 || lhs < static_cast<uint32_t>(bestLhs)
                            || (lhs == static_cast<uint32_t>(bestLhs) && rhs < static_cast<uint32_t>(bestRhs))))) {
                    bestCost = cost;
                    bestLhs = lhs;
                    bestRhs = rhs;
                }
            }
        }
        const uint64_t floatRow = bestCostOffset + static_cast<uint64_t>(block) * 32U;
        const uint64_t intRow = bestPairOffset + static_cast<uint64_t>(block) * 16U;
        floatWorkspaceGm.SetValue(floatRow, bestCost);
        intWorkspaceGm.SetValue(intRow, bestLhs);
        intWorkspaceGm.SetValue(intRow + 1U, bestRhs);
    }

    __aicore__ inline void SelectAndApply(uint32_t round)
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        float bestCost = 3.402823466e38F;
        int64_t bestLhs = -1;
        int64_t bestRhs = -1;
        for (uint32_t block = 0; block < tiling.blockCount; ++block) {
            const float cost = floatWorkspaceGm.GetValue(bestCostOffset + static_cast<uint64_t>(block) * 32U);
            const int64_t lhs = intWorkspaceGm.GetValue(bestPairOffset + static_cast<uint64_t>(block) * 16U);
            const int64_t rhs = intWorkspaceGm.GetValue(bestPairOffset + static_cast<uint64_t>(block) * 16U + 1U);
            if (lhs < 0 || rhs < 0) {
                continue;
            }
            if (cost < bestCost || (cost == bestCost && (lhs < bestLhs || (lhs == bestLhs && rhs < bestRhs)))) {
                bestCost = cost;
                bestLhs = lhs;
                bestRhs = rhs;
            }
        }
        const float currentCost = floatWorkspaceGm.GetValue(auxOffset);
        const bool active = intWorkspaceGm.GetValue(controlOffset) == static_cast<int64_t>(round);
        const bool accepted = active && bestLhs >= 0 && bestCost < currentCost;
        intWorkspaceGm.SetValue(controlOffset + 1U, accepted ? 1 : 0);
        if (!accepted) {
            return;
        }

        int32_t candidateCommRank = 0;
        int32_t candidateComputeRank = 0;
        bestCost = CandidateCost(
            static_cast<int32_t>(bestLhs),
            static_cast<int32_t>(bestRhs),
            candidateCommRank,
            candidateComputeRank);
        floatWorkspaceGm.SetValue(auxOffset, bestCost);
        intWorkspaceGm.SetValue(controlOffset + 2U, candidateCommRank);
        intWorkspaceGm.SetValue(controlOffset + 3U, candidateComputeRank);

        const int64_t lhsSlot = updatedOwnersGm.GetValue(static_cast<uint32_t>(bestLhs));
        const int64_t rhsSlot = updatedOwnersGm.GetValue(static_cast<uint32_t>(bestRhs));
        const uint32_t lhsRank = static_cast<uint32_t>(lhsSlot) / tiling.slotsPerRank;
        const uint32_t rhsRank = static_cast<uint32_t>(rhsSlot) / tiling.slotsPerRank;
        const float lhsAssignments = floatWorkspaceGm.GetValue(
            expertAssignmentOffset + static_cast<uint32_t>(bestLhs));
        const float rhsAssignments = floatWorkspaceGm.GetValue(
            expertAssignmentOffset + static_cast<uint32_t>(bestRhs));
        floatWorkspaceGm.SetValue(
            assignmentLoadOffset + lhsRank,
            floatWorkspaceGm.GetValue(assignmentLoadOffset + lhsRank) + rhsAssignments - lhsAssignments);
        floatWorkspaceGm.SetValue(
            assignmentLoadOffset + rhsRank,
            floatWorkspaceGm.GetValue(assignmentLoadOffset + rhsRank) + lhsAssignments - rhsAssignments);
        for (uint32_t level = 0; level < tiling.numLevels; ++level) {
            const uint32_t lhsGroup = lhsRank / LevelSize(level);
            const uint32_t rhsGroup = rhsRank / LevelSize(level);
            if (lhsGroup == rhsGroup) {
                continue;
            }
            const float lhsValue = GroupValue(
                level, lhsGroup, static_cast<int32_t>(bestLhs), static_cast<int32_t>(bestRhs));
            const float rhsValue = GroupValue(
                level, rhsGroup, static_cast<int32_t>(bestLhs), static_cast<int32_t>(bestRhs));
            floatWorkspaceGm.SetValue(
                globalStatsOffset + (LevelOffset(level) + lhsGroup) * 32U,
                lhsValue);
            floatWorkspaceGm.SetValue(
                globalStatsOffset + (LevelOffset(level) + rhsGroup) * 32U,
                rhsValue);
        }
        updatedOwnersGm.SetValue(static_cast<uint32_t>(bestLhs), rhsSlot);
        updatedOwnersGm.SetValue(static_cast<uint32_t>(bestRhs), lhsSlot);
        const int64_t lhsValue = updatedLayoutGm.GetValue(static_cast<uint32_t>(lhsSlot));
        const int64_t rhsValue = updatedLayoutGm.GetValue(static_cast<uint32_t>(rhsSlot));
        updatedLayoutGm.SetValue(static_cast<uint32_t>(lhsSlot), rhsValue);
        updatedLayoutGm.SetValue(static_cast<uint32_t>(rhsSlot), lhsValue);
        intWorkspaceGm.SetValue(usedOffset + static_cast<uint32_t>(bestLhs), 1);
        intWorkspaceGm.SetValue(usedOffset + static_cast<uint32_t>(bestRhs), 1);

        const uint64_t actionOffset = static_cast<uint64_t>(round) * 5U;
        actionsGm.SetValue(actionOffset, bestLhs);
        actionsGm.SetValue(actionOffset + 1U, bestRhs);
        actionsGm.SetValue(actionOffset + 2U, lhsSlot);
        actionsGm.SetValue(actionOffset + 3U, rhsSlot);
        actionsGm.SetValue(actionOffset + 4U, 1);
        intWorkspaceGm.SetValue(controlOffset, static_cast<int64_t>(round + 1U));
    }

    AscendC::GlobalTensor<int64_t> sampleRoutesGm;
    AscendC::GlobalTensor<float> sampleWeightsGm;
    AscendC::GlobalTensor<int64_t> assignmentCountsGm;
    AscendC::GlobalTensor<int64_t> slotToLogicalGm;
    AscendC::GlobalTensor<int64_t> ownerSlotsGm;
    AscendC::GlobalTensor<int64_t> updatedLayoutGm;
    AscendC::GlobalTensor<int64_t> updatedOwnersGm;
    AscendC::GlobalTensor<int64_t> actionsGm;
    AscendC::GlobalTensor<int32_t> metadataGm;
    AscendC::GlobalTensor<float> floatWorkspaceGm;
    AscendC::GlobalTensor<int64_t> intWorkspaceGm;
    HiermoeSwapSearchTilingData tiling;
    uint32_t globalStatsOffset = 0;
    uint32_t auxOffset = 0;
    uint32_t assignmentLoadOffset = 0;
    uint32_t expertAssignmentOffset = 0;
    uint32_t bestCostOffset = 0;
    uint32_t usedOffset = 0;
    uint32_t blockExpertCountOffset = 0;
    uint32_t blockExpertCountStride = 0;
    uint32_t bestPairOffset = 0;
    uint32_t controlOffset = 0;
    uint32_t packedRouteOffset = 0;
    uint32_t packedCountOffset = 0;
};

extern "C" __global__ __aicore__ void hiermoe_swap_search(
    GM_ADDR sampleRoutes,
    GM_ADDR sampleWeights,
    GM_ADDR assignmentCounts,
    GM_ADDR slotToLogical,
    GM_ADDR ownerSlots,
    GM_ADDR updatedLayout,
    GM_ADDR updatedOwners,
    GM_ADDR actions,
    GM_ADDR metadata,
    GM_ADDR floatWorkspace,
    GM_ADDR intWorkspace,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeSwapSearchTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeSwapSearch op;
    op.Init(
        sampleRoutes,
        sampleWeights,
        assignmentCounts,
        slotToLogical,
        ownerSlots,
        updatedLayout,
        updatedOwners,
        actions,
        metadata,
        floatWorkspace,
        intWorkspace,
        tilingData);
    op.Process();
}
