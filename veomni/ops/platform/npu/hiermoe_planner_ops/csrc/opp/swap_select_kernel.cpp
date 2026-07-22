#include "kernel_operator.h"

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

class KernelHiermoeSwapSelect {
public:
    __aicore__ inline void Init(
        GM_ADDR expertTokenCounts,
        GM_ADDR expertAssignmentCounts,
        GM_ADDR baseCounts,
        GM_ADDR expertGroupCounts,
        GM_ADDR soleExpertCounts,
        GM_ADDR solePairCounts,
        GM_ADDR sampleRoutes,
        GM_ADDR sampleWeights,
        GM_ADDR slotToLogical,
        GM_ADDR ownerSlots,
        GM_ADDR expertStateBytes,
        GM_ADDR expertGradientBytes,
        GM_ADDR updatedLayout,
        GM_ADDR updatedOwners,
        GM_ADDR actions,
        GM_ADDR metadata,
        GM_ADDR floatWorkspace,
        GM_ADDR intWorkspace,
        const HiermoeSwapSelectTilingData &tiling)
    {
        this->tiling = tiling;
        expertTokenCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expertTokenCounts));
        expertAssignmentCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expertAssignmentCounts));
        baseCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(baseCounts));
        expertGroupCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expertGroupCounts));
        soleExpertCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(soleExpertCounts));
        solePairCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(solePairCounts));
        sampleRoutesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleRoutes));
        sampleWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sampleWeights));
        slotToLogicalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slotToLogical));
        ownerSlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ownerSlots));
        expertStateBytesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(expertStateBytes));
        expertGradientBytesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(expertGradientBytes));
        updatedLayoutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(updatedLayout));
        updatedOwnersGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(updatedOwners));
        actionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actions));
        metadataGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(metadata));
        floatWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(floatWorkspace));
        intWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(intWorkspace));

        workingBaseOffset = 0;
        assignmentLoadOffset = Align16(workingBaseOffset + tiling.totalGroups);
        statePayloadOffset = Align16(assignmentLoadOffset + tiling.epSize);
        stateRankCostOffset = Align16(statePayloadOffset + tiling.epSize * tiling.epSize);
        gradientPayloadOffset = Align16(stateRankCostOffset + tiling.epSize);
        gradientRankCostOffset = Align16(gradientPayloadOffset + tiling.epSize * tiling.epSize);
        bestCostOffset = Align16(gradientRankCostOffset + tiling.epSize);
        currentCostOffset = bestCostOffset + tiling.blockCount * 16U;
        expertGroupDeltaStride = Align16(tiling.totalGroups);
        soleExpertDeltaStride = Align16(tiling.numLevels);
        solePairDeltaStride = Align16(tiling.numLevels * tiling.numExperts);
        expertGroupDeltaOffset = Align16(currentCostOffset + 16U);
        soleExpertDeltaOffset = Align16(
            expertGroupDeltaOffset + tiling.numExperts * expertGroupDeltaStride);
        solePairDeltaOffset = Align16(
            soleExpertDeltaOffset + tiling.numExperts * soleExpertDeltaStride);
        privateDeltaSize = Align16(
            solePairDeltaOffset + tiling.numExperts * solePairDeltaStride - expertGroupDeltaOffset);
        privateDeltaOffset = Align16(
            solePairDeltaOffset + tiling.numExperts * solePairDeltaStride);
        updateBlockCount = tiling.blockCount < MAX_UPDATE_BLOCKS ? tiling.blockCount : MAX_UPDATE_BLOCKS;
        usedOffset = 0;
        bestPairOffset = Align16(tiling.numExperts);
        controlOffset = bestPairOffset + tiling.blockCount * 16U;
        membershipOffset = Align16(controlOffset + 16U);
        pipe.InitBuffer(reduceLeftBuffer, REDUCE_TILE * sizeof(float));
        pipe.InitBuffer(reduceRightBuffer, REDUCE_TILE * sizeof(float));
        pipe.InitBuffer(privateDeltaBuffer, LOCAL_DELTA_CAPACITY * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t scoreCycles = 0;
        int64_t selectCycles = 0;
        int64_t buildCycles = 0;
        int64_t reduceCycles = 0;
        int64_t recomputeCycles = 0;
        int64_t stageStart = 0;
        Initialize();
        if (tiling.maxSwaps > 1U) {
            InitializeMutableStats();
            if (tiling.numExperts <= MEMBERSHIP_EXPERT_CAPACITY) {
                InitializeMembership();
            }
        }
        Flush();
        AscendC::SyncAll<true>();
        for (uint32_t round = 0; round < tiling.maxSwaps; ++round) {
            if (AscendC::GetBlockIdx() == 0) {
                stageStart = AscendC::GetSystemCycle();
            }
            ScoreCandidates(round);
            Flush();
            AscendC::SyncAll<true>();
            if (AscendC::GetBlockIdx() == 0) {
                scoreCycles += AscendC::GetSystemCycle() - stageStart;
                stageStart = AscendC::GetSystemCycle();
            }
            SelectAndApply(round);
            Flush();
            AscendC::SyncAll<true>();
            if (AscendC::GetBlockIdx() == 0) {
                selectCycles += AscendC::GetSystemCycle() - stageStart;
            }
            if (round + 1U < tiling.maxSwaps) {
                const bool accepted = intWorkspaceGm.GetValue(controlOffset + 1U) != 0;
                if (accepted && AscendC::GetBlockIdx() == 0) {
                    stageStart = AscendC::GetSystemCycle();
                }
                if (accepted) {
                    BuildPrivateDeltas(round);
                }
                Flush();
                AscendC::SyncAll<true>();
                if (accepted && AscendC::GetBlockIdx() == 0) {
                    buildCycles += AscendC::GetSystemCycle() - stageStart;
                    stageStart = AscendC::GetSystemCycle();
                }
                if (accepted) {
                    ReducePrivateDeltas();
                }
                Flush();
                AscendC::SyncAll<true>();
                if (accepted && AscendC::GetBlockIdx() == 0) {
                    reduceCycles += AscendC::GetSystemCycle() - stageStart;
                    stageStart = AscendC::GetSystemCycle();
                }
                if (accepted && AscendC::GetBlockIdx() == 0) {
                    int32_t commRank = 0;
                    int32_t computeRank = 0;
                    const float current = CandidateCost(-1, -1, commRank, computeRank);
                    floatWorkspaceGm.SetValue(currentCostOffset, current);
                    intWorkspaceGm.SetValue(controlOffset + 2U, commRank);
                    intWorkspaceGm.SetValue(controlOffset + 3U, computeRank);
                }
                Flush();
                AscendC::SyncAll<true>();
                if (accepted && AscendC::GetBlockIdx() == 0) {
                    recomputeCycles += AscendC::GetSystemCycle() - stageStart;
                }
            }
        }
        if (AscendC::GetBlockIdx() == 0) {
            metadataGm.SetValue(0, static_cast<int32_t>(intWorkspaceGm.GetValue(controlOffset)));
            metadataGm.SetValue(1, static_cast<int32_t>(intWorkspaceGm.GetValue(controlOffset + 2U)));
            metadataGm.SetValue(2, static_cast<int32_t>(intWorkspaceGm.GetValue(controlOffset + 3U)));
            metadataGm.SetValue(3, static_cast<int32_t>(scoreCycles));
            metadataGm.SetValue(4, static_cast<int32_t>(selectCycles));
            metadataGm.SetValue(5, static_cast<int32_t>(buildCycles));
            metadataGm.SetValue(6, static_cast<int32_t>(reduceCycles));
            metadataGm.SetValue(7, static_cast<int32_t>(recomputeCycles));
            Flush();
        }
    }

private:
    static constexpr uint32_t REDUCE_TILE = 512U;
    static constexpr uint32_t MAX_UPDATE_BLOCKS = 64U;
    static constexpr uint32_t LOCAL_DELTA_CAPACITY = 40U * 1024U;
    static constexpr uint32_t MEMBERSHIP_EXPERT_CAPACITY = 256U;
    static constexpr uint32_t MEMBERSHIP_WORDS = 4U;
    static constexpr uint32_t MEMBERSHIP_STRIDE = 8U;
    static constexpr uint32_t FAST_LEVEL_CACHE_SIZE = 9U;
    static constexpr uint32_t FAST_ASSIGNMENT_CACHE = FAST_LEVEL_CACHE_SIZE;
    static constexpr uint32_t FAST_STATE_CACHE = FAST_ASSIGNMENT_CACHE + 3U;
    static constexpr uint32_t FAST_CACHE_SIZE = FAST_STATE_CACHE + 3U;

    __aicore__ inline uint32_t Align16(uint32_t value) const
    {
        return (value + 15U) / 16U * 16U;
    }

    __aicore__ inline uint32_t FastValueOffset(uint32_t slot) const
    {
        return bestCostOffset + 1U + slot;
    }

    __aicore__ inline uint32_t FastIndexOffset(uint32_t slot) const
    {
        return slot < 14U ? bestPairOffset + 2U + slot : controlOffset + 5U;
    }

    __aicore__ inline void BuildTop3(uint32_t sourceOffset, uint32_t count, uint32_t cacheOffset)
    {
        float values[3] = {-3.402823466e38F, -3.402823466e38F, -3.402823466e38F};
        int64_t indices[3] = {-1, -1, -1};
        for (uint32_t index = 0; index < count; ++index) {
            const float value = floatWorkspaceGm.GetValue(sourceOffset + index);
            uint32_t destination = 3U;
            for (uint32_t position = 0; position < 3U; ++position) {
                if (value > values[position]
                    || (value == values[position]
                        && (indices[position] < 0 || index < static_cast<uint32_t>(indices[position])))) {
                    destination = position;
                    break;
                }
            }
            if (destination == 3U) {
                continue;
            }
            for (uint32_t position = 2U; position > destination; --position) {
                values[position] = values[position - 1U];
                indices[position] = indices[position - 1U];
            }
            values[destination] = value;
            indices[destination] = static_cast<int64_t>(index);
        }
        for (uint32_t position = 0; position < 3U; ++position) {
            floatWorkspaceGm.SetValue(FastValueOffset(cacheOffset + position), values[position]);
            intWorkspaceGm.SetValue(FastIndexOffset(cacheOffset + position), indices[position]);
        }
    }

    __aicore__ inline void RefreshFastCostCache()
    {
        for (uint32_t level = 0; level < tiling.numLevels; ++level) {
            BuildTop3(
                workingBaseOffset + LevelOffset(level),
                LevelGroups(level),
                level * 3U);
        }
        BuildTop3(assignmentLoadOffset, tiling.epSize, FAST_ASSIGNMENT_CACHE);
        BuildTop3(stateRankCostOffset, tiling.epSize, FAST_STATE_CACHE);
    }

    __aicore__ inline float CachedUnaffectedMaximum(
        uint32_t cacheOffset,
        uint32_t firstAffected,
        uint32_t secondAffected,
        float initial) const
    {
        float maximum = initial;
        for (uint32_t position = 0; position < 3U; ++position) {
            const int64_t index = intWorkspaceGm.GetValue(FastIndexOffset(cacheOffset + position));
            if (index < 0 || static_cast<uint32_t>(index) == firstAffected
                || static_cast<uint32_t>(index) == secondAffected) {
                continue;
            }
            const float value = floatWorkspaceGm.GetValue(FastValueOffset(cacheOffset + position));
            if (value > maximum) {
                maximum = value;
            }
            break;
        }
        return maximum;
    }

    __aicore__ inline void InitializeMutableStats()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t stride = tiling.blockCount;
        for (uint32_t logical = block; logical < tiling.numExperts; logical += stride) {
            for (uint32_t index = 0; index < expertGroupDeltaStride; ++index) {
                floatWorkspaceGm.SetValue(
                    expertGroupDeltaOffset + logical * expertGroupDeltaStride + index, 0.0F);
            }
            for (uint32_t index = 0; index < soleExpertDeltaStride; ++index) {
                floatWorkspaceGm.SetValue(
                    soleExpertDeltaOffset + logical * soleExpertDeltaStride + index, 0.0F);
            }
            for (uint32_t index = 0; index < solePairDeltaStride; ++index) {
                floatWorkspaceGm.SetValue(
                    solePairDeltaOffset + logical * solePairDeltaStride + index, 0.0F);
            }
        }
    }

    __aicore__ inline uint32_t UniqueExperts(uint32_t token, int64_t (&experts)[16]) const
    {
        uint32_t count = 0;
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        for (uint32_t position = 0; position < tiling.topK; ++position) {
            const int64_t logical = sampleRoutesGm.GetValue(routeOffset + position);
            if (logical < 0 || logical >= static_cast<int64_t>(tiling.numExperts)) {
                continue;
            }
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

    __aicore__ inline void InitializeMembership()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        for (uint32_t token = block; token < tiling.numSamples; token += tiling.blockCount) {
            uint64_t masks[MEMBERSHIP_WORDS] = {0U, 0U, 0U, 0U};
            const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
            for (uint32_t position = 0; position < tiling.topK; ++position) {
                const int64_t logical = sampleRoutesGm.GetValue(routeOffset + position);
                if (logical < 0 || logical >= static_cast<int64_t>(tiling.numExperts)) {
                    continue;
                }
                const uint32_t logicalIndex = static_cast<uint32_t>(logical);
                masks[logicalIndex >> 6U] |= 1ULL << (logicalIndex & 63U);
            }
            const uint32_t membershipBase = membershipOffset + token * MEMBERSHIP_STRIDE;
            for (uint32_t word = 0; word < MEMBERSHIP_WORDS; ++word) {
                intWorkspaceGm.SetValue(membershipBase + word, static_cast<int64_t>(masks[word]));
            }
        }
    }

    __aicore__ inline void Flush()
    {
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
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

    __aicore__ inline bool IsIntra(uint32_t lhs, uint32_t rhs) const
    {
        const uint32_t width = tiling.localWorldSize > 0 ? tiling.localWorldSize : 1U;
        return lhs / width == rhs / width;
    }

    __aicore__ inline float LinkCost(
        uint32_t lhs,
        uint32_t rhs,
        float payload,
        float intraAlpha,
        float intraBeta,
        float interAlpha,
        float interBeta) const
    {
        if (lhs == rhs || payload <= 0.0F) {
            return 0.0F;
        }
        return IsIntra(lhs, rhs)
            ? intraAlpha + intraBeta * payload
            : interAlpha + interBeta * payload;
    }

    __aicore__ inline void Initialize()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        for (uint32_t slot = 0; slot < tiling.numSlots; ++slot) {
            updatedLayoutGm.SetValue(slot, slotToLogicalGm.GetValue(slot));
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            floatWorkspaceGm.SetValue(assignmentLoadOffset + rank, 0.0F);
            floatWorkspaceGm.SetValue(stateRankCostOffset + rank, 0.0F);
            floatWorkspaceGm.SetValue(gradientRankCostOffset + rank, 0.0F);
            for (uint32_t peer = 0; peer < tiling.epSize; ++peer) {
                floatWorkspaceGm.SetValue(statePayloadOffset + rank * tiling.epSize + peer, 0.0F);
                floatWorkspaceGm.SetValue(gradientPayloadOffset + rank * tiling.epSize + peer, 0.0F);
            }
        }
        for (uint32_t expert = 0; expert < tiling.numExperts; ++expert) {
            const int64_t owner = ownerSlotsGm.GetValue(expert);
            updatedOwnersGm.SetValue(expert, owner);
            intWorkspaceGm.SetValue(usedOffset + expert, 0);
            const uint32_t rank = static_cast<uint32_t>(owner) / tiling.slotsPerRank;
            floatWorkspaceGm.SetValue(
                assignmentLoadOffset + rank,
                floatWorkspaceGm.GetValue(assignmentLoadOffset + rank)
                    + expertAssignmentCountsGm.GetValue(expert));
        }
        for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
            floatWorkspaceGm.SetValue(workingBaseOffset + group, baseCountsGm.GetValue(group));
        }
        for (uint32_t row = 0; row < (tiling.maxSwaps > 0 ? tiling.maxSwaps : 1U); ++row) {
            for (uint32_t column = 0; column < 5; ++column) {
                actionsGm.SetValue(static_cast<uint64_t>(row) * 5U + column, -1);
            }
        }
        intWorkspaceGm.SetValue(controlOffset, 0);
        intWorkspaceGm.SetValue(controlOffset + 1U, 0);
        intWorkspaceGm.SetValue(controlOffset + 2U, 0);
        intWorkspaceGm.SetValue(controlOffset + 3U, 0);
        intWorkspaceGm.SetValue(controlOffset + 4U, 0);
        for (uint32_t slot = 0; slot < tiling.numSlots; ++slot) {
            const int64_t logical = updatedLayoutGm.GetValue(slot);
            if (logical >= 0 && updatedOwnersGm.GetValue(static_cast<uint32_t>(logical)) != static_cast<int64_t>(slot)) {
                intWorkspaceGm.SetValue(controlOffset + 4U, 1);
                const uint32_t source = slot / tiling.slotsPerRank;
                const uint32_t destination = static_cast<uint32_t>(
                    updatedOwnersGm.GetValue(static_cast<uint32_t>(logical))) / tiling.slotsPerRank;
                const uint32_t payloadOffset = gradientPayloadOffset + source * tiling.epSize + destination;
                floatWorkspaceGm.SetValue(
                    payloadOffset,
                    floatWorkspaceGm.GetValue(payloadOffset)
                        + static_cast<float>(expertGradientBytesGm.GetValue(static_cast<uint32_t>(logical))));
            }
        }
        for (uint32_t lhsRank = 0; lhsRank < tiling.epSize; ++lhsRank) {
            for (uint32_t rhsRank = lhsRank + 1U; rhsRank < tiling.epSize; ++rhsRank) {
                const float forward = floatWorkspaceGm.GetValue(
                    gradientPayloadOffset + lhsRank * tiling.epSize + rhsRank);
                const float backward = floatWorkspaceGm.GetValue(
                    gradientPayloadOffset + rhsRank * tiling.epSize + lhsRank);
                const float pairCost = GradientPairCost(lhsRank, rhsRank, forward, backward);
                floatWorkspaceGm.SetValue(
                    gradientRankCostOffset + lhsRank,
                    floatWorkspaceGm.GetValue(gradientRankCostOffset + lhsRank) + pairCost);
                floatWorkspaceGm.SetValue(
                    gradientRankCostOffset + rhsRank,
                    floatWorkspaceGm.GetValue(gradientRankCostOffset + rhsRank) + pairCost);
            }
        }
        int32_t commRank = 0;
        int32_t computeRank = 0;
        const float current = CandidateCost(-1, -1, commRank, computeRank);
        floatWorkspaceGm.SetValue(currentCostOffset, current);
        intWorkspaceGm.SetValue(controlOffset + 2U, commRank);
        intWorkspaceGm.SetValue(controlOffset + 3U, computeRank);
        RefreshFastCostCache();
    }

    __aicore__ inline bool RankContains(uint32_t rank, uint32_t logical) const
    {
        const uint32_t first = rank * tiling.slotsPerRank;
        for (uint32_t local = 0; local < tiling.slotsPerRank; ++local) {
            if (updatedLayoutGm.GetValue(first + local) == static_cast<int64_t>(logical)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline float ExpertGroupCount(uint32_t logical, uint32_t flatGroup) const
    {
        const uint32_t inputIndex = logical * tiling.totalGroups + flatGroup;
        float value = expertGroupCountsGm.GetValue(inputIndex);
        if (tiling.maxSwaps > 1U) {
            value += floatWorkspaceGm.GetValue(
                expertGroupDeltaOffset + logical * expertGroupDeltaStride + flatGroup);
        }
        return value;
    }

    __aicore__ inline float SoleExpertCount(uint32_t logical, uint32_t level) const
    {
        const uint32_t inputIndex = logical * tiling.numLevels + level;
        float value = soleExpertCountsGm.GetValue(inputIndex);
        if (tiling.maxSwaps > 1U) {
            value += floatWorkspaceGm.GetValue(
                soleExpertDeltaOffset + logical * soleExpertDeltaStride + level);
        }
        return value;
    }

    __aicore__ inline float SolePairCount(uint32_t logical, uint32_t level, uint32_t other) const
    {
        const uint32_t inputIndex = (logical * tiling.numLevels + level) * tiling.numExperts + other;
        float value = solePairCountsGm.GetValue(inputIndex);
        if (tiling.maxSwaps > 1U) {
            value += floatWorkspaceGm.GetValue(
                solePairDeltaOffset + logical * solePairDeltaStride + level * tiling.numExperts + other);
        }
        return value;
    }

    __aicore__ inline float GroupValue(uint32_t level, uint32_t group, int32_t lhs, int32_t rhs) const
    {
        float value = floatWorkspaceGm.GetValue(workingBaseOffset + LevelOffset(level) + group);
        if (lhs < 0 || rhs < 0) {
            return value;
        }
        const uint32_t lhsLogical = static_cast<uint32_t>(lhs);
        const uint32_t rhsLogical = static_cast<uint32_t>(rhs);
        const uint32_t size = LevelSize(level);
        const uint32_t lhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(lhsLogical)) / tiling.slotsPerRank;
        const uint32_t rhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(rhsLogical)) / tiling.slotsPerRank;
        const uint32_t lhsGroup = lhsRank / size;
        const uint32_t rhsGroup = rhsRank / size;
        if (lhsGroup == rhsGroup) {
            return value;
        }
        if (group == lhsGroup) {
            value += expertTokenCountsGm.GetValue(rhsLogical)
                - ExpertGroupCount(rhsLogical, LevelOffset(level) + lhsGroup)
                - SoleExpertCount(lhsLogical, level)
                + SolePairCount(lhsLogical, level, rhsLogical);
        }
        if (group == rhsGroup) {
            value += expertTokenCountsGm.GetValue(lhsLogical)
                - ExpertGroupCount(lhsLogical, LevelOffset(level) + rhsGroup)
                - SoleExpertCount(rhsLogical, level)
                + SolePairCount(rhsLogical, level, lhsLogical);
        }
        return value;
    }

    __aicore__ inline float StateMoveCost(int32_t lhs, int32_t rhs) const
    {
        float maximum = 0.0F;
        uint32_t lhsRank = 0;
        uint32_t rhsRank = 0;
        float oldPairCost = 0.0F;
        float newPairCost = 0.0F;
        if (lhs >= 0 && rhs >= 0) {
            lhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(static_cast<uint32_t>(lhs)))
                / tiling.slotsPerRank;
            rhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(static_cast<uint32_t>(rhs)))
                / tiling.slotsPerRank;
            const float oldForward = floatWorkspaceGm.GetValue(statePayloadOffset + lhsRank * tiling.epSize + rhsRank);
            const float oldBackward = floatWorkspaceGm.GetValue(statePayloadOffset + rhsRank * tiling.epSize + lhsRank);
            oldPairCost = LinkCost(
                lhsRank,
                rhsRank,
                oldForward > oldBackward ? oldForward : oldBackward,
                tiling.stateIntraAlpha,
                tiling.stateIntraBeta,
                tiling.stateInterAlpha,
                tiling.stateInterBeta);
            const float newForward = oldForward
                + static_cast<float>(expertStateBytesGm.GetValue(static_cast<uint32_t>(lhs)));
            const float newBackward = oldBackward
                + static_cast<float>(expertStateBytesGm.GetValue(static_cast<uint32_t>(rhs)));
            newPairCost = LinkCost(
                lhsRank,
                rhsRank,
                newForward > newBackward ? newForward : newBackward,
                tiling.stateIntraAlpha,
                tiling.stateIntraBeta,
                tiling.stateInterAlpha,
                tiling.stateInterBeta);
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            float value = floatWorkspaceGm.GetValue(stateRankCostOffset + rank);
            if (lhs >= 0 && rhs >= 0 && (rank == lhsRank || rank == rhsRank)) {
                value += newPairCost - oldPairCost;
            }
            if (value > maximum) {
                maximum = value;
            }
        }
        return maximum * tiling.runtimeCostScale;
    }

    __aicore__ inline float GradientPairCost(
        uint32_t lhsRank,
        uint32_t rhsRank,
        float forward,
        float backward) const
    {
        const float bytes = forward > backward ? forward : backward;
        if (bytes <= 0.0F) {
            return 0.0F;
        }
        return LinkCost(
                   lhsRank,
                   rhsRank,
                   bytes,
                   tiling.gatherIntraAlpha,
                   tiling.gatherIntraBeta,
                   tiling.gatherInterAlpha,
                   tiling.gatherInterBeta)
            + LinkCost(
                   lhsRank,
                   rhsRank,
                   bytes,
                   tiling.scatterIntraAlpha,
                   tiling.scatterIntraBeta,
                   tiling.scatterInterAlpha,
                   tiling.scatterInterBeta);
    }

    __aicore__ inline void AddGradientDelta(
        uint32_t id,
        float delta,
        uint32_t (&ids)[128],
        float (&values)[128],
        uint32_t &count) const
    {
        for (uint32_t index = 0; index < count; ++index) {
            if (ids[index] == id) {
                values[index] += delta;
                return;
            }
        }
        if (count < 128U) {
            ids[count] = id;
            values[count] = delta;
            ++count;
        }
    }

    __aicore__ inline void BuildGradientDeltas(
        uint32_t lhs,
        uint32_t rhs,
        uint32_t (&ids)[128],
        float (&values)[128],
        uint32_t &count) const
    {
        count = 0;
        const int64_t lhsSlot = updatedOwnersGm.GetValue(lhs);
        const int64_t rhsSlot = updatedOwnersGm.GetValue(rhs);
        const uint32_t lhsRank = static_cast<uint32_t>(lhsSlot) / tiling.slotsPerRank;
        const uint32_t rhsRank = static_cast<uint32_t>(rhsSlot) / tiling.slotsPerRank;
        for (uint32_t slot = 0; slot < tiling.numSlots; ++slot) {
            const int64_t logical = updatedLayoutGm.GetValue(slot);
            if (logical != static_cast<int64_t>(lhs) && logical != static_cast<int64_t>(rhs)) {
                continue;
            }
            const int64_t ownerSlot = logical == static_cast<int64_t>(lhs) ? lhsSlot : rhsSlot;
            if (static_cast<int64_t>(slot) == ownerSlot) {
                continue;
            }
            const uint32_t source = slot / tiling.slotsPerRank;
            const uint32_t oldOwner = logical == static_cast<int64_t>(lhs) ? lhsRank : rhsRank;
            const uint32_t newOwner = logical == static_cast<int64_t>(lhs) ? rhsRank : lhsRank;
            const float bytes = static_cast<float>(
                expertGradientBytesGm.GetValue(static_cast<uint32_t>(logical)));
            AddGradientDelta(source * tiling.epSize + oldOwner, -bytes, ids, values, count);
            AddGradientDelta(source * tiling.epSize + newOwner, bytes, ids, values, count);
        }
    }

    __aicore__ inline float GradientSyncCost(int32_t lhs, int32_t rhs) const
    {
        float rankDeltas[64];
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            rankDeltas[rank] = 0.0F;
        }
        if (lhs >= 0 && rhs >= 0 && intWorkspaceGm.GetValue(controlOffset + 4U) != 0) {
            uint32_t ids[128];
            float values[128];
            uint32_t count = 0;
            BuildGradientDeltas(
                static_cast<uint32_t>(lhs), static_cast<uint32_t>(rhs), ids, values, count);
            uint32_t pairIds[128];
            uint32_t pairCount = 0;
            for (uint32_t index = 0; index < count; ++index) {
                const uint32_t source = ids[index] / tiling.epSize;
                const uint32_t destination = ids[index] % tiling.epSize;
                const uint32_t first = source < destination ? source : destination;
                const uint32_t second = source < destination ? destination : source;
                const uint32_t pair = first * tiling.epSize + second;
                bool seen = false;
                for (uint32_t previous = 0; previous < pairCount; ++previous) {
                    seen = seen || pairIds[previous] == pair;
                }
                if (!seen) {
                    pairIds[pairCount++] = pair;
                }
            }
            for (uint32_t pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
                const uint32_t first = pairIds[pairIndex] / tiling.epSize;
                const uint32_t second = pairIds[pairIndex] % tiling.epSize;
                const uint32_t forwardId = first * tiling.epSize + second;
                const uint32_t backwardId = second * tiling.epSize + first;
                float forwardDelta = 0.0F;
                float backwardDelta = 0.0F;
                for (uint32_t index = 0; index < count; ++index) {
                    forwardDelta += ids[index] == forwardId ? values[index] : 0.0F;
                    backwardDelta += ids[index] == backwardId ? values[index] : 0.0F;
                }
                const float oldForward = floatWorkspaceGm.GetValue(gradientPayloadOffset + forwardId);
                const float oldBackward = floatWorkspaceGm.GetValue(gradientPayloadOffset + backwardId);
                const float delta = GradientPairCost(
                    first, second, oldForward + forwardDelta, oldBackward + backwardDelta)
                    - GradientPairCost(first, second, oldForward, oldBackward);
                rankDeltas[first] += delta;
                rankDeltas[second] += delta;
            }
        }
        float maximum = 0.0F;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            const float value = floatWorkspaceGm.GetValue(gradientRankCostOffset + rank) + rankDeltas[rank];
            if (value > maximum) {
                maximum = value;
            }
        }
        return maximum * tiling.runtimeCostScale;
    }

    __aicore__ inline void ApplyGradientSwap(uint32_t lhs, uint32_t rhs)
    {
        if (intWorkspaceGm.GetValue(controlOffset + 4U) == 0) {
            return;
        }
        uint32_t ids[128];
        float values[128];
        uint32_t count = 0;
        BuildGradientDeltas(lhs, rhs, ids, values, count);
        uint32_t pairIds[128];
        uint32_t pairCount = 0;
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t source = ids[index] / tiling.epSize;
            const uint32_t destination = ids[index] % tiling.epSize;
            const uint32_t first = source < destination ? source : destination;
            const uint32_t second = source < destination ? destination : source;
            const uint32_t pair = first * tiling.epSize + second;
            bool seen = false;
            for (uint32_t previous = 0; previous < pairCount; ++previous) {
                seen = seen || pairIds[previous] == pair;
            }
            if (!seen) {
                pairIds[pairCount++] = pair;
            }
        }
        for (uint32_t pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
            const uint32_t first = pairIds[pairIndex] / tiling.epSize;
            const uint32_t second = pairIds[pairIndex] % tiling.epSize;
            const uint32_t forwardId = first * tiling.epSize + second;
            const uint32_t backwardId = second * tiling.epSize + first;
            const float oldForward = floatWorkspaceGm.GetValue(gradientPayloadOffset + forwardId);
            const float oldBackward = floatWorkspaceGm.GetValue(gradientPayloadOffset + backwardId);
            float forwardDelta = 0.0F;
            float backwardDelta = 0.0F;
            for (uint32_t index = 0; index < count; ++index) {
                forwardDelta += ids[index] == forwardId ? values[index] : 0.0F;
                backwardDelta += ids[index] == backwardId ? values[index] : 0.0F;
            }
            const float newForward = oldForward + forwardDelta;
            const float newBackward = oldBackward + backwardDelta;
            const float delta = GradientPairCost(first, second, newForward, newBackward)
                - GradientPairCost(first, second, oldForward, oldBackward);
            floatWorkspaceGm.SetValue(gradientPayloadOffset + forwardId, newForward);
            floatWorkspaceGm.SetValue(gradientPayloadOffset + backwardId, newBackward);
            floatWorkspaceGm.SetValue(
                gradientRankCostOffset + first,
                floatWorkspaceGm.GetValue(gradientRankCostOffset + first) + delta);
            floatWorkspaceGm.SetValue(
                gradientRankCostOffset + second,
                floatWorkspaceGm.GetValue(gradientRankCostOffset + second) + delta);
        }
    }

    __aicore__ inline void AddPrivateDelta(
        const AscendC::LocalTensor<float> &localDelta,
        bool useLocalDelta,
        uint32_t categoryOffset,
        uint32_t index,
        float delta)
    {
        const uint32_t localIndex = categoryOffset - expertGroupDeltaOffset + index;
        if (useLocalDelta) {
            localDelta.SetValue(localIndex, localDelta.GetValue(localIndex) + delta);
            return;
        }
        const uint32_t blockBase = privateDeltaOffset + AscendC::GetBlockIdx() * privateDeltaSize;
        const uint32_t scratchIndex = blockBase + localIndex;
        floatWorkspaceGm.SetValue(scratchIndex, floatWorkspaceGm.GetValue(scratchIndex) + delta);
    }

    __aicore__ inline void BuildPrivateDeltas(uint32_t round)
    {
        const uint64_t actionOffset = static_cast<uint64_t>(round) * 5U;
        const uint32_t lhs = static_cast<uint32_t>(actionsGm.GetValue(actionOffset));
        const uint32_t rhs = static_cast<uint32_t>(actionsGm.GetValue(actionOffset + 1U));
        const uint32_t lhsRank = static_cast<uint32_t>(actionsGm.GetValue(actionOffset + 2U))
            / tiling.slotsPerRank;
        const uint32_t rhsRank = static_cast<uint32_t>(actionsGm.GetValue(actionOffset + 3U))
            / tiling.slotsPerRank;

        if (AscendC::GetBlockIdx() >= updateBlockCount) {
            return;
        }
        AscendC::LocalTensor<float> localDelta = privateDeltaBuffer.Get<float>();
        const bool useLocalDelta = privateDeltaSize <= LOCAL_DELTA_CAPACITY;
        const bool useMembership = tiling.numExperts <= MEMBERSHIP_EXPERT_CAPACITY;
        if (useLocalDelta) {
            for (uint32_t offset = 0; offset < privateDeltaSize; offset += REDUCE_TILE) {
                const uint32_t count = privateDeltaSize - offset < REDUCE_TILE
                    ? privateDeltaSize - offset
                    : REDUCE_TILE;
                AscendC::Duplicate(localDelta[offset], 0.0F, count);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        for (uint32_t token = AscendC::GetBlockIdx(); token < tiling.numSamples; token += updateBlockCount) {
            bool containsLhs = false;
            bool containsRhs = false;
            if (useMembership) {
                const uint32_t membershipBase = membershipOffset + token * MEMBERSHIP_STRIDE;
                const uint64_t lhsMask = static_cast<uint64_t>(
                    intWorkspaceGm.GetValue(membershipBase + (lhs >> 6U)));
                const uint64_t rhsMask = static_cast<uint64_t>(
                    intWorkspaceGm.GetValue(membershipBase + (rhs >> 6U)));
                containsLhs = (lhsMask & (1ULL << (lhs & 63U))) != 0U;
                containsRhs = (rhsMask & (1ULL << (rhs & 63U))) != 0U;
                if (containsLhs == containsRhs) {
                    continue;
                }
            }
            int64_t experts[16];
            uint32_t oldRanks[16];
            uint32_t newRanks[16];
            bool activeExperts[16];
            const uint32_t expertCount = UniqueExperts(token, experts);
            for (uint32_t index = 0; index < expertCount; ++index) {
                const uint32_t expert = static_cast<uint32_t>(experts[index]);
                if (!useMembership) {
                    containsLhs = containsLhs || expert == lhs;
                    containsRhs = containsRhs || expert == rhs;
                }
                newRanks[index] = static_cast<uint32_t>(updatedOwnersGm.GetValue(expert))
                    / tiling.slotsPerRank;
                oldRanks[index] = expert == lhs ? lhsRank : (expert == rhs ? rhsRank : newRanks[index]);
                activeExperts[index] = intWorkspaceGm.GetValue(usedOffset + expert) == 0;
            }
            if (!useMembership && containsLhs == containsRhs) {
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
                    const uint32_t oldGroup = oldRanks[index] / levelSize;
                    lhsOccupancy += oldGroup == lhsGroup;
                    rhsOccupancy += oldGroup == rhsGroup;
                }
                const uint32_t newLhsOccupancy = lhsOccupancy
                    - static_cast<uint32_t>(containsLhs) + static_cast<uint32_t>(containsRhs);
                const uint32_t newRhsOccupancy = rhsOccupancy
                    - static_cast<uint32_t>(containsRhs) + static_cast<uint32_t>(containsLhs);

                for (uint32_t logicalIndex = 0; logicalIndex < expertCount; ++logicalIndex) {
                    const uint32_t logical = static_cast<uint32_t>(experts[logicalIndex]);
                    if (!activeExperts[logicalIndex]) {
                        continue;
                    }

                    const uint32_t expertGroupBase = logical * expertGroupDeltaStride + LevelOffset(level);
                    if ((lhsOccupancy == 0U) != (newLhsOccupancy == 0U)) {
                        AddPrivateDelta(
                            localDelta,
                            useLocalDelta,
                            expertGroupDeltaOffset,
                            expertGroupBase + lhsGroup,
                            newLhsOccupancy == 0U ? -weight : weight);
                    }
                    if ((rhsOccupancy == 0U) != (newRhsOccupancy == 0U)) {
                        AddPrivateDelta(
                            localDelta,
                            useLocalDelta,
                            expertGroupDeltaOffset,
                            expertGroupBase + rhsGroup,
                            newRhsOccupancy == 0U ? -weight : weight);
                    }

                    const uint32_t logicalOldGroup = oldRanks[logicalIndex] / levelSize;
                    const uint32_t logicalNewGroup = newRanks[logicalIndex] / levelSize;
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
                    AddPrivateDelta(
                        localDelta,
                        useLocalDelta,
                        soleExpertDeltaOffset,
                        logical * soleExpertDeltaStride + level,
                        delta);
                    const uint32_t pairBase = logical * solePairDeltaStride + level * tiling.numExperts;
                    for (uint32_t index = 0; index < expertCount; ++index) {
                        if (!activeExperts[index]) {
                            continue;
                        }
                        AddPrivateDelta(
                            localDelta,
                            useLocalDelta,
                            solePairDeltaOffset,
                            pairBase + static_cast<uint32_t>(experts[index]),
                            delta);
                    }
                }
            }
        }
        if (useLocalDelta) {
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint32_t privateBase = privateDeltaOffset + AscendC::GetBlockIdx() * privateDeltaSize;
            for (uint32_t offset = 0; offset < privateDeltaSize; offset += REDUCE_TILE) {
                const uint32_t count = privateDeltaSize - offset < REDUCE_TILE
                    ? privateDeltaSize - offset
                    : REDUCE_TILE;
                AscendC::DataCopy(floatWorkspaceGm[privateBase + offset], localDelta[offset], count);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ReduceLocalPrivateDeltasSharded()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        if (block >= updateBlockCount) {
            return;
        }
        AscendC::LocalTensor<float> accumulator = reduceLeftBuffer.Get<float>();
        AscendC::LocalTensor<float> sourceTile = reduceRightBuffer.Get<float>();
        const uint32_t alignedUnits = privateDeltaSize / 16U;
        const uint32_t unitStart = alignedUnits * block / updateBlockCount;
        const uint32_t unitEnd = alignedUnits * (block + 1U) / updateBlockCount;
        const uint32_t segmentStart = unitStart * 16U;
        const uint32_t segmentEnd = unitEnd * 16U;
        for (uint32_t offset = segmentStart; offset < segmentEnd; offset += REDUCE_TILE) {
            const uint32_t count = segmentEnd - offset < REDUCE_TILE
                ? segmentEnd - offset
                : REDUCE_TILE;
            AscendC::Duplicate(accumulator, 0.0F, count);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t sourceBlock = 0; sourceBlock < updateBlockCount; ++sourceBlock) {
                const uint32_t sourceBase = privateDeltaOffset + sourceBlock * privateDeltaSize;
                AscendC::DataCopy(sourceTile, floatWorkspaceGm[sourceBase + offset], count);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(accumulator, accumulator, sourceTile, count);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::DataCopy(sourceTile, floatWorkspaceGm[expertGroupDeltaOffset + offset], count);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(accumulator, accumulator, sourceTile, count);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(floatWorkspaceGm[expertGroupDeltaOffset + offset], accumulator, count);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ReducePrivateDeltas()
    {
        if (privateDeltaSize <= LOCAL_DELTA_CAPACITY) {
            ReduceLocalPrivateDeltasSharded();
            return;
        }
        AscendC::LocalTensor<float> left = reduceLeftBuffer.Get<float>();
        AscendC::LocalTensor<float> right = reduceRightBuffer.Get<float>();
        const uint32_t block = AscendC::GetBlockIdx();
        for (uint32_t stride = 1U; stride < updateBlockCount; stride <<= 1U) {
            if (block < updateBlockCount && block % (2U * stride) == 0U && block + stride < updateBlockCount) {
                const uint32_t destinationBase = privateDeltaOffset + block * privateDeltaSize;
                const uint32_t sourceBase = privateDeltaOffset + (block + stride) * privateDeltaSize;
                for (uint32_t offset = 0; offset < privateDeltaSize; offset += REDUCE_TILE) {
                    const uint32_t count = privateDeltaSize - offset < REDUCE_TILE
                        ? privateDeltaSize - offset
                        : REDUCE_TILE;
                    AscendC::DataCopy(left, floatWorkspaceGm[destinationBase + offset], count);
                    AscendC::DataCopy(right, floatWorkspaceGm[sourceBase + offset], count);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Add(left, left, right, count);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopy(floatWorkspaceGm[destinationBase + offset], left, count);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            Flush();
            AscendC::SyncAll<true>();
        }

        if (block == 0U) {
            for (uint32_t offset = 0; offset < privateDeltaSize; offset += REDUCE_TILE) {
                const uint32_t count = privateDeltaSize - offset < REDUCE_TILE
                    ? privateDeltaSize - offset
                    : REDUCE_TILE;
                AscendC::DataCopy(left, floatWorkspaceGm[expertGroupDeltaOffset + offset], count);
                AscendC::DataCopy(right, floatWorkspaceGm[privateDeltaOffset + offset], count);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(left, left, right, count);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(floatWorkspaceGm[expertGroupDeltaOffset + offset], left, count);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        Flush();
        AscendC::SyncAll<true>();

        if (block >= updateBlockCount) {
            return;
        }
        const uint32_t privateBase = privateDeltaOffset + block * privateDeltaSize;
        for (uint32_t offset = 0; offset < privateDeltaSize; offset += REDUCE_TILE) {
            const uint32_t count = privateDeltaSize - offset < REDUCE_TILE
                ? privateDeltaSize - offset
                : REDUCE_TILE;
            AscendC::Duplicate(left, 0.0F, count);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(floatWorkspaceGm[privateBase + offset], left, count);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline float FastStateMoveCost(
        uint32_t lhs,
        uint32_t rhs,
        uint32_t lhsRank,
        uint32_t rhsRank) const
    {
        const float oldForward = floatWorkspaceGm.GetValue(
            statePayloadOffset + lhsRank * tiling.epSize + rhsRank);
        const float oldBackward = floatWorkspaceGm.GetValue(
            statePayloadOffset + rhsRank * tiling.epSize + lhsRank);
        const float oldPairCost = LinkCost(
            lhsRank,
            rhsRank,
            oldForward > oldBackward ? oldForward : oldBackward,
            tiling.stateIntraAlpha,
            tiling.stateIntraBeta,
            tiling.stateInterAlpha,
            tiling.stateInterBeta);
        const float newForward = oldForward + static_cast<float>(expertStateBytesGm.GetValue(lhs));
        const float newBackward = oldBackward + static_cast<float>(expertStateBytesGm.GetValue(rhs));
        const float newPairCost = LinkCost(
            lhsRank,
            rhsRank,
            newForward > newBackward ? newForward : newBackward,
            tiling.stateIntraAlpha,
            tiling.stateIntraBeta,
            tiling.stateInterAlpha,
            tiling.stateInterBeta);
        const float delta = newPairCost - oldPairCost;
        float maximum = CachedUnaffectedMaximum(
            FAST_STATE_CACHE, lhsRank, rhsRank, 0.0F);
        const float lhsValue = floatWorkspaceGm.GetValue(stateRankCostOffset + lhsRank) + delta;
        if (lhsValue > maximum) {
            maximum = lhsValue;
        }
        const float rhsValue = floatWorkspaceGm.GetValue(stateRankCostOffset + rhsRank) + delta;
        if (rhsValue > maximum) {
            maximum = rhsValue;
        }
        return maximum * tiling.runtimeCostScale;
    }

    __aicore__ inline float CandidateCostFast(uint32_t lhs, uint32_t rhs) const
    {
        const uint32_t lhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(lhs))
            / tiling.slotsPerRank;
        const uint32_t rhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(rhs))
            / tiling.slotsPerRank;
        float levelMax[3] = {0.0F, 0.0F, 0.0F};
        float lhsLevelValue[3] = {0.0F, 0.0F, 0.0F};
        float rhsLevelValue[3] = {0.0F, 0.0F, 0.0F};
        uint32_t lhsGroup[3] = {0U, 0U, 0U};
        uint32_t rhsGroup[3] = {0U, 0U, 0U};
        for (uint32_t level = 0; level < tiling.numLevels; ++level) {
            lhsGroup[level] = lhsRank / LevelSize(level);
            rhsGroup[level] = rhsRank / LevelSize(level);
            levelMax[level] = CachedUnaffectedMaximum(
                level * 3U, lhsGroup[level], rhsGroup[level], 0.0F);
            lhsLevelValue[level] = GroupValue(
                level, lhsGroup[level], static_cast<int32_t>(lhs), static_cast<int32_t>(rhs));
            if (lhsLevelValue[level] > levelMax[level]) {
                levelMax[level] = lhsLevelValue[level];
            }
            if (rhsGroup[level] != lhsGroup[level]) {
                rhsLevelValue[level] = GroupValue(
                    level, rhsGroup[level], static_cast<int32_t>(lhs), static_cast<int32_t>(rhs));
                if (rhsLevelValue[level] > levelMax[level]) {
                    levelMax[level] = rhsLevelValue[level];
                }
            } else {
                rhsLevelValue[level] = lhsLevelValue[level];
            }
        }

        const uint32_t rankLevel = tiling.numLevels - 1U;
        float rankMaximum = CachedUnaffectedMaximum(
            rankLevel * 3U, lhsGroup[rankLevel], rhsGroup[rankLevel], -1.0F);
        if (lhsLevelValue[rankLevel] > rankMaximum) {
            rankMaximum = lhsLevelValue[rankLevel];
        }
        if (rhsLevelValue[rankLevel] > rankMaximum) {
            rankMaximum = rhsLevelValue[rankLevel];
        }
        float oneWay = tiling.a2aAlpha + tiling.flatPayloadFactor * rankMaximum * tiling.a2aBeta;
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

        const float lhsAssignments = expertAssignmentCountsGm.GetValue(lhs);
        const float rhsAssignments = expertAssignmentCountsGm.GetValue(rhs);
        float peakAssignment = CachedUnaffectedMaximum(
            FAST_ASSIGNMENT_CACHE, lhsRank, rhsRank, -1.0F);
        const float lhsLoad = floatWorkspaceGm.GetValue(assignmentLoadOffset + lhsRank)
            + rhsAssignments - lhsAssignments;
        if (lhsLoad > peakAssignment) {
            peakAssignment = lhsLoad;
        }
        const float rhsLoad = floatWorkspaceGm.GetValue(assignmentLoadOffset + rhsRank)
            + lhsAssignments - rhsAssignments;
        if (rhsLoad > peakAssignment) {
            peakAssignment = rhsLoad;
        }
        return 4.0F * oneWay * tiling.communicationScale
            + tiling.computePerAssignment * peakAssignment
            + FastStateMoveCost(lhs, rhs, lhsRank, rhsRank)
            + 0.0F;
    }

    __aicore__ inline float CandidateCost(int32_t lhs, int32_t rhs, int32_t &commRank, int32_t &computeRank) const
    {
        float levelMax[3] = {0.0F, 0.0F, 0.0F};
        const uint32_t rankLevel = tiling.numLevels - 1U;
        float rankMaximum = -1.0F;
        commRank = 0;
        for (uint32_t level = 0; level < tiling.numLevels; ++level) {
            for (uint32_t group = 0; group < LevelGroups(level); ++group) {
                const float value = GroupValue(level, group, lhs, rhs);
                if (value > levelMax[level]) {
                    levelMax[level] = value;
                }
                if (level == rankLevel && value > rankMaximum) {
                    rankMaximum = value;
                    commRank = static_cast<int32_t>(group);
                }
            }
        }
        float oneWay = tiling.a2aAlpha + tiling.flatPayloadFactor * rankMaximum * tiling.a2aBeta;
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

        uint32_t lhsRank = 0;
        uint32_t rhsRank = 0;
        float lhsAssignments = 0.0F;
        float rhsAssignments = 0.0F;
        if (lhs >= 0 && rhs >= 0) {
            lhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(static_cast<uint32_t>(lhs)))
                / tiling.slotsPerRank;
            rhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(static_cast<uint32_t>(rhs)))
                / tiling.slotsPerRank;
            lhsAssignments = expertAssignmentCountsGm.GetValue(static_cast<uint32_t>(lhs));
            rhsAssignments = expertAssignmentCountsGm.GetValue(static_cast<uint32_t>(rhs));
        }
        float peakAssignment = -1.0F;
        computeRank = 0;
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
            + tiling.computePerAssignment * peakAssignment
            + StateMoveCost(lhs, rhs)
            + GradientSyncCost(lhs, rhs);
    }

    __aicore__ inline void ScoreCandidates(uint32_t round)
    {
        const uint32_t block = AscendC::GetBlockIdx();
        float bestCost = 3.402823466e38F;
        int64_t bestLhs = -1;
        int64_t bestRhs = -1;
        const bool active = intWorkspaceGm.GetValue(controlOffset) == static_cast<int64_t>(round);
        const bool useFastCost = intWorkspaceGm.GetValue(controlOffset + 4U) == 0;
        const uint32_t commRank = static_cast<uint32_t>(intWorkspaceGm.GetValue(controlOffset + 2U));
        const uint32_t computeRank = static_cast<uint32_t>(intWorkspaceGm.GetValue(controlOffset + 3U));
        const uint32_t totalPairs = tiling.numExperts * (tiling.numExperts - 1U) / 2U;
        uint32_t pairIndex = block;
        uint32_t lhs = 0;
        uint32_t rowStart = 0;
        uint32_t rowEnd = tiling.numExperts - 1U;
        uint32_t cachedLhs = tiling.numExperts;
        uint32_t lhsRank = 0;
        bool lhsUsed = false;
        bool lhsHot = false;
        while (active && pairIndex < totalPairs) {
            while (pairIndex >= rowEnd) {
                rowStart = rowEnd;
                ++lhs;
                rowEnd += tiling.numExperts - lhs - 1U;
            }
            const uint32_t rhs = lhs + 1U + pairIndex - rowStart;
            if (lhs != cachedLhs) {
                cachedLhs = lhs;
                lhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(lhs)) / tiling.slotsPerRank;
                lhsUsed = intWorkspaceGm.GetValue(usedOffset + lhs) != 0;
                lhsHot = lhsRank == commRank || lhsRank == computeRank;
            }
            const uint32_t rhsRank = static_cast<uint32_t>(updatedOwnersGm.GetValue(rhs))
                / tiling.slotsPerRank;
            const bool rhsUsed = intWorkspaceGm.GetValue(usedOffset + rhs) != 0;
            const bool rhsHot = rhsRank == commRank || rhsRank == computeRank;
            if (!lhsUsed && !rhsUsed && lhsRank != rhsRank && lhsHot != rhsHot
                && !RankContains(lhsRank, rhs) && !RankContains(rhsRank, lhs)) {
                int32_t candidateCommRank = 0;
                int32_t candidateComputeRank = 0;
                const float cost = useFastCost
                    ? CandidateCostFast(lhs, rhs)
                    : CandidateCost(
                          static_cast<int32_t>(lhs),
                          static_cast<int32_t>(rhs),
                          candidateCommRank,
                          candidateComputeRank);
                if (cost < bestCost
                    || (cost == bestCost
                        && (bestLhs < 0 || lhs < static_cast<uint32_t>(bestLhs)
                            || (lhs == static_cast<uint32_t>(bestLhs)
                                && rhs < static_cast<uint32_t>(bestRhs))))) {
                    bestCost = cost;
                    bestLhs = lhs;
                    bestRhs = rhs;
                }
            }
            pairIndex += tiling.blockCount;
        }
        floatWorkspaceGm.SetValue(bestCostOffset + block * 16U, bestCost);
        intWorkspaceGm.SetValue(bestPairOffset + block * 16U, bestLhs);
        intWorkspaceGm.SetValue(bestPairOffset + block * 16U + 1U, bestRhs);
    }

    __aicore__ inline void UpdateStateWave(uint32_t lhs, uint32_t rhs, uint32_t lhsRank, uint32_t rhsRank)
    {
        const uint32_t forwardOffset = statePayloadOffset + lhsRank * tiling.epSize + rhsRank;
        const uint32_t backwardOffset = statePayloadOffset + rhsRank * tiling.epSize + lhsRank;
        const float oldForward = floatWorkspaceGm.GetValue(forwardOffset);
        const float oldBackward = floatWorkspaceGm.GetValue(backwardOffset);
        const float oldCost = LinkCost(
            lhsRank,
            rhsRank,
            oldForward > oldBackward ? oldForward : oldBackward,
            tiling.stateIntraAlpha,
            tiling.stateIntraBeta,
            tiling.stateInterAlpha,
            tiling.stateInterBeta);
        const float newForward = oldForward + static_cast<float>(expertStateBytesGm.GetValue(lhs));
        const float newBackward = oldBackward + static_cast<float>(expertStateBytesGm.GetValue(rhs));
        const float newCost = LinkCost(
            lhsRank,
            rhsRank,
            newForward > newBackward ? newForward : newBackward,
            tiling.stateIntraAlpha,
            tiling.stateIntraBeta,
            tiling.stateInterAlpha,
            tiling.stateInterBeta);
        floatWorkspaceGm.SetValue(forwardOffset, newForward);
        floatWorkspaceGm.SetValue(backwardOffset, newBackward);
        const float delta = newCost - oldCost;
        floatWorkspaceGm.SetValue(
            stateRankCostOffset + lhsRank,
            floatWorkspaceGm.GetValue(stateRankCostOffset + lhsRank) + delta);
        floatWorkspaceGm.SetValue(
            stateRankCostOffset + rhsRank,
            floatWorkspaceGm.GetValue(stateRankCostOffset + rhsRank) + delta);
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
            const float cost = floatWorkspaceGm.GetValue(bestCostOffset + block * 16U);
            const int64_t lhs = intWorkspaceGm.GetValue(bestPairOffset + block * 16U);
            const int64_t rhs = intWorkspaceGm.GetValue(bestPairOffset + block * 16U + 1U);
            if (lhs < 0 || rhs < 0) {
                continue;
            }
            if (cost < bestCost || (cost == bestCost && (bestLhs < 0 || lhs < bestLhs || (lhs == bestLhs && rhs < bestRhs)))) {
                bestCost = cost;
                bestLhs = lhs;
                bestRhs = rhs;
            }
        }
        const bool active = intWorkspaceGm.GetValue(controlOffset) == static_cast<int64_t>(round);
        if (!active || bestLhs < 0 || bestRhs < 0) {
            intWorkspaceGm.SetValue(controlOffset + 1U, 0);
            return;
        }

        const uint32_t lhs = static_cast<uint32_t>(bestLhs);
        const uint32_t rhs = static_cast<uint32_t>(bestRhs);
        int32_t candidateCommRank = 0;
        int32_t candidateComputeRank = 0;
        bestCost = CandidateCost(
            static_cast<int32_t>(lhs),
            static_cast<int32_t>(rhs),
            candidateCommRank,
            candidateComputeRank);
        const bool accepted = bestCost < floatWorkspaceGm.GetValue(currentCostOffset);
        intWorkspaceGm.SetValue(controlOffset + 1U, accepted ? 1 : 0);
        if (!accepted) {
            return;
        }
        const int64_t lhsSlot = updatedOwnersGm.GetValue(lhs);
        const int64_t rhsSlot = updatedOwnersGm.GetValue(rhs);
        const uint32_t lhsRank = static_cast<uint32_t>(lhsSlot) / tiling.slotsPerRank;
        const uint32_t rhsRank = static_cast<uint32_t>(rhsSlot) / tiling.slotsPerRank;
        for (uint32_t level = 0; level < tiling.numLevels; ++level) {
            const uint32_t lhsGroup = lhsRank / LevelSize(level);
            const uint32_t rhsGroup = rhsRank / LevelSize(level);
            if (lhsGroup != rhsGroup) {
                floatWorkspaceGm.SetValue(
                    workingBaseOffset + LevelOffset(level) + lhsGroup,
                    GroupValue(level, lhsGroup, bestLhs, bestRhs));
                floatWorkspaceGm.SetValue(
                    workingBaseOffset + LevelOffset(level) + rhsGroup,
                    GroupValue(level, rhsGroup, bestLhs, bestRhs));
            }
        }
        const float lhsAssignments = expertAssignmentCountsGm.GetValue(lhs);
        const float rhsAssignments = expertAssignmentCountsGm.GetValue(rhs);
        floatWorkspaceGm.SetValue(
            assignmentLoadOffset + lhsRank,
            floatWorkspaceGm.GetValue(assignmentLoadOffset + lhsRank) + rhsAssignments - lhsAssignments);
        floatWorkspaceGm.SetValue(
            assignmentLoadOffset + rhsRank,
            floatWorkspaceGm.GetValue(assignmentLoadOffset + rhsRank) + lhsAssignments - rhsAssignments);
        UpdateStateWave(lhs, rhs, lhsRank, rhsRank);
        ApplyGradientSwap(lhs, rhs);
        updatedOwnersGm.SetValue(lhs, rhsSlot);
        updatedOwnersGm.SetValue(rhs, lhsSlot);
        const int64_t lhsValue = updatedLayoutGm.GetValue(static_cast<uint32_t>(lhsSlot));
        const int64_t rhsValue = updatedLayoutGm.GetValue(static_cast<uint32_t>(rhsSlot));
        updatedLayoutGm.SetValue(static_cast<uint32_t>(lhsSlot), rhsValue);
        updatedLayoutGm.SetValue(static_cast<uint32_t>(rhsSlot), lhsValue);
        intWorkspaceGm.SetValue(usedOffset + lhs, 1);
        intWorkspaceGm.SetValue(usedOffset + rhs, 1);
        const uint64_t actionOffset = static_cast<uint64_t>(round) * 5U;
        actionsGm.SetValue(actionOffset, lhs);
        actionsGm.SetValue(actionOffset + 1U, rhs);
        actionsGm.SetValue(actionOffset + 2U, lhsSlot);
        actionsGm.SetValue(actionOffset + 3U, rhsSlot);
        actionsGm.SetValue(actionOffset + 4U, 1);
        intWorkspaceGm.SetValue(controlOffset, static_cast<int64_t>(round + 1U));
        intWorkspaceGm.SetValue(controlOffset + 2U, candidateCommRank);
        intWorkspaceGm.SetValue(controlOffset + 3U, candidateComputeRank);
        floatWorkspaceGm.SetValue(currentCostOffset, bestCost);
        RefreshFastCostCache();
    }

    AscendC::GlobalTensor<float> expertTokenCountsGm;
    AscendC::GlobalTensor<float> expertAssignmentCountsGm;
    AscendC::GlobalTensor<float> baseCountsGm;
    AscendC::GlobalTensor<float> expertGroupCountsGm;
    AscendC::GlobalTensor<float> soleExpertCountsGm;
    AscendC::GlobalTensor<float> solePairCountsGm;
    AscendC::GlobalTensor<int64_t> sampleRoutesGm;
    AscendC::GlobalTensor<float> sampleWeightsGm;
    AscendC::GlobalTensor<int64_t> slotToLogicalGm;
    AscendC::GlobalTensor<int64_t> ownerSlotsGm;
    AscendC::GlobalTensor<int64_t> expertStateBytesGm;
    AscendC::GlobalTensor<int64_t> expertGradientBytesGm;
    AscendC::GlobalTensor<int64_t> updatedLayoutGm;
    AscendC::GlobalTensor<int64_t> updatedOwnersGm;
    AscendC::GlobalTensor<int64_t> actionsGm;
    AscendC::GlobalTensor<int32_t> metadataGm;
    AscendC::GlobalTensor<float> floatWorkspaceGm;
    AscendC::GlobalTensor<int64_t> intWorkspaceGm;
    HiermoeSwapSelectTilingData tiling;
    uint32_t workingBaseOffset = 0;
    uint32_t assignmentLoadOffset = 0;
    uint32_t statePayloadOffset = 0;
    uint32_t stateRankCostOffset = 0;
    uint32_t gradientPayloadOffset = 0;
    uint32_t gradientRankCostOffset = 0;
    uint32_t bestCostOffset = 0;
    uint32_t currentCostOffset = 0;
    uint32_t expertGroupDeltaStride = 0;
    uint32_t soleExpertDeltaStride = 0;
    uint32_t solePairDeltaStride = 0;
    uint32_t expertGroupDeltaOffset = 0;
    uint32_t soleExpertDeltaOffset = 0;
    uint32_t solePairDeltaOffset = 0;
    uint32_t privateDeltaOffset = 0;
    uint32_t privateDeltaSize = 0;
    uint32_t updateBlockCount = 0;
    uint32_t usedOffset = 0;
    uint32_t bestPairOffset = 0;
    uint32_t controlOffset = 0;
    uint32_t membershipOffset = 0;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceLeftBuffer;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceRightBuffer;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> privateDeltaBuffer;
};

extern "C" __global__ __aicore__ void hiermoe_swap_select(
    GM_ADDR expertTokenCounts,
    GM_ADDR expertAssignmentCounts,
    GM_ADDR baseCounts,
    GM_ADDR expertGroupCounts,
    GM_ADDR soleExpertCounts,
    GM_ADDR solePairCounts,
    GM_ADDR sampleRoutes,
    GM_ADDR sampleWeights,
    GM_ADDR slotToLogical,
    GM_ADDR ownerSlots,
    GM_ADDR expertStateBytes,
    GM_ADDR expertGradientBytes,
    GM_ADDR updatedLayout,
    GM_ADDR updatedOwners,
    GM_ADDR actions,
    GM_ADDR metadata,
    GM_ADDR floatWorkspace,
    GM_ADDR intWorkspace,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeSwapSelectTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeSwapSelect op;
    op.Init(
        expertTokenCounts,
        expertAssignmentCounts,
        baseCounts,
        expertGroupCounts,
        soleExpertCounts,
        solePairCounts,
        sampleRoutes,
        sampleWeights,
        slotToLogical,
        ownerSlots,
        expertStateBytes,
        expertGradientBytes,
        updatedLayout,
        updatedOwners,
        actions,
        metadata,
        floatWorkspace,
        intWorkspace,
        tilingData);
    op.Process();
}
