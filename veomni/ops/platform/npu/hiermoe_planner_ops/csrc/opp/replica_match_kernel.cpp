#include "kernel_operator.h"

#include "replica_match_tiling.h"

class KernelHiermoeReplicaMatch {
public:
    __aicore__ inline void Init(
        GM_ADDR baseCounts,
        GM_ADDR assignmentLoads,
        GM_ADDR addGroupDeltas,
        GM_ADDR addAssignmentDeltas,
        GM_ADDR removeGroupDeltas,
        GM_ADDR removeAssignmentDeltas,
        GM_ADDR slotToLogical,
        GM_ADDR ownerSlots,
        GM_ADDR redundantSlots,
        GM_ADDR candidateExperts,
        GM_ADDR expertStateBytes,
        GM_ADDR expertGradientBytes,
        GM_ADDR updatedLayout,
        GM_ADDR actions,
        GM_ADDR actionGains,
        GM_ADDR selectedColumns,
        GM_ADDR gainMatrix,
        GM_ADDR metadata,
        GM_ADDR floatWorkspace,
        const HiermoeReplicaMatchTilingData &tiling)
    {
        this->tiling = tiling;
        baseCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(baseCounts));
        assignmentLoadsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(assignmentLoads));
        addGroupDeltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addGroupDeltas));
        addAssignmentDeltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addAssignmentDeltas));
        removeGroupDeltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(removeGroupDeltas));
        removeAssignmentDeltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(removeAssignmentDeltas));
        slotToLogicalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slotToLogical));
        ownerSlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ownerSlots));
        redundantSlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(redundantSlots));
        candidateExpertsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(candidateExperts));
        expertStateBytesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(expertStateBytes));
        expertGradientBytesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(expertGradientBytes));
        updatedLayoutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(updatedLayout));
        actionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actions));
        actionGainsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(actionGains));
        selectedColumnsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selectedColumns));
        gainMatrixGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gainMatrix));
        metadataGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(metadata));
        floatWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(floatWorkspace));

        gradientPayloadOffset = 0;
        gradientRankCostOffset = Align16(tiling.epSize * tiling.epSize);
        baselineCostOffset = Align16(gradientRankCostOffset + tiling.epSize);
        selectedStride = Align16(tiling.redundantSlotsPerRank);
        gainStride = Align16(tiling.redundantSlotsPerRank * tiling.numColumns);
    }

    __aicore__ inline void Process()
    {
        Initialize();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(floatWorkspaceGm);
        AscendC::SyncAll<true>();
        ScoreAndMatchRanks();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gainMatrixGm);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(selectedColumnsGm);
        AscendC::SyncAll<true>();
        if (AscendC::GetBlockIdx() == 0) {
            SelectAndApply();
        }
    }

private:
    static constexpr uint32_t kMaxExperts = 256U;
    static constexpr uint32_t kMaxRedundantSlots = 8U;
    static constexpr uint32_t kMaxCopiesPerExpert = 8U;
    static constexpr uint32_t kMaxColumns = kMaxExperts + 2U * kMaxRedundantSlots;
    static constexpr uint32_t kMaxSlotActions = 64U * kMaxRedundantSlots;
    static constexpr float kInvalidGain = -3.402823466e38F;
    static constexpr float kHugeCost = 1.0e30F;

    __aicore__ inline uint32_t Align16(uint32_t value) const
    {
        return (value + 15U) / 16U * 16U;
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

    __aicore__ inline float GradientPairCost(
        uint32_t lhs,
        uint32_t rhs,
        float forward,
        float backward) const
    {
        const float payload = forward > backward ? forward : backward;
        return LinkCost(
                   lhs,
                   rhs,
                   payload,
                   tiling.gatherIntraAlpha,
                   tiling.gatherIntraBeta,
                   tiling.gatherInterAlpha,
                   tiling.gatherInterBeta)
            + LinkCost(
                   lhs,
                   rhs,
                   payload,
                   tiling.scatterIntraAlpha,
                   tiling.scatterIntraBeta,
                   tiling.scatterInterAlpha,
                   tiling.scatterInterBeta);
    }

    __aicore__ inline bool OwnerValid(uint32_t logical) const
    {
        if (logical >= tiling.numExperts) {
            return false;
        }
        const int64_t slot = ownerSlotsGm.GetValue(logical);
        return slot >= 0 && slot < static_cast<int64_t>(tiling.numSlots)
            && static_cast<uint32_t>(slot) / tiling.slotsPerRank < tiling.epSize
            && slotToLogicalGm.GetValue(static_cast<uint32_t>(slot)) == static_cast<int64_t>(logical);
    }

    __aicore__ inline bool SlotValid(uint32_t rank, uint32_t row, int64_t &slot, int64_t &oldLogical) const
    {
        slot = redundantSlotsGm.GetValue(rank * tiling.redundantSlotsPerRank + row);
        if (slot < 0 || slot >= static_cast<int64_t>(tiling.numSlots)
            || static_cast<uint32_t>(slot) / tiling.slotsPerRank != rank) {
            oldLogical = -1;
            return false;
        }
        for (uint32_t previous = 0; previous < row; ++previous) {
            // Slot-id order is part of the deterministic Hungarian tie rule.
            if (redundantSlotsGm.GetValue(rank * tiling.redundantSlotsPerRank + previous) >= slot) {
                oldLogical = -1;
                return false;
            }
        }
        oldLogical = slotToLogicalGm.GetValue(static_cast<uint32_t>(slot));
        if (oldLogical < -1 || oldLogical >= static_cast<int64_t>(tiling.numExperts)) {
            return false;
        }
        if (oldLogical >= 0 && ownerSlotsGm.GetValue(static_cast<uint32_t>(oldLogical)) == slot) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool RankContains(uint32_t rank, uint32_t logical) const
    {
        const uint32_t first = rank * tiling.slotsPerRank;
        for (uint32_t local = 0; local < tiling.slotsPerRank; ++local) {
            if (slotToLogicalGm.GetValue(first + local) == static_cast<int64_t>(logical)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void Initialize()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        for (uint32_t slot = 0; slot < tiling.numSlots; ++slot) {
            updatedLayoutGm.SetValue(slot, slotToLogicalGm.GetValue(slot));
        }
        for (uint32_t row = 0; row < tiling.epSize * tiling.redundantSlotsPerRank; ++row) {
            const uint32_t rank = row / tiling.redundantSlotsPerRank;
            const uint32_t localRow = row % tiling.redundantSlotsPerRank;
            selectedColumnsGm.SetValue(rank * selectedStride + localRow, -1);
            actionGainsGm.SetValue(row, 0.0F);
        }
        for (uint32_t row = 0; row < (tiling.epSize * tiling.redundantSlotsPerRank > 0
                 ? tiling.epSize * tiling.redundantSlotsPerRank
                 : 1U);
             ++row) {
            for (uint32_t column = 0; column < 5U; ++column) {
                actionsGm.SetValue(static_cast<uint64_t>(row) * 5U + column, -1);
            }
        }
        for (uint32_t index = 0; index < 16U; ++index) {
            metadataGm.SetValue(index, 0);
        }

        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            floatWorkspaceGm.SetValue(gradientRankCostOffset + rank, 0.0F);
            for (uint32_t peer = 0; peer < tiling.epSize; ++peer) {
                floatWorkspaceGm.SetValue(gradientPayloadOffset + rank * tiling.epSize + peer, 0.0F);
            }
        }
        for (uint32_t slot = 0; slot < tiling.numSlots; ++slot) {
            const int64_t logical = slotToLogicalGm.GetValue(slot);
            if (logical < 0 || logical >= static_cast<int64_t>(tiling.numExperts)
                || !OwnerValid(static_cast<uint32_t>(logical))
                || ownerSlotsGm.GetValue(static_cast<uint32_t>(logical)) == static_cast<int64_t>(slot)) {
                continue;
            }
            const uint32_t source = slot / tiling.slotsPerRank;
            const uint32_t owner = static_cast<uint32_t>(ownerSlotsGm.GetValue(static_cast<uint32_t>(logical)))
                / tiling.slotsPerRank;
            if (source == owner) {
                continue;
            }
            const uint32_t offset = gradientPayloadOffset + source * tiling.epSize + owner;
            floatWorkspaceGm.SetValue(
                offset,
                floatWorkspaceGm.GetValue(offset)
                    + static_cast<float>(expertGradientBytesGm.GetValue(static_cast<uint32_t>(logical))));
        }
        for (uint32_t lhs = 0; lhs < tiling.epSize; ++lhs) {
            for (uint32_t rhs = lhs + 1U; rhs < tiling.epSize; ++rhs) {
                const float pairCost = GradientPairCost(
                    lhs,
                    rhs,
                    floatWorkspaceGm.GetValue(gradientPayloadOffset + lhs * tiling.epSize + rhs),
                    floatWorkspaceGm.GetValue(gradientPayloadOffset + rhs * tiling.epSize + lhs));
                floatWorkspaceGm.SetValue(
                    gradientRankCostOffset + lhs,
                    floatWorkspaceGm.GetValue(gradientRankCostOffset + lhs) + pairCost);
                floatWorkspaceGm.SetValue(
                    gradientRankCostOffset + rhs,
                    floatWorkspaceGm.GetValue(gradientRankCostOffset + rhs) + pairCost);
            }
        }
        int32_t commRank = 0;
        int32_t computeRank = 0;
        const float baseline = CommunicationComputeCost(-1, 0, 0, false, commRank, computeRank)
            + GradientSyncCost(-1, 0, -1);
        floatWorkspaceGm.SetValue(baselineCostOffset, baseline);
        metadataGm.SetValue(1, commRank);
        metadataGm.SetValue(2, computeRank);
        metadataGm.SetValue(3, static_cast<int32_t>(tiling.numColumns));
        metadataGm.SetValue(4, static_cast<int32_t>(tiling.redundantSlotsPerRank));
        metadataGm.SetValue(5, 1);
    }

    __aicore__ inline float GroupDelta(
        int32_t addExpert,
        uint32_t destinationRank,
        uint32_t actionIndex,
        bool remove,
        uint32_t group) const
    {
        float delta = 0.0F;
        if (remove) {
            delta += removeGroupDeltasGm.GetValue(
                static_cast<uint64_t>(actionIndex) * tiling.totalGroups + group);
        }
        if (addExpert >= 0) {
            const uint64_t candidate = static_cast<uint64_t>(addExpert) * tiling.epSize + destinationRank;
            delta += addGroupDeltasGm.GetValue(candidate * tiling.totalGroups + group);
        }
        return delta;
    }

    __aicore__ inline float AssignmentDelta(
        int32_t addExpert,
        uint32_t destinationRank,
        uint32_t actionIndex,
        bool remove,
        uint32_t rank) const
    {
        float delta = 0.0F;
        if (remove) {
            delta += removeAssignmentDeltasGm.GetValue(
                static_cast<uint64_t>(actionIndex) * tiling.epSize + rank);
        }
        if (addExpert >= 0) {
            const uint64_t candidate = static_cast<uint64_t>(addExpert) * tiling.epSize + destinationRank;
            delta += addAssignmentDeltasGm.GetValue(candidate * tiling.epSize + rank);
        }
        return delta;
    }

    __aicore__ inline float CommunicationComputeCost(
        int32_t addExpert,
        uint32_t destinationRank,
        uint32_t actionIndex,
        bool remove,
        int32_t &commRank,
        int32_t &computeRank) const
    {
        float levelMax[3] = {0.0F, 0.0F, 0.0F};
        for (uint32_t level = 0; level < tiling.numLevels; ++level) {
            for (uint32_t group = 0; group < LevelGroups(level); ++group) {
                const uint32_t index = LevelOffset(level) + group;
                float value = baseCountsGm.GetValue(index)
                    + GroupDelta(addExpert, destinationRank, actionIndex, remove, index);
                value = value > 0.0F ? value : 0.0F;
                if (value > levelMax[level]) {
                    levelMax[level] = value;
                }
            }
        }
        const uint32_t rankLevel = tiling.numLevels - 1U;
        float rankMaximum = -1.0F;
        commRank = 0;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            const uint32_t index = LevelOffset(rankLevel) + rank;
            float value = baseCountsGm.GetValue(index)
                + GroupDelta(addExpert, destinationRank, actionIndex, remove, index);
            value = value > 0.0F ? value : 0.0F;
            if (value > rankMaximum) {
                rankMaximum = value;
                commRank = static_cast<int32_t>(rank);
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

        float peakAssignment = -1.0F;
        computeRank = 0;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            float value = assignmentLoadsGm.GetValue(rank)
                + AssignmentDelta(addExpert, destinationRank, actionIndex, remove, rank);
            value = value > 0.0F ? value : 0.0F;
            if (value > peakAssignment) {
                peakAssignment = value;
                computeRank = static_cast<int32_t>(rank);
            }
        }
        return 4.0F * oneWay * tiling.communicationScale
            + tiling.computePerAssignment * peakAssignment;
    }

    __aicore__ inline float StateMoveCost(int32_t addExpert, uint32_t destinationRank) const
    {
        if (addExpert < 0 || !OwnerValid(static_cast<uint32_t>(addExpert))) {
            return 0.0F;
        }
        const uint32_t owner = static_cast<uint32_t>(ownerSlotsGm.GetValue(static_cast<uint32_t>(addExpert)))
            / tiling.slotsPerRank;
        return LinkCost(
                   owner,
                   destinationRank,
                   static_cast<float>(expertStateBytesGm.GetValue(static_cast<uint32_t>(addExpert))),
                   tiling.stateIntraAlpha,
                   tiling.stateIntraBeta,
                   tiling.stateInterAlpha,
                   tiling.stateInterBeta)
            * tiling.runtimeCostScale;
    }

    __aicore__ inline void AddDirectionalDelta(
        uint32_t id,
        float value,
        uint32_t (&ids)[2],
        float (&values)[2],
        uint32_t &count) const
    {
        for (uint32_t index = 0; index < count; ++index) {
            if (ids[index] == id) {
                values[index] += value;
                return;
            }
        }
        if (count < 2U) {
            ids[count] = id;
            values[count] = value;
            ++count;
        }
    }

    __aicore__ inline float GradientSyncCost(
        int32_t addExpert,
        uint32_t destinationRank,
        int32_t oldLogical) const
    {
        uint32_t ids[2];
        float values[2];
        uint32_t count = 0;
        if (oldLogical >= 0 && OwnerValid(static_cast<uint32_t>(oldLogical))) {
            const uint32_t owner = static_cast<uint32_t>(ownerSlotsGm.GetValue(static_cast<uint32_t>(oldLogical)))
                / tiling.slotsPerRank;
            if (owner != destinationRank) {
                AddDirectionalDelta(
                    destinationRank * tiling.epSize + owner,
                    -static_cast<float>(expertGradientBytesGm.GetValue(static_cast<uint32_t>(oldLogical))),
                    ids,
                    values,
                    count);
            }
        }
        if (addExpert >= 0 && OwnerValid(static_cast<uint32_t>(addExpert))) {
            const uint32_t owner = static_cast<uint32_t>(ownerSlotsGm.GetValue(static_cast<uint32_t>(addExpert)))
                / tiling.slotsPerRank;
            if (owner != destinationRank) {
                AddDirectionalDelta(
                    destinationRank * tiling.epSize + owner,
                    static_cast<float>(expertGradientBytesGm.GetValue(static_cast<uint32_t>(addExpert))),
                    ids,
                    values,
                    count);
            }
        }

        float rankDeltas[64];
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            rankDeltas[rank] = 0.0F;
        }
        uint32_t pairIds[2];
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
            float newForward = oldForward + forwardDelta;
            float newBackward = oldBackward + backwardDelta;
            newForward = newForward > 0.0F ? newForward : 0.0F;
            newBackward = newBackward > 0.0F ? newBackward : 0.0F;
            const float delta = GradientPairCost(first, second, newForward, newBackward)
                - GradientPairCost(first, second, oldForward, oldBackward);
            rankDeltas[first] += delta;
            rankDeltas[second] += delta;
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

    __aicore__ inline float CandidateGain(
        uint32_t destinationRank,
        uint32_t row,
        int32_t addExpert,
        int32_t oldLogical,
        bool remove) const
    {
        const uint32_t actionIndex = destinationRank * tiling.redundantSlotsPerRank + row;
        int32_t commRank = 0;
        int32_t computeRank = 0;
        const float cost = CommunicationComputeCost(
                               addExpert,
                               destinationRank,
                               actionIndex,
                               remove,
                               commRank,
                               computeRank)
            + StateMoveCost(addExpert, destinationRank)
            + GradientSyncCost(addExpert, destinationRank, remove ? oldLogical : -1);
        const float gain = floatWorkspaceGm.GetValue(baselineCostOffset) - cost;
        return gain > 0.0F ? gain : kInvalidGain;
    }

    __aicore__ inline void ScoreRank(uint32_t rank)
    {
        for (uint32_t row = 0; row < tiling.redundantSlotsPerRank; ++row) {
            const uint64_t matrixOffset =
                static_cast<uint64_t>(rank) * gainStride + row * tiling.numColumns;
            for (uint32_t column = 0; column < tiling.numColumns; ++column) {
                gainMatrixGm.SetValue(matrixOffset + column, kInvalidGain);
            }
            gainMatrixGm.SetValue(matrixOffset + 2U * row, 0.0F);
            int64_t slot = -1;
            int64_t oldLogical = -1;
            if (!SlotValid(rank, row, slot, oldLogical)) {
                continue;
            }
            if (oldLogical >= 0) {
                gainMatrixGm.SetValue(
                    matrixOffset + 2U * row + 1U,
                    CandidateGain(rank, row, -1, static_cast<int32_t>(oldLogical), true));
            }
            for (uint32_t logical = 0; logical < tiling.numExperts; ++logical) {
                if (candidateExpertsGm.GetValue(logical) == 0
                    || !OwnerValid(logical)
                    || oldLogical == static_cast<int64_t>(logical)
                    || RankContains(rank, logical)) {
                    continue;
                }
                gainMatrixGm.SetValue(
                    matrixOffset + 2U * tiling.redundantSlotsPerRank + logical,
                    CandidateGain(
                        rank,
                        row,
                        static_cast<int32_t>(logical),
                        static_cast<int32_t>(oldLogical),
                        oldLogical >= 0));
            }
        }
    }

    __aicore__ inline void MatchRank(uint32_t rank)
    {
        float largest = 0.0F;
        const uint64_t rankOffset =
            static_cast<uint64_t>(rank) * gainStride;
        for (uint32_t row = 0; row < tiling.redundantSlotsPerRank; ++row) {
            for (uint32_t column = 0; column < tiling.numColumns; ++column) {
                const float value = gainMatrixGm.GetValue(
                    rankOffset + static_cast<uint64_t>(row) * tiling.numColumns + column);
                if (value > largest) {
                    largest = value;
                }
            }
        }

        float u[kMaxRedundantSlots + 1U];
        float v[kMaxColumns + 1U];
        float minimum[kMaxColumns + 1U];
        int32_t matchedRow[kMaxColumns + 1U];
        int32_t parent[kMaxColumns + 1U];
        bool used[kMaxColumns + 1U];
        for (uint32_t row = 0; row <= tiling.redundantSlotsPerRank; ++row) {
            u[row] = 0.0F;
        }
        for (uint32_t column = 0; column <= tiling.numColumns; ++column) {
            v[column] = 0.0F;
            matchedRow[column] = 0;
            parent[column] = 0;
        }
        for (uint32_t row = 1; row <= tiling.redundantSlotsPerRank; ++row) {
            matchedRow[0] = static_cast<int32_t>(row);
            for (uint32_t column = 0; column <= tiling.numColumns; ++column) {
                minimum[column] = kHugeCost;
                used[column] = false;
            }
            uint32_t column0 = 0;
            while (true) {
                used[column0] = true;
                const uint32_t row0 = static_cast<uint32_t>(matchedRow[column0]);
                float delta = kHugeCost;
                uint32_t column1 = 0;
                for (uint32_t column = 1; column <= tiling.numColumns; ++column) {
                    if (used[column]) {
                        continue;
                    }
                    const float weight = gainMatrixGm.GetValue(
                        rankOffset + static_cast<uint64_t>(row0 - 1U) * tiling.numColumns + column - 1U);
                    const float cost = weight > kInvalidGain * 0.5F ? largest - weight : kHugeCost;
                    const float current = cost - u[row0] - v[column];
                    if (current < minimum[column]) {
                        minimum[column] = current;
                        parent[column] = static_cast<int32_t>(column0);
                    }
                    if (minimum[column] < delta) {
                        delta = minimum[column];
                        column1 = column;
                    }
                }
                for (uint32_t column = 0; column <= tiling.numColumns; ++column) {
                    if (used[column]) {
                        u[static_cast<uint32_t>(matchedRow[column])] += delta;
                        v[column] -= delta;
                    } else {
                        minimum[column] -= delta;
                    }
                }
                column0 = column1;
                if (matchedRow[column0] == 0) {
                    break;
                }
            }
            while (true) {
                const uint32_t column1 = static_cast<uint32_t>(parent[column0]);
                matchedRow[column0] = matchedRow[column1];
                column0 = column1;
                if (column0 == 0) {
                    break;
                }
            }
        }
        for (uint32_t row = 0; row < tiling.redundantSlotsPerRank; ++row) {
            selectedColumnsGm.SetValue(rank * selectedStride + row, -1);
        }
        for (uint32_t column = 1; column <= tiling.numColumns; ++column) {
            if (matchedRow[column] > 0) {
                selectedColumnsGm.SetValue(
                    rank * selectedStride + static_cast<uint32_t>(matchedRow[column] - 1),
                    static_cast<int32_t>(column - 1U));
            }
        }
    }

    __aicore__ inline void ScoreAndMatchRanks()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        for (uint32_t rank = block; rank < tiling.epSize; rank += tiling.blockCount) {
            ScoreRank(rank);
            MatchRank(rank);
        }
    }

    __aicore__ inline bool DecodeSelected(
        uint32_t actionIndex,
        int32_t &kind,
        int64_t &srcSlot,
        int64_t &dstSlot,
        int64_t &newLogical,
        int64_t &oldLogical,
        float &gain) const
    {
        const uint32_t rank = actionIndex / tiling.redundantSlotsPerRank;
        const uint32_t row = actionIndex % tiling.redundantSlotsPerRank;
        const int32_t column = selectedColumnsGm.GetValue(rank * selectedStride + row);
        if (column < 0) {
            return false;
        }
        int64_t slot = -1;
        if (!SlotValid(rank, row, slot, oldLogical)) {
            return false;
        }
        dstSlot = slot;
        const uint64_t matrixOffset = static_cast<uint64_t>(rank) * gainStride + row * tiling.numColumns;
        gain = gainMatrixGm.GetValue(matrixOffset + static_cast<uint32_t>(column));
        if (gain <= 0.0F) {
            return false;
        }
        if (static_cast<uint32_t>(column) == 2U * row + 1U) {
            if (oldLogical < 0) {
                return false;
            }
            kind = 1;
            srcSlot = -1;
            newLogical = -1;
            return true;
        }
        if (static_cast<uint32_t>(column) < 2U * tiling.redundantSlotsPerRank) {
            return false;
        }
        newLogical = static_cast<int64_t>(column) - 2LL * tiling.redundantSlotsPerRank;
        if (newLogical < 0 || newLogical >= static_cast<int64_t>(tiling.numExperts)
            || !OwnerValid(static_cast<uint32_t>(newLogical))) {
            return false;
        }
        kind = 2;
        srcSlot = ownerSlotsGm.GetValue(static_cast<uint32_t>(newLogical));
        return true;
    }

    __aicore__ inline void SelectAndApply()
    {
        bool chosen[kMaxSlotActions];
        uint32_t copyCounts[kMaxExperts];
        const uint32_t numActions = tiling.epSize * tiling.redundantSlotsPerRank;
        for (uint32_t index = 0; index < numActions; ++index) {
            chosen[index] = false;
        }
        for (uint32_t logical = 0; logical < tiling.numExperts; ++logical) {
            copyCounts[logical] = 0U;
        }
        for (uint32_t slot = 0; slot < tiling.numSlots; ++slot) {
            const int64_t logical = slotToLogicalGm.GetValue(slot);
            if (logical >= 0 && logical < static_cast<int64_t>(tiling.numExperts)) {
                ++copyCounts[static_cast<uint32_t>(logical)];
            }
        }
        uint32_t accepted = 0;
        uint32_t matchedPositive = 0;
        for (uint32_t index = 0; index < numActions; ++index) {
            int32_t kind = 0;
            int64_t srcSlot = -1;
            int64_t dstSlot = -1;
            int64_t newLogical = -1;
            int64_t oldLogical = -1;
            float gain = 0.0F;
            matchedPositive += DecodeSelected(
                index, kind, srcSlot, dstSlot, newLogical, oldLogical, gain)
                ? 1U
                : 0U;
        }
        while (accepted < tiling.maxActions) {
            int32_t best = -1;
            float bestGain = 0.0F;
            int64_t bestDst = 0;
            int64_t bestLogical = 0;
            int32_t bestKind = 0;
            for (uint32_t index = 0; index < numActions; ++index) {
                if (chosen[index]) {
                    continue;
                }
                int32_t kind = 0;
                int64_t srcSlot = -1;
                int64_t dstSlot = -1;
                int64_t newLogical = -1;
                int64_t oldLogical = -1;
                float gain = 0.0F;
                if (!DecodeSelected(index, kind, srcSlot, dstSlot, newLogical, oldLogical, gain)) {
                    continue;
                }
                // Runtime routing stores at most eight physical copies for one
                // logical expert.  Rank-local Hungarian matching cannot enforce
                // that global limit because the expert column is shared only
                // within a rank, so apply it during deterministic global
                // truncation and keep scanning other profitable actions.
                if (newLogical >= 0
                    && copyCounts[static_cast<uint32_t>(newLogical)] >= kMaxCopiesPerExpert) {
                    continue;
                }
                const int64_t logicalKey = newLogical >= 0 ? newLogical : static_cast<int64_t>(tiling.numExperts);
                if (best < 0 || gain > bestGain
                    || (gain == bestGain
                        && (dstSlot < bestDst
                            || (dstSlot == bestDst
                                && (logicalKey < bestLogical
                                    || (logicalKey == bestLogical && kind < bestKind)))))) {
                    best = static_cast<int32_t>(index);
                    bestGain = gain;
                    bestDst = dstSlot;
                    bestLogical = logicalKey;
                    bestKind = kind;
                }
            }
            if (best < 0) {
                break;
            }
            chosen[static_cast<uint32_t>(best)] = true;
            int32_t kind = 0;
            int64_t srcSlot = -1;
            int64_t dstSlot = -1;
            int64_t newLogical = -1;
            int64_t oldLogical = -1;
            float gain = 0.0F;
            if (!DecodeSelected(
                    static_cast<uint32_t>(best),
                    kind,
                    srcSlot,
                    dstSlot,
                    newLogical,
                    oldLogical,
                    gain)) {
                continue;
            }
            if (oldLogical >= 0) {
                const uint32_t oldIndex = static_cast<uint32_t>(oldLogical);
                if (copyCounts[oldIndex] > 0U) {
                    --copyCounts[oldIndex];
                }
            }
            if (newLogical >= 0) {
                ++copyCounts[static_cast<uint32_t>(newLogical)];
            }
            updatedLayoutGm.SetValue(static_cast<uint32_t>(dstSlot), newLogical);
            const uint64_t output = static_cast<uint64_t>(accepted) * 5U;
            actionsGm.SetValue(output, kind);
            actionsGm.SetValue(output + 1U, srcSlot);
            actionsGm.SetValue(output + 2U, dstSlot);
            actionsGm.SetValue(output + 3U, newLogical);
            actionsGm.SetValue(output + 4U, oldLogical);
            actionGainsGm.SetValue(accepted, gain);
            ++accepted;
        }
        metadataGm.SetValue(0, static_cast<int32_t>(accepted));
        metadataGm.SetValue(6, static_cast<int32_t>(matchedPositive));
    }

    AscendC::GlobalTensor<float> baseCountsGm;
    AscendC::GlobalTensor<float> assignmentLoadsGm;
    AscendC::GlobalTensor<float> addGroupDeltasGm;
    AscendC::GlobalTensor<float> addAssignmentDeltasGm;
    AscendC::GlobalTensor<float> removeGroupDeltasGm;
    AscendC::GlobalTensor<float> removeAssignmentDeltasGm;
    AscendC::GlobalTensor<int64_t> slotToLogicalGm;
    AscendC::GlobalTensor<int64_t> ownerSlotsGm;
    AscendC::GlobalTensor<int64_t> redundantSlotsGm;
    AscendC::GlobalTensor<int32_t> candidateExpertsGm;
    AscendC::GlobalTensor<int64_t> expertStateBytesGm;
    AscendC::GlobalTensor<int64_t> expertGradientBytesGm;
    AscendC::GlobalTensor<int64_t> updatedLayoutGm;
    AscendC::GlobalTensor<int64_t> actionsGm;
    AscendC::GlobalTensor<float> actionGainsGm;
    AscendC::GlobalTensor<int32_t> selectedColumnsGm;
    AscendC::GlobalTensor<float> gainMatrixGm;
    AscendC::GlobalTensor<int32_t> metadataGm;
    AscendC::GlobalTensor<float> floatWorkspaceGm;
    HiermoeReplicaMatchTilingData tiling;
    uint32_t gradientPayloadOffset = 0;
    uint32_t gradientRankCostOffset = 0;
    uint32_t baselineCostOffset = 0;
    uint32_t selectedStride = 0;
    uint32_t gainStride = 0;
};

extern "C" __global__ __aicore__ void hiermoe_replica_match(
    GM_ADDR baseCounts,
    GM_ADDR assignmentLoads,
    GM_ADDR addGroupDeltas,
    GM_ADDR addAssignmentDeltas,
    GM_ADDR removeGroupDeltas,
    GM_ADDR removeAssignmentDeltas,
    GM_ADDR slotToLogical,
    GM_ADDR ownerSlots,
    GM_ADDR redundantSlots,
    GM_ADDR candidateExperts,
    GM_ADDR expertStateBytes,
    GM_ADDR expertGradientBytes,
    GM_ADDR updatedLayout,
    GM_ADDR actions,
    GM_ADDR actionGains,
    GM_ADDR selectedColumns,
    GM_ADDR gainMatrix,
    GM_ADDR metadata,
    GM_ADDR floatWorkspace,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeReplicaMatchTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeReplicaMatch op;
    op.Init(
        baseCounts,
        assignmentLoads,
        addGroupDeltas,
        addAssignmentDeltas,
        removeGroupDeltas,
        removeAssignmentDeltas,
        slotToLogical,
        ownerSlots,
        redundantSlots,
        candidateExperts,
        expertStateBytes,
        expertGradientBytes,
        updatedLayout,
        actions,
        actionGains,
        selectedColumns,
        gainMatrix,
        metadata,
        floatWorkspace,
        tilingData);
    op.Process();
}
