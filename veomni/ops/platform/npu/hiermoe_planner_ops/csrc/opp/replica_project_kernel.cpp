#include "kernel_operator.h"

#include "replica_project_tiling.h"

class KernelHiermoeReplicaProject {
public:
    __aicore__ inline void Init(
        GM_ADDR sampleRoutes,
        GM_ADDR sampleMultiplicity,
        GM_ADDR sampleWeights,
        GM_ADDR sampleSources,
        GM_ADDR sampleOrdinals,
        GM_ADDR assignmentCounts,
        GM_ADDR seedBaseCounts,
        GM_ADDR slotToLogical,
        GM_ADDR ownerSlots,
        GM_ADDR redundantSlots,
        GM_ADDR candidateExperts,
        GM_ADDR baseCounts,
        GM_ADDR assignmentLoads,
        GM_ADDR addGroupDeltas,
        GM_ADDR addAssignmentDeltas,
        GM_ADDR removeGroupDeltas,
        GM_ADDR removeAssignmentDeltas,
        GM_ADDR intWorkspace,
        GM_ADDR floatWorkspace,
        const HiermoeReplicaProjectTilingData &tiling)
    {
        this->tiling = tiling;
        sampleRoutesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleRoutes));
        sampleMultiplicityGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleMultiplicity));
        sampleWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sampleWeights));
        sampleSourcesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleSources));
        sampleOrdinalsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleOrdinals));
        assignmentCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(assignmentCounts));
        seedBaseCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(seedBaseCounts));
        slotToLogicalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slotToLogical));
        ownerSlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ownerSlots));
        redundantSlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(redundantSlots));
        candidateExpertsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(candidateExperts));
        baseCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(baseCounts));
        assignmentLoadsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(assignmentLoads));
        addGroupDeltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addGroupDeltas));
        addAssignmentDeltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addAssignmentDeltas));
        removeGroupDeltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(removeGroupDeltas));
        removeAssignmentDeltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(removeAssignmentDeltas));
        intWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(intWorkspace));
        floatWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(floatWorkspace));
        pipe.InitBuffer(hashSortScratchBuf, 2U * kHardwareSortCapacity * sizeof(float));
        pipe.InitBuffer(hashSortPackedBuf, 2U * kHardwareSortCapacity * sizeof(float));
        ConfigureActionWorkspace(AscendC::GetBlockIdx());
    }

    __aicore__ inline void ConfigureWorkspace(
        uint64_t intWorkspaceBase,
        uint64_t floatWorkspaceBase,
        uint32_t maxRecords,
        uint32_t hashCapacity,
        uint32_t distributionSize)
    {
        workspaceMaxRecords = maxRecords;
        workspaceHashCapacity = hashCapacity;
        workspaceDistributionSize = distributionSize;
        recordPositionOffset = intWorkspaceBase;
        recordMaskOffset = recordPositionOffset + workspaceMaxRecords;
        recordHashOffset = recordMaskOffset + workspaceMaxRecords;
        recordChosenOffset = recordHashOffset + workspaceMaxRecords;
        recordBucketOffset = recordChosenOffset + workspaceMaxRecords;
        tableKeyOffset = recordBucketOffset + workspaceMaxRecords;
        tableMaskOffset = tableKeyOffset + workspaceHashCapacity;
        tableBucketOffset = tableMaskOffset + workspaceHashCapacity;
        bucketKeyOffset = tableBucketOffset + workspaceHashCapacity;
        bucketMaskOffset = bucketKeyOffset + workspaceMaxRecords;
        bucketProcessedOffset = bucketMaskOffset + workspaceMaxRecords;
        distributionOffset = bucketProcessedOffset + workspaceMaxRecords;
        tokenStartOffset = distributionOffset + workspaceDistributionSize;

        rankLoadOffset = floatWorkspaceBase;
        bucketTotalOffset = Align16(tiling.epSize);
        bucketTotalOffset += floatWorkspaceBase;
    }

    __aicore__ inline void ConfigureBaselineWorkspace()
    {
        ConfigureWorkspace(
            0ULL,
            0ULL,
            tiling.maxRecords,
            tiling.hashCapacity,
            tiling.distributionSize);
    }

    __aicore__ inline void ConfigureActionWorkspace(uint32_t block)
    {
        ConfigureWorkspace(
            tiling.actionIntWorkspaceOffset
                + static_cast<uint64_t>(block) * tiling.actionIntWorkspaceStride,
            tiling.actionFloatWorkspaceOffset
                + static_cast<uint64_t>(block) * tiling.actionFloatWorkspaceStride,
            tiling.actionMaxRecords,
            tiling.actionHashCapacity,
            tiling.actionDistributionSize);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        if (block == 0U) {
            ConfigureBaselineWorkspace();
            BuildLogicalCopyMasks();
            const bool oneCopy = StrictOneCopyLayout();
            intWorkspaceGm.SetValue(BaselineModeOffset(), oneCopy ? 1LL : 0LL);
            if (!oneCopy) {
                BuildTokenCoverage();
                Project(kBaseline, 0U, 0U, -1, 0U);
                const uint32_t recordCount = static_cast<uint32_t>(
                    intWorkspaceGm.GetValue(tokenStartOffset + tiling.numSamples));
                BuildLogicalRecordLists(recordCount);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(baseCountsGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(assignmentLoadsGm);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(floatWorkspaceGm);
        AscendC::SyncAll<true>();

        if (OneCopyMode()) {
            BuildOneCopyTokenSummary(block);
        } else {
            BuildSharedLogicalRecordCounts(block);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(floatWorkspaceGm);
        AscendC::SyncAll<true>();
        if (block == 0U) {
            if (OneCopyMode()) {
                FinalizeOneCopyBaseline();
                BuildDirectLogicalRecordOffsets();
            } else {
                BuildSharedLogicalRecordOffsets();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(baseCountsGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(assignmentLoadsGm);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(floatWorkspaceGm);
        AscendC::SyncAll<true>();
        ConfigureActionWorkspace(block);
        if (OneCopyMode()) {
            BuildDirectLogicalRecords(block);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
            AscendC::SyncAll<true>();
            ConfigureActionWorkspace(block);
            BuildDirectSortedLogicalRecords(block);
        } else {
            BuildSharedSortedLogicalRecords(block);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
        AscendC::SyncAll<true>();

        if (OneCopyMode()) {
            BuildAllDestinationMasksParallel(block);
        }
        const uint32_t addActions = tiling.numExperts * tiling.epSize;
        for (uint32_t action = block; action < addActions; action += tiling.blockCount) {
            const uint32_t logical = action / tiling.epSize;
            const uint32_t rank = action % tiling.epSize;
            const uint32_t logicalRecords = SharedLogicalRecordCount(logical);
            if (!SchedulableAdd(logical, rank, logicalRecords)) {
                ZeroAdd(action);
            }
        }
        if (block == 0U) {
            BuildAddSchedule();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
        AscendC::SyncAll<true>();

        int64_t scheduledAction = intWorkspaceGm.GetValue(AddScheduleHeadOffset() + block);
        while (scheduledAction >= 0LL) {
            const uint32_t action = static_cast<uint32_t>(scheduledAction);
            const int64_t nextAction = intWorkspaceGm.GetValue(AddScheduleNextOffset() + action);
            const uint32_t logical = action / tiling.epSize;
            const uint32_t rank = action % tiling.epSize;
            ProjectAction(
                kAdd,
                logical,
                rank,
                -1,
                action,
                SharedLogicalRecordCount(logical));
            scheduledAction = nextAction;
        }

        const uint32_t removeActions = tiling.epSize * tiling.redundantSlotsPerRank;
        for (uint32_t action = block; action < removeActions; action += tiling.blockCount) {
            const uint32_t rank = action / tiling.redundantSlotsPerRank;
            const int64_t slot = redundantSlotsGm.GetValue(action);
            const int64_t logical = slot >= 0 && slot < static_cast<int64_t>(tiling.numSlots)
                ? slotToLogicalGm.GetValue(static_cast<uint32_t>(slot))
                : -1;
            const uint32_t logicalRecords = logical >= 0 && logical < static_cast<int64_t>(tiling.numExperts)
                ? SharedLogicalRecordCount(static_cast<uint32_t>(logical))
                : 0U;
            if (!RemovableSlot(rank, slot) || logicalRecords > tiling.actionMaxRecords) {
                ZeroRemove(action);
                continue;
            }
            ProjectAction(
                kRemove,
                static_cast<uint32_t>(logical),
                rank,
                slot,
                action,
                logicalRecords);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(addGroupDeltasGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(addAssignmentDeltasGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(removeGroupDeltasGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(removeAssignmentDeltasGm);
    }

private:
    static constexpr uint32_t kMaxRanks = 64U;
    static constexpr uint32_t kMaxExperts = 256U;
    static constexpr uint32_t kMaxGroups = 192U;
    static constexpr uint32_t kMaxLevels = 3U;
    static constexpr uint32_t kBaseline = 0U;
    static constexpr uint32_t kAdd = 1U;
    static constexpr uint32_t kRemove = 2U;
    static constexpr uint32_t kHardwareSortCapacity = 4096U;
    static constexpr uint32_t kHardwareSortMaxRecords = 4095U;
    static constexpr uint32_t kRouteHashMax = 1048572U;
    static constexpr uint32_t kFloatIntegerBiasBits = 0x4b000000U;
    static constexpr uint32_t kPaddedScoreBiasBits = 0x4afffffeU;
    static constexpr float kFloatIntegerBias = 8388608.0F;

    __aicore__ inline uint32_t Align16(uint32_t value) const
    {
        return (value + 15U) / 16U * 16U;
    }

    __aicore__ inline uint64_t AlignOffset16(uint64_t value) const
    {
        return (value + 15ULL) / 16ULL * 16ULL;
    }

    __aicore__ inline uint32_t AlignRecordCacheLine(uint32_t value) const
    {
        return (value + 7U) / 8U * 8U;
    }

    __aicore__ inline uint64_t BaselineRecordPositionOffset() const
    {
        return 0ULL;
    }

    __aicore__ inline uint64_t BaselineRecordHashOffset() const
    {
        return 2ULL * tiling.maxRecords;
    }

    __aicore__ inline uint64_t BaselineRecordChosenOffset() const
    {
        return 3ULL * tiling.maxRecords;
    }

    __aicore__ inline uint64_t BaselineRecordNextOffset() const
    {
        return 4ULL * tiling.maxRecords;
    }

    __aicore__ inline uint64_t BaselineDistributionOffset() const
    {
        return 8ULL * tiling.maxRecords + 3ULL * tiling.hashCapacity;
    }

    __aicore__ inline uint64_t BaselineTokenStartOffset() const
    {
        return BaselineDistributionOffset() + tiling.distributionSize;
    }

    __aicore__ inline uint64_t BaselineCopyMaskOffset() const
    {
        return tiling.sharedSummaryOffset;
    }

    __aicore__ inline uint64_t BaselineModeOffset() const
    {
        return AlignOffset16(BaselineCopyMaskOffset() + tiling.numExperts);
    }

    __aicore__ inline uint64_t BaselineTokenCoverageOffset() const
    {
        return BaselineModeOffset() + 16ULL;
    }

    __aicore__ inline uint64_t SharedLogicalStartOffset() const
    {
        return AlignOffset16(
            BaselineTokenCoverageOffset() + 8ULL * tiling.numSamples);
    }

    __aicore__ inline uint64_t SharedLogicalEntryOffset(uint32_t logical) const
    {
        return SharedLogicalStartOffset() + 8ULL * logical;
    }

    __aicore__ inline uint64_t DirectRawRecordsOffset() const
    {
        return AlignOffset16(SharedLogicalStartOffset() + 8ULL * (tiling.numExperts + 1ULL));
    }

    __aicore__ inline uint64_t DirectRawHashesOffset() const
    {
        return AlignOffset16(DirectRawRecordsOffset() + tiling.directRawRecordCapacity);
    }

    __aicore__ inline uint64_t SharedSortedRecordsOffset() const
    {
        return AlignOffset16(DirectRawHashesOffset() + tiling.directRawRecordCapacity);
    }

    __aicore__ inline uint64_t SharedSortedHashesOffset() const
    {
        return AlignOffset16(SharedSortedRecordsOffset() + tiling.sharedRecordCapacity);
    }

    __aicore__ inline uint64_t AddScheduleNextOffset() const
    {
        // Direct raw hashes are dead after all logical-record sorts complete.
        return DirectRawHashesOffset();
    }

    __aicore__ inline uint64_t AddScheduleHeadOffset() const
    {
        return AddScheduleNextOffset()
            + static_cast<uint64_t>(tiling.numExperts) * tiling.epSize;
    }

    __aicore__ inline uint64_t AllDestinationMoveMasksOffset() const
    {
        return AlignOffset16(SharedSortedHashesOffset() + tiling.sharedRecordCapacity);
    }

    __aicore__ inline uint64_t AllDestinationTieMasksOffset() const
    {
        return AllDestinationMoveMasksOffset() + AlignOffset16(tiling.sharedRecordCapacity);
    }

    __aicore__ inline uint32_t BaselineRankStride() const
    {
        return Align16(tiling.epSize);
    }

    __aicore__ inline uint64_t BaselineBlockRankLoadOffset(uint32_t block) const
    {
        return static_cast<uint64_t>(BaselineRankStride())
            + static_cast<uint64_t>(block) * BaselineRankStride();
    }

    __aicore__ inline uint64_t BaselineExpertLoadOffset(uint32_t logical) const
    {
        return static_cast<uint64_t>(BaselineRankStride())
            + static_cast<uint64_t>(tiling.blockCount) * BaselineRankStride()
            + 16ULL * logical;
    }

    __aicore__ inline uint64_t DirectBlockCountOffset(uint32_t block, uint32_t logical) const
    {
        return tiling.directBlockCountOffset
            + static_cast<uint64_t>(block) * tiling.directBlockCountStride + logical;
    }

    __aicore__ inline uint64_t DirectBlockLoadOffset(uint32_t block, uint32_t logical) const
    {
        return tiling.directBlockLoadOffset
            + static_cast<uint64_t>(block) * tiling.directBlockLoadStride + logical;
    }

    __aicore__ inline uint64_t DirectBlockStartOffset(uint32_t block, uint32_t logical) const
    {
        return DirectBlockCountOffset(block, logical) + tiling.directBlockCountStride / 3U;
    }

    __aicore__ inline uint64_t DirectBlockCursorOffset(uint32_t block, uint32_t logical) const
    {
        return DirectBlockStartOffset(block, logical) + tiling.directBlockCountStride / 3U;
    }

    __aicore__ inline uint32_t BlockTokenBegin(uint32_t block) const
    {
        return static_cast<uint32_t>(
            static_cast<uint64_t>(tiling.numSamples) * block / tiling.blockCount);
    }

    __aicore__ inline uint32_t BlockTokenEnd(uint32_t block) const
    {
        return static_cast<uint32_t>(
            static_cast<uint64_t>(tiling.numSamples) * (block + 1U) / tiling.blockCount);
    }

    __aicore__ inline bool OneCopyMode() const
    {
        return intWorkspaceGm.GetValue(BaselineModeOffset()) != 0LL;
    }

    __aicore__ inline bool CanUseAllDestinationMasks(uint32_t logical, uint32_t recordCount) const
    {
        return OneCopyMode() && tiling.epSize > 0U && tiling.epSize <= kMaxRanks
            && candidateExpertsGm.GetValue(logical) != 0
            && recordCount <= tiling.actionMaxRecords;
    }

    __aicore__ inline uint32_t SharedLogicalRecordStart(uint32_t logical) const
    {
        return static_cast<uint32_t>(
            static_cast<uint64_t>(intWorkspaceGm.GetValue(SharedLogicalEntryOffset(logical))));
    }

    __aicore__ inline uint32_t SharedLogicalRecordCount(uint32_t logical) const
    {
        return static_cast<uint32_t>(
            static_cast<uint64_t>(intWorkspaceGm.GetValue(SharedLogicalEntryOffset(logical))) >> 32U);
    }

    __aicore__ inline uint32_t SharedLogicalRecord(uint32_t logical, uint32_t index) const
    {
        return static_cast<uint32_t>(intWorkspaceGm.GetValue(
            SharedSortedRecordsOffset() + SharedLogicalRecordStart(logical) + index));
    }

    __aicore__ inline int64_t SharedLogicalHash(uint32_t logical, uint32_t index) const
    {
        return intWorkspaceGm.GetValue(
            SharedSortedHashesOffset() + SharedLogicalRecordStart(logical) + index);
    }

    __aicore__ inline uint64_t BaselineRouteIndex(uint32_t logical, uint32_t index) const
    {
        const uint32_t value = SharedLogicalRecord(logical, index);
        if (OneCopyMode()) {
            return value;
        }
        return static_cast<uint64_t>(
            intWorkspaceGm.GetValue(BaselineRecordPositionOffset() + value));
    }

    __aicore__ inline int64_t BaselineRouteHash(uint32_t logical, uint32_t index) const
    {
        if (OneCopyMode()) {
            return SharedLogicalHash(logical, index);
        }
        const uint32_t record = SharedLogicalRecord(logical, index);
        return intWorkspaceGm.GetValue(BaselineRecordHashOffset() + record);
    }

    __aicore__ inline uint32_t BaselineChosenRank(uint32_t logical, uint32_t index) const
    {
        if (OneCopyMode()) {
            return OwnerRank(logical);
        }
        const uint32_t record = SharedLogicalRecord(logical, index);
        return static_cast<uint32_t>(
            intWorkspaceGm.GetValue(BaselineRecordChosenOffset() + record));
    }

    __aicore__ inline int64_t LogicalHead(uint32_t logical) const
    {
        return intWorkspaceGm.GetValue(BaselineDistributionOffset() + logical);
    }

    __aicore__ inline uint32_t LevelSize(uint32_t level) const
    {
        return level == 0U ? tiling.levelSize0 : (level == 1U ? tiling.levelSize1 : tiling.levelSize2);
    }

    __aicore__ inline uint32_t LevelGroups(uint32_t level) const
    {
        return level == 0U ? tiling.levelGroups0 : (level == 1U ? tiling.levelGroups1 : tiling.levelGroups2);
    }

    __aicore__ inline uint32_t LevelOffset(uint32_t level) const
    {
        return level == 0U ? tiling.levelOffset0 : (level == 1U ? tiling.levelOffset1 : tiling.levelOffset2);
    }

    __aicore__ inline uint32_t OwnerRank(uint32_t logical) const
    {
        return static_cast<uint32_t>(ownerSlotsGm.GetValue(logical)) / tiling.slotsPerRank;
    }

    __aicore__ inline bool RankContains(uint32_t logical, uint32_t rank) const
    {
        const uint32_t first = rank * tiling.slotsPerRank;
        for (uint32_t local = 0; local < tiling.slotsPerRank; ++local) {
            if (slotToLogicalGm.GetValue(first + local) == static_cast<int64_t>(logical)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool RemovableSlot(uint32_t rank, int64_t slot) const
    {
        if (slot < 0 || slot >= static_cast<int64_t>(tiling.numSlots)
            || static_cast<uint32_t>(slot) / tiling.slotsPerRank != rank) {
            return false;
        }
        const int64_t logical = slotToLogicalGm.GetValue(slot);
        return logical >= 0 && logical < static_cast<int64_t>(tiling.numExperts)
            && ownerSlotsGm.GetValue(static_cast<uint32_t>(logical)) != slot;
    }

    __aicore__ inline bool SchedulableAdd(
        uint32_t logical,
        uint32_t rank,
        uint32_t logicalRecords) const
    {
        return logicalRecords > 0U && logicalRecords <= tiling.actionMaxRecords
            && candidateExpertsGm.GetValue(logical) != 0
            && !RankContains(logical, rank);
    }

    __aicore__ inline void BuildAddSchedule()
    {
        uint64_t selected[(kMaxExperts + 63U) / 64U];
        int32_t heads[kMaxRanks];
        int32_t tails[kMaxRanks];
        uint64_t loads[kMaxRanks];
        for (uint32_t word = 0U; word < (kMaxExperts + 63U) / 64U; ++word) {
            selected[word] = 0ULL;
        }
        for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
            heads[block] = -1;
            tails[block] = -1;
            loads[block] = 0ULL;
        }

        while (true) {
            int32_t bestLogical = -1;
            uint32_t bestRecords = 0U;
            for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
                if ((selected[logical / 64U] & (1ULL << (logical % 64U))) != 0ULL
                    || candidateExpertsGm.GetValue(logical) == 0) {
                    continue;
                }
                const uint32_t records = SharedLogicalRecordCount(logical);
                if (records == 0U || records > tiling.actionMaxRecords) {
                    continue;
                }
                if (bestLogical < 0 || records > bestRecords
                    || (records == bestRecords && logical < static_cast<uint32_t>(bestLogical))) {
                    bestLogical = static_cast<int32_t>(logical);
                    bestRecords = records;
                }
            }
            if (bestLogical < 0) {
                break;
            }

            const uint32_t logical = static_cast<uint32_t>(bestLogical);
            selected[logical / 64U] |= 1ULL << (logical % 64U);
            for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
                if (!SchedulableAdd(logical, rank, bestRecords)) {
                    continue;
                }
                uint32_t targetBlock = 0U;
                for (uint32_t block = 1U; block < tiling.blockCount; ++block) {
                    if (loads[block] < loads[targetBlock]) {
                        targetBlock = block;
                    }
                }
                const uint32_t action = logical * tiling.epSize + rank;
                intWorkspaceGm.SetValue(AddScheduleNextOffset() + action, -1LL);
                if (tails[targetBlock] < 0) {
                    heads[targetBlock] = static_cast<int32_t>(action);
                } else {
                    intWorkspaceGm.SetValue(
                        AddScheduleNextOffset() + static_cast<uint32_t>(tails[targetBlock]),
                        static_cast<int64_t>(action));
                }
                tails[targetBlock] = static_cast<int32_t>(action);
                loads[targetBlock] += bestRecords;
            }
        }

        for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
            intWorkspaceGm.SetValue(AddScheduleHeadOffset() + block, heads[block]);
        }
    }

    __aicore__ inline bool StrictOneCopyLayout() const
    {
        uint32_t validSlots = 0U;
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const int64_t ownerSlot = ownerSlotsGm.GetValue(logical);
            if (ownerSlot < 0 || ownerSlot >= static_cast<int64_t>(tiling.numSlots)
                || slotToLogicalGm.GetValue(static_cast<uint32_t>(ownerSlot))
                    != static_cast<int64_t>(logical)) {
                return false;
            }
        }
        for (uint32_t slot = 0U; slot < tiling.numSlots; ++slot) {
            const int64_t logical = slotToLogicalGm.GetValue(slot);
            if (logical < 0) {
                continue;
            }
            if (logical >= static_cast<int64_t>(tiling.numExperts)
                || ownerSlotsGm.GetValue(static_cast<uint32_t>(logical))
                    != static_cast<int64_t>(slot)) {
                return false;
            }
            ++validSlots;
        }
        return validSlots == tiling.numExperts;
    }

    __aicore__ inline void BuildOneCopyTokenSummary(uint32_t block)
    {
        float rankLoads[kMaxRanks];
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            rankLoads[rank] = 0.0F;
        }
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            intWorkspaceGm.SetValue(DirectBlockCountOffset(block, logical), 0LL);
            floatWorkspaceGm.SetValue(DirectBlockLoadOffset(block, logical), 0.0F);
        }
        const uint32_t tokenBegin = BlockTokenBegin(block);
        const uint32_t tokenEnd = BlockTokenEnd(block);
        for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
            uint64_t coverage[kMaxLevels];
            uint64_t duplicate[kMaxLevels];
            for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
                coverage[level] = 0ULL;
                duplicate[level] = 0ULL;
            }
            const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
            for (uint32_t position = 0U; position < tiling.topK; ++position) {
                const uint64_t routeIndex = routeOffset + position;
                const int64_t multiplicity = sampleMultiplicityGm.GetValue(routeIndex);
                if (multiplicity <= 0) {
                    continue;
                }
                const uint32_t logical = static_cast<uint32_t>(sampleRoutesGm.GetValue(routeIndex));
                const uint32_t owner = OwnerRank(logical);
                const float units = static_cast<float>(multiplicity) * sampleWeightsGm.GetValue(token);
                rankLoads[owner] += units;
                const uint64_t countOffset = DirectBlockCountOffset(block, logical);
                intWorkspaceGm.SetValue(
                    countOffset,
                    intWorkspaceGm.GetValue(countOffset) + 1LL);
                const uint64_t loadOffset = DirectBlockLoadOffset(block, logical);
                floatWorkspaceGm.SetValue(
                    loadOffset,
                    floatWorkspaceGm.GetValue(loadOffset) + units);
                for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
                    const uint32_t group = owner / ComparisonLevelSize(level);
                    const uint64_t bit = 1ULL << group;
                    if ((coverage[level] & bit) != 0ULL) {
                        duplicate[level] |= bit;
                    } else {
                        coverage[level] |= bit;
                    }
                }
            }
            const uint64_t coverageOffset = BaselineTokenCoverageOffset() + 8ULL * token;
            for (uint32_t index = 0U; index < 8U; ++index) {
                intWorkspaceGm.SetValue(coverageOffset + index, 0LL);
            }
            for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
                intWorkspaceGm.SetValue(
                    TokenCoverageIndex(token, level, false),
                    static_cast<int64_t>(coverage[level]));
                intWorkspaceGm.SetValue(
                    TokenCoverageIndex(token, level, true),
                    static_cast<int64_t>(duplicate[level]));
            }
        }
        const uint64_t output = BaselineBlockRankLoadOffset(block);
        for (uint32_t rank = 0U; rank < BaselineRankStride(); ++rank) {
            floatWorkspaceGm.SetValue(output + rank, rank < tiling.epSize ? rankLoads[rank] : 0.0F);
        }
    }

    __aicore__ inline void FinalizeOneCopyBaseline()
    {
        for (uint32_t group = 0U; group < tiling.totalGroups; ++group) {
            baseCountsGm.SetValue(group, seedBaseCountsGm.GetValue(group));
        }
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            float sampleLoad = 0.0F;
            for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
                sampleLoad += floatWorkspaceGm.GetValue(BaselineBlockRankLoadOffset(block) + rank);
            }
            floatWorkspaceGm.SetValue(rank, sampleLoad);
            assignmentLoadsGm.SetValue(rank, 0.0F);
        }
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            int64_t exact = 0LL;
            for (uint32_t source = 0U; source < tiling.epSize; ++source) {
                exact += assignmentCountsGm.GetValue(source * tiling.numExperts + logical);
            }
            const uint32_t owner = OwnerRank(logical);
            assignmentLoadsGm.SetValue(
                owner,
                assignmentLoadsGm.GetValue(owner) + static_cast<float>(exact));
        }
    }

    __aicore__ inline void BuildLogicalCopyMasks()
    {
        // The post-swap layout is immutable while every replica edge is scored,
        // so build its rank-level copy masks once in the shared summary tail.
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            intWorkspaceGm.SetValue(BaselineCopyMaskOffset() + logical, 0LL);
        }
        for (uint32_t slot = 0U; slot < tiling.numSlots; ++slot) {
            const int64_t logical = slotToLogicalGm.GetValue(slot);
            if (logical >= 0 && logical < static_cast<int64_t>(tiling.numExperts)) {
                const uint64_t offset = BaselineCopyMaskOffset() + static_cast<uint32_t>(logical);
                const uint64_t mask = static_cast<uint64_t>(intWorkspaceGm.GetValue(offset))
                    | (1ULL << (slot / tiling.slotsPerRank));
                intWorkspaceGm.SetValue(offset, static_cast<int64_t>(mask));
            }
        }
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const uint64_t offset = BaselineCopyMaskOffset() + logical;
            if (intWorkspaceGm.GetValue(offset) == 0LL) {
                intWorkspaceGm.SetValue(offset, static_cast<int64_t>(1ULL << OwnerRank(logical)));
            }
        }
    }

    __aicore__ inline uint32_t ComparisonLevelSize(uint32_t level) const
    {
        return level + 1U < tiling.numLevels ? LevelSize(level) : 1U;
    }

    __aicore__ inline uint64_t TokenCoverageIndex(uint32_t token, uint32_t level, bool duplicate) const
    {
        return BaselineTokenCoverageOffset()
            + static_cast<uint64_t>(token) * 8ULL
            + static_cast<uint64_t>(duplicate ? tiling.numLevels : 0U) + level;
    }

    __aicore__ inline uint64_t ActualTokenCoverageIndex(
        uint32_t token,
        uint32_t level,
        bool duplicate) const
    {
        return TokenCoverageIndex(token, level, duplicate);
    }

    __aicore__ inline void BuildTokenCoverage()
    {
        // For each hierarchy level, `coverage` identifies owner groups reached
        // by the token and `duplicate` identifies groups reached by at least two
        // distinct logical experts.  The latter lets a record exclude itself
        // without rescanning the token's top-k route.
        for (uint32_t token = 0U; token < tiling.numSamples; ++token) {
            uint64_t coverage[kMaxLevels];
            uint64_t duplicate[kMaxLevels];
            for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
                coverage[level] = 0ULL;
                duplicate[level] = 0ULL;
            }
            const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
            for (uint32_t position = 0U; position < tiling.topK; ++position) {
                const uint64_t routeIndex = routeOffset + position;
                if (sampleMultiplicityGm.GetValue(routeIndex) <= 0) {
                    continue;
                }
                const uint32_t logical = static_cast<uint32_t>(sampleRoutesGm.GetValue(routeIndex));
                const uint32_t owner = OwnerRank(logical);
                for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
                    const uint32_t group = owner / ComparisonLevelSize(level);
                    const uint64_t bit = 1ULL << group;
                    if ((coverage[level] & bit) != 0ULL) {
                        duplicate[level] |= bit;
                    } else {
                        coverage[level] |= bit;
                    }
                }
            }
            for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
                intWorkspaceGm.SetValue(
                    TokenCoverageIndex(token, level, false),
                    static_cast<int64_t>(coverage[level]));
                intWorkspaceGm.SetValue(
                    TokenCoverageIndex(token, level, true),
                    static_cast<int64_t>(duplicate[level]));
            }
        }
    }

    __aicore__ inline uint64_t LogicalCopyMask(uint32_t logical) const
    {
        return static_cast<uint64_t>(intWorkspaceGm.GetValue(BaselineCopyMaskOffset() + logical));
    }

    __aicore__ inline uint64_t CopyMask(
        uint32_t logical,
        uint32_t kind,
        uint32_t actionLogical,
        uint32_t actionRank,
        int64_t removeSlot) const
    {
        uint64_t mask = 0ULL;
        for (uint32_t slot = 0; slot < tiling.numSlots; ++slot) {
            if (static_cast<int64_t>(slot) != removeSlot
                && slotToLogicalGm.GetValue(slot) == static_cast<int64_t>(logical)) {
                mask |= 1ULL << (slot / tiling.slotsPerRank);
            }
        }
        if (kind == kAdd && logical == actionLogical) {
            mask |= 1ULL << actionRank;
        }
        if (mask == 0ULL) {
            mask = 1ULL << OwnerRank(logical);
        }
        return mask;
    }

    __aicore__ inline bool NovelFromCoverage(
        uint32_t destination,
        uint32_t source,
        uint32_t logicalOwner,
        uint32_t level,
        uint64_t coverage,
        uint64_t duplicate) const
    {
        const uint32_t size = ComparisonLevelSize(level);
        if (destination / size == source / size) {
            return false;
        }
        const uint32_t destinationGroup = destination / size;
        const uint64_t bit = 1ULL << destinationGroup;
        const bool sameAsLogicalOwner = destinationGroup == logicalOwner / size;
        const uint64_t occupiedByOther = sameAsLogicalOwner ? duplicate : coverage;
        return (occupiedByOther & bit) == 0ULL;
    }

    __aicore__ inline int32_t CompareClass(
        uint32_t lhs,
        uint32_t rhs,
        uint32_t source,
        uint32_t logicalOwner,
        const uint64_t (&coverage)[kMaxLevels],
        const uint64_t (&duplicate)[kMaxLevels]) const
    {
        for (uint32_t cursor = tiling.numLevels - 1U; cursor > 0U; --cursor) {
            const uint32_t level = cursor - 1U;
            const bool lhsNovel = NovelFromCoverage(
                lhs, source, logicalOwner, level, coverage[level], duplicate[level]);
            const bool rhsNovel = NovelFromCoverage(
                rhs, source, logicalOwner, level, coverage[level], duplicate[level]);
            if (lhsNovel != rhsNovel) {
                return lhsNovel ? 1 : -1;
            }
        }
        const uint32_t finalLevel = tiling.numLevels - 1U;
        const bool lhsNovel = NovelFromCoverage(
            lhs, source, logicalOwner, finalLevel, coverage[finalLevel], duplicate[finalLevel]);
        const bool rhsNovel = NovelFromCoverage(
            rhs, source, logicalOwner, finalLevel, coverage[finalLevel], duplicate[finalLevel]);
        return lhsNovel == rhsNovel ? 0 : (lhsNovel ? 1 : -1);
    }

    __aicore__ inline uint64_t EligibleMask(
        uint32_t token,
        uint32_t logicalOwner,
        uint64_t copies) const
    {
        if (copies != 0ULL && (copies & (copies - 1ULL)) == 0ULL) {
            return copies;
        }
        uint64_t coverage[kMaxLevels];
        uint64_t duplicate[kMaxLevels];
        for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
            coverage[level] = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(TokenCoverageIndex(token, level, false)));
            duplicate[level] = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(TokenCoverageIndex(token, level, true)));
        }
        const uint32_t source = static_cast<uint32_t>(sampleSourcesGm.GetValue(token));
        uint64_t eligible = 0ULL;
        uint32_t best = 0U;
        bool initialized = false;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            if ((copies & (1ULL << rank)) == 0ULL) {
                continue;
            }
            if (!initialized) {
                best = rank;
                eligible = 1ULL << rank;
                initialized = true;
                continue;
            }
            const int32_t comparison = CompareClass(
                rank, best, source, logicalOwner, coverage, duplicate);
            if (comparison < 0) {
                best = rank;
                eligible = 1ULL << rank;
            } else if (comparison == 0) {
                eligible |= 1ULL << rank;
            }
        }
        return eligible;
    }

    __aicore__ inline uint64_t EligiblePairMask(
        uint32_t token,
        uint32_t logicalOwner,
        uint32_t destination) const
    {
        uint64_t coverage[kMaxLevels];
        uint64_t duplicate[kMaxLevels];
        for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
            coverage[level] = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(TokenCoverageIndex(token, level, false)));
            duplicate[level] = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(TokenCoverageIndex(token, level, true)));
        }
        const uint32_t source = static_cast<uint32_t>(sampleSourcesGm.GetValue(token));
        const uint32_t first = logicalOwner < destination ? logicalOwner : destination;
        const uint32_t second = logicalOwner < destination ? destination : logicalOwner;
        const int32_t comparison = CompareClass(
            second, first, source, logicalOwner, coverage, duplicate);
        if (comparison < 0) {
            return 1ULL << second;
        }
        if (comparison > 0) {
            return 1ULL << first;
        }
        return (1ULL << first) | (1ULL << second);
    }

    __aicore__ inline uint64_t GroupRankMask(uint32_t size, uint32_t group) const
    {
        if (size == kMaxRanks) {
            return ~0ULL;
        }
        return ((1ULL << size) - 1ULL) << (group * size);
    }

    __aicore__ inline uint64_t NovelRankMask(
        uint32_t source,
        uint32_t logicalOwner,
        uint32_t level,
        uint64_t coverage,
        uint64_t duplicate) const
    {
        const uint32_t size = ComparisonLevelSize(level);
        const uint32_t sourceGroup = source / size;
        const uint32_t ownerGroup = logicalOwner / size;
        const uint32_t groupCount = tiling.epSize / size;
        uint64_t mask = 0ULL;
        for (uint32_t group = 0U; group < groupCount; ++group) {
            if (group == sourceGroup) {
                continue;
            }
            const uint64_t occupied = group == ownerGroup ? duplicate : coverage;
            if ((occupied & (1ULL << group)) == 0ULL) {
                mask |= GroupRankMask(size, group);
            }
        }
        return mask;
    }

    __aicore__ inline void ResolveAllDestinationMaskLevel(
        uint32_t source,
        uint32_t logicalOwner,
        uint32_t level,
        uint64_t coverage,
        uint64_t duplicate,
        uint64_t valid,
        uint64_t &unresolved,
        uint64_t &move) const
    {
        const uint64_t novel = NovelRankMask(
            source, logicalOwner, level, coverage, duplicate);
        if ((novel & (1ULL << logicalOwner)) != 0ULL) {
            move |= unresolved & ~novel & valid;
            unresolved &= novel;
        } else {
            unresolved &= ~novel & valid;
        }
    }

    __aicore__ inline void BuildAllDestinationMasksForLogical(
        uint32_t logical,
        uint32_t recordCount,
        uint32_t block)
    {
        const uint32_t logicalOwner = OwnerRank(logical);
        const uint64_t validRanks = tiling.epSize == kMaxRanks
            ? ~0ULL
            : (1ULL << tiling.epSize) - 1ULL;
        const uint64_t validDestinations = validRanks & ~(1ULL << logicalOwner);
        const uint32_t start = SharedLogicalRecordStart(logical);
        for (uint32_t first = block * 8U; first < recordCount;
             first += tiling.blockCount * 8U) {
            const uint32_t end = first + 8U < recordCount ? first + 8U : recordCount;
            for (uint32_t record = first; record < end; ++record) {
                const uint64_t position = BaselineRouteIndex(logical, record);
                const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
                const uint32_t source = static_cast<uint32_t>(sampleSourcesGm.GetValue(token));
                uint64_t coverage[kMaxLevels];
                uint64_t duplicate[kMaxLevels];
                for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
                    coverage[level] = static_cast<uint64_t>(
                        intWorkspaceGm.GetValue(TokenCoverageIndex(token, level, false)));
                    duplicate[level] = static_cast<uint64_t>(
                        intWorkspaceGm.GetValue(TokenCoverageIndex(token, level, true)));
                }

                uint64_t unresolved = validDestinations;
                uint64_t move = 0ULL;
                for (uint32_t cursor = tiling.numLevels - 1U; cursor > 0U; --cursor) {
                    const uint32_t level = cursor - 1U;
                    ResolveAllDestinationMaskLevel(
                        source,
                        logicalOwner,
                        level,
                        coverage[level],
                        duplicate[level],
                        validDestinations,
                        unresolved,
                        move);
                }
                const uint32_t finalLevel = tiling.numLevels - 1U;
                ResolveAllDestinationMaskLevel(
                    source,
                    logicalOwner,
                    finalLevel,
                    coverage[finalLevel],
                    duplicate[finalLevel],
                    validDestinations,
                    unresolved,
                    move);

                const uint32_t sharedRecord = start + record;
                intWorkspaceGm.SetValue(
                    AllDestinationMoveMasksOffset() + sharedRecord,
                    static_cast<int64_t>(move));
                intWorkspaceGm.SetValue(
                    AllDestinationTieMasksOffset() + sharedRecord,
                    static_cast<int64_t>(unresolved));
            }
        }
    }

    __aicore__ inline void BuildAllDestinationMasksParallel(uint32_t block)
    {
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const uint32_t recordCount = SharedLogicalRecordCount(logical);
            if (CanUseAllDestinationMasks(logical, recordCount)) {
                BuildAllDestinationMasksForLogical(logical, recordCount, block);
            }
        }
    }

    __aicore__ inline uint32_t FirstRank(uint64_t mask) const
    {
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            if ((mask & (1ULL << rank)) != 0ULL) {
                return rank;
            }
        }
        return 0U;
    }

    __aicore__ inline bool SingleRank(uint64_t mask) const
    {
        return mask != 0ULL && (mask & (mask - 1ULL)) == 0ULL;
    }

    __aicore__ inline int64_t RouteHash(uint32_t token, uint32_t logical) const
    {
        const int64_t ordinal = sampleOrdinalsGm.GetValue(token);
        uint64_t wrapped = static_cast<uint64_t>(ordinal) * 1000003ULL
            + static_cast<uint64_t>(logical) * 65537ULL + static_cast<uint64_t>(tiling.step) * 131ULL
            + static_cast<uint64_t>(tiling.layerSeed) * 17ULL;
        wrapped = wrapped * 48271ULL + 1ULL;
        int64_t value = 0;
        if (wrapped == (1ULL << 63U)) {
            value = -9223372036854775807LL - 1LL;
        } else if ((wrapped & (1ULL << 63U)) != 0ULL) {
            value = -static_cast<int64_t>((~wrapped) + 1ULL);
        } else {
            value = static_cast<int64_t>(wrapped);
        }
        value %= 2147483647LL;
        if (value < 0) {
            value += 2147483647LL;
        }
        return value % 1048573LL;
    }

    __aicore__ inline uint32_t BucketHash(int64_t key, uint64_t mask) const
    {
        uint64_t value = static_cast<uint64_t>(key) * 11400714819323198485ULL;
        value ^= mask + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
        return static_cast<uint32_t>(value) & (activeHashCapacity - 1U);
    }

    __aicore__ inline uint32_t FindOrCreateBucket(
        int64_t key,
        uint64_t mask,
        uint32_t &bucketCount)
    {
        uint32_t index = BucketHash(key, mask);
        while (true) {
            const int64_t current = intWorkspaceGm.GetValue(tableKeyOffset + index);
            if (current == -1) {
                const uint32_t bucket = bucketCount++;
                intWorkspaceGm.SetValue(tableKeyOffset + index, key);
                intWorkspaceGm.SetValue(tableMaskOffset + index, static_cast<int64_t>(mask));
                intWorkspaceGm.SetValue(tableBucketOffset + index, bucket);
                intWorkspaceGm.SetValue(bucketKeyOffset + bucket, key);
                intWorkspaceGm.SetValue(bucketMaskOffset + bucket, static_cast<int64_t>(mask));
                intWorkspaceGm.SetValue(bucketProcessedOffset + bucket, 0);
                floatWorkspaceGm.SetValue(bucketTotalOffset + bucket, 0.0F);
                return bucket;
            }
            if (current == key
                && static_cast<uint64_t>(intWorkspaceGm.GetValue(tableMaskOffset + index)) == mask) {
                return static_cast<uint32_t>(intWorkspaceGm.GetValue(tableBucketOffset + index));
            }
            index = (index + 1U) & (activeHashCapacity - 1U);
        }
    }

    __aicore__ inline void ResetProjection()
    {
        activeHashCapacity = tiling.hashCapacity;
        for (uint32_t index = 0; index < activeHashCapacity; ++index) {
            intWorkspaceGm.SetValue(tableKeyOffset + index, -1);
        }
        for (uint32_t index = 0; index < tiling.distributionSize; ++index) {
            intWorkspaceGm.SetValue(distributionOffset + index, 0);
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            floatWorkspaceGm.SetValue(rankLoadOffset + rank, 0.0F);
        }
    }

    __aicore__ inline uint32_t BuildRecords(uint32_t &bucketCount)
    {
        uint32_t recordCount = 0U;
        bucketCount = 0U;
        for (uint32_t token = 0; token < tiling.numSamples; ++token) {
            intWorkspaceGm.SetValue(tokenStartOffset + token, recordCount);
            const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
            for (uint32_t position = 0; position < tiling.topK; ++position) {
                const uint64_t routeIndex = routeOffset + position;
                const int64_t multiplicity = sampleMultiplicityGm.GetValue(routeIndex);
                if (multiplicity <= 0) {
                    continue;
                }
                const uint32_t logical = static_cast<uint32_t>(sampleRoutesGm.GetValue(routeIndex));
                const uint64_t mask = EligibleMask(
                    token, OwnerRank(logical), LogicalCopyMask(logical));
                intWorkspaceGm.SetValue(recordPositionOffset + recordCount, routeIndex);
                intWorkspaceGm.SetValue(recordMaskOffset + recordCount, static_cast<int64_t>(mask));
                intWorkspaceGm.SetValue(recordHashOffset + recordCount, RouteHash(token, logical));
                intWorkspaceGm.SetValue(recordBucketOffset + recordCount, -1);
                const float units = static_cast<float>(multiplicity) * sampleWeightsGm.GetValue(token);
                if (SingleRank(mask)) {
                    const uint32_t rank = FirstRank(mask);
                    intWorkspaceGm.SetValue(recordChosenOffset + recordCount, rank);
                    floatWorkspaceGm.SetValue(
                        rankLoadOffset + rank,
                        floatWorkspaceGm.GetValue(rankLoadOffset + rank) + units);
                } else {
                    intWorkspaceGm.SetValue(recordChosenOffset + recordCount, -1);
                    const int64_t source = sampleSourcesGm.GetValue(token);
                    const int64_t key = source * static_cast<int64_t>(tiling.numExperts) + logical;
                    const uint32_t bucket = FindOrCreateBucket(key, mask, bucketCount);
                    intWorkspaceGm.SetValue(recordBucketOffset + recordCount, bucket);
                    floatWorkspaceGm.SetValue(
                        bucketTotalOffset + bucket,
                        floatWorkspaceGm.GetValue(bucketTotalOffset + bucket) + units);
                }
                ++recordCount;
            }
        }
        intWorkspaceGm.SetValue(tokenStartOffset + tiling.numSamples, recordCount);
        return recordCount;
    }

    __aicore__ inline void BuildBucketRecordLists(uint32_t bucketCount, uint32_t recordCount)
    {
        // The open-addressing table is no longer needed once every record has
        // found its bucket.  Reuse its first bucketCount entries as list heads
        // and the per-record mask storage as next pointers.  Building the
        // lists backwards preserves ascending record order for exact
        // (hash, ordinal) ties, matching the former full-record scan.
        for (uint32_t bucket = 0; bucket < bucketCount; ++bucket) {
            intWorkspaceGm.SetValue(tableKeyOffset + bucket, -1);
        }
        for (uint32_t cursor = recordCount; cursor > 0U; --cursor) {
            const uint32_t record = cursor - 1U;
            const int64_t bucket = intWorkspaceGm.GetValue(recordBucketOffset + record);
            if (bucket < 0) {
                continue;
            }
            const uint32_t index = static_cast<uint32_t>(bucket);
            intWorkspaceGm.SetValue(
                recordMaskOffset + record,
                intWorkspaceGm.GetValue(tableKeyOffset + index));
            intWorkspaceGm.SetValue(tableKeyOffset + index, record);
        }
    }

    __aicore__ inline void BuildLogicalRecordLists(uint32_t recordCount)
    {
        for (uint32_t logical = 0; logical < tiling.numExperts; ++logical) {
            intWorkspaceGm.SetValue(BaselineDistributionOffset() + logical, -1);
        }
        for (uint32_t cursor = recordCount; cursor > 0U; --cursor) {
            const uint32_t record = cursor - 1U;
            const uint64_t position = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(BaselineRecordPositionOffset() + record));
            const uint32_t logical = static_cast<uint32_t>(sampleRoutesGm.GetValue(position));
            const int64_t head = intWorkspaceGm.GetValue(BaselineDistributionOffset() + logical);
            intWorkspaceGm.SetValue(BaselineRecordNextOffset() + record, head);
            intWorkspaceGm.SetValue(BaselineDistributionOffset() + logical, record);
        }
    }

    __aicore__ inline bool RecordLessEqual(uint32_t lhs, uint32_t rhs) const
    {
        const int64_t lhsHash = intWorkspaceGm.GetValue(BaselineRecordHashOffset() + lhs);
        const int64_t rhsHash = intWorkspaceGm.GetValue(BaselineRecordHashOffset() + rhs);
        if (lhsHash != rhsHash) {
            return lhsHash < rhsHash;
        }
        const uint64_t lhsPosition = static_cast<uint64_t>(
            intWorkspaceGm.GetValue(BaselineRecordPositionOffset() + lhs));
        const uint64_t rhsPosition = static_cast<uint64_t>(
            intWorkspaceGm.GetValue(BaselineRecordPositionOffset() + rhs));
        const int64_t lhsOrdinal = sampleOrdinalsGm.GetValue(lhsPosition / tiling.topK);
        const int64_t rhsOrdinal = sampleOrdinalsGm.GetValue(rhsPosition / tiling.topK);
        return lhsOrdinal != rhsOrdinal ? lhsOrdinal < rhsOrdinal : lhs <= rhs;
    }

    __aicore__ inline int64_t SortedRecordValue(bool scratch, uint32_t index) const
    {
        const uint64_t offset = scratch ? recordPositionOffset : tokenStartOffset;
        return intWorkspaceGm.GetValue(offset + index);
    }

    __aicore__ inline void SetSortedRecordValue(bool scratch, uint32_t index, int64_t value)
    {
        const uint64_t offset = scratch ? recordPositionOffset : tokenStartOffset;
        intWorkspaceGm.SetValue(offset + index, value);
    }

    __aicore__ inline uint32_t PrepareSortedLogical(uint32_t logical)
    {
        uint32_t count = 0U;
        int64_t cursor = LogicalHead(logical);
        while (cursor >= 0) {
            intWorkspaceGm.SetValue(tokenStartOffset + count, cursor);
            ++count;
            cursor = intWorkspaceGm.GetValue(
                BaselineRecordNextOffset() + static_cast<uint32_t>(cursor));
        }
        bool sourceScratch = false;
        for (uint32_t width = 1U; width < count; width <<= 1U) {
            const bool destinationScratch = !sourceScratch;
            for (uint32_t left = 0U; left < count; left += 2U * width) {
                const uint32_t middle = left + width < count ? left + width : count;
                const uint32_t right = left + 2U * width < count ? left + 2U * width : count;
                uint32_t lhs = left;
                uint32_t rhs = middle;
                uint32_t output = left;
                while (lhs < middle || rhs < right) {
                    bool takeLeft = rhs >= right;
                    if (lhs < middle && rhs < right) {
                        takeLeft = RecordLessEqual(
                            static_cast<uint32_t>(SortedRecordValue(sourceScratch, lhs)),
                            static_cast<uint32_t>(SortedRecordValue(sourceScratch, rhs)));
                    }
                    uint32_t cursor = rhs;
                    if (takeLeft) {
                        cursor = lhs;
                        ++lhs;
                    } else {
                        ++rhs;
                    }
                    SetSortedRecordValue(
                        destinationScratch,
                        output++,
                        SortedRecordValue(sourceScratch, cursor));
                }
            }
            sourceScratch = destinationScratch;
        }
        if (sourceScratch) {
            for (uint32_t index = 0U; index < count; ++index) {
                SetSortedRecordValue(false, index, SortedRecordValue(true, index));
            }
        }
        return count;
    }

    __aicore__ inline uint32_t CountLogicalRecords(uint32_t logical) const
    {
        uint32_t count = 0U;
        int64_t cursor = LogicalHead(logical);
        while (cursor >= 0) {
            ++count;
            cursor = intWorkspaceGm.GetValue(
                BaselineRecordNextOffset() + static_cast<uint32_t>(cursor));
        }
        return count;
    }

    __aicore__ inline bool LogicalRequired(uint32_t logical) const
    {
        if (candidateExpertsGm.GetValue(logical) != 0) {
            return true;
        }
        const uint32_t removeActions = tiling.epSize * tiling.redundantSlotsPerRank;
        for (uint32_t action = 0U; action < removeActions; ++action) {
            const uint32_t rank = action / tiling.redundantSlotsPerRank;
            const int64_t slot = redundantSlotsGm.GetValue(action);
            if (RemovableSlot(rank, slot)
                && slotToLogicalGm.GetValue(static_cast<uint32_t>(slot)) == static_cast<int64_t>(logical)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void BuildSharedLogicalRecordCounts(uint32_t block)
    {
        const uint32_t groups = (tiling.numExperts + 7U) / 8U;
        for (uint32_t group = block; group < groups; group += tiling.blockCount) {
            const uint32_t first = group * 8U;
            const uint32_t end = first + 8U < tiling.numExperts ? first + 8U : tiling.numExperts;
            for (uint32_t logical = first; logical < end; ++logical) {
                const uint32_t count = LogicalRequired(logical) ? CountLogicalRecords(logical) : 0U;
                intWorkspaceGm.SetValue(SharedLogicalEntryOffset(logical), count);
            }
        }
    }

    __aicore__ inline void BuildSharedLogicalRecordOffsets()
    {
        uint32_t output = 0U;
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const uint32_t count = static_cast<uint32_t>(
                intWorkspaceGm.GetValue(SharedLogicalEntryOffset(logical)));
            output = AlignRecordCacheLine(output);
            const uint64_t entry = (static_cast<uint64_t>(count) << 32U) | output;
            intWorkspaceGm.SetValue(SharedLogicalEntryOffset(logical), static_cast<int64_t>(entry));
            output += count;
        }
        intWorkspaceGm.SetValue(
            SharedLogicalEntryOffset(tiling.numExperts),
            AlignRecordCacheLine(output));
    }

    __aicore__ inline void BuildDirectLogicalRecordOffsets()
    {
        uint32_t finalOutput = 0U;
        uint32_t rawOutput = 0U;
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const bool required = LogicalRequired(logical);
            finalOutput = AlignRecordCacheLine(finalOutput);
            const uint32_t finalStart = finalOutput;
            uint32_t totalCount = 0U;
            float sampleLoad = 0.0F;
            for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
                const uint64_t countOffset = DirectBlockCountOffset(block, logical);
                const uint32_t count = required
                    ? static_cast<uint32_t>(intWorkspaceGm.GetValue(countOffset))
                    : 0U;
                rawOutput = AlignRecordCacheLine(rawOutput);
                intWorkspaceGm.SetValue(
                    DirectBlockStartOffset(block, logical),
                    static_cast<int64_t>(rawOutput));
                intWorkspaceGm.SetValue(
                    DirectBlockCursorOffset(block, logical),
                    static_cast<int64_t>(rawOutput));
                rawOutput += count;
                totalCount += count;
                if (required) {
                    sampleLoad += floatWorkspaceGm.GetValue(DirectBlockLoadOffset(block, logical));
                }
            }
            finalOutput += totalCount;
            const uint64_t metadata = SharedLogicalEntryOffset(logical);
            for (uint32_t index = 0U; index < 8U; ++index) {
                intWorkspaceGm.SetValue(metadata + index, 0LL);
            }
            const uint64_t entry = (static_cast<uint64_t>(totalCount) << 32U) | finalStart;
            intWorkspaceGm.SetValue(metadata, static_cast<int64_t>(entry));
            const uint64_t expertLoad = BaselineExpertLoadOffset(logical);
            for (uint32_t index = 0U; index < 16U; ++index) {
                floatWorkspaceGm.SetValue(expertLoad + index, index == 0U ? sampleLoad : 0.0F);
            }
        }
        intWorkspaceGm.SetValue(
            SharedLogicalEntryOffset(tiling.numExperts),
            AlignRecordCacheLine(finalOutput));
    }

    __aicore__ inline void BuildSharedSortedLogicalRecords(uint32_t block)
    {
        uint32_t workIndex = 0U;
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const uint32_t count = SharedLogicalRecordCount(logical);
            if (count == 0U) {
                continue;
            }
            const uint32_t assigned = workIndex++;
            if (assigned % tiling.blockCount != block) {
                continue;
            }
            uint32_t output = SharedLogicalRecordStart(logical);
            if (count <= tiling.actionMaxRecords) {
                const uint32_t sorted = PrepareSortedLogical(logical);
                for (uint32_t index = 0U; index < sorted; ++index) {
                    intWorkspaceGm.SetValue(
                        SharedSortedRecordsOffset() + output++,
                        SortedRecordValue(false, index));
                }
                continue;
            }
            // Non-canonical multiplicity can expose more than one positive
            // record for the same (token, logical) pair.  Preserve memory
            // safety and the shared index shape; action scoring rejects this
            // edge instead of overrunning its token-bounded private scratch.
            int64_t cursor = LogicalHead(logical);
            while (cursor >= 0) {
                intWorkspaceGm.SetValue(SharedSortedRecordsOffset() + output++, cursor);
                cursor = intWorkspaceGm.GetValue(
                    BaselineRecordNextOffset() + static_cast<uint32_t>(cursor));
            }
        }
    }

    __aicore__ inline int64_t DirectSortedRoute(bool secondary, uint32_t index) const
    {
        const uint64_t offset = secondary ? recordMaskOffset : recordPositionOffset;
        return intWorkspaceGm.GetValue(offset + index);
    }

    __aicore__ inline int64_t DirectSortedHash(bool secondary, uint32_t index) const
    {
        const uint64_t offset = secondary ? recordChosenOffset : recordHashOffset;
        return intWorkspaceGm.GetValue(offset + index);
    }

    __aicore__ inline void SetDirectSortedPair(
        bool secondary,
        uint32_t index,
        int64_t routeIndex,
        int64_t hash)
    {
        const uint64_t routeOffset = secondary ? recordMaskOffset : recordPositionOffset;
        const uint64_t hashOffset = secondary ? recordChosenOffset : recordHashOffset;
        intWorkspaceGm.SetValue(routeOffset + index, routeIndex);
        intWorkspaceGm.SetValue(hashOffset + index, hash);
    }

    __aicore__ inline bool DirectPairLessEqual(bool secondary, uint32_t lhs, uint32_t rhs) const
    {
        const int64_t lhsHash = DirectSortedHash(secondary, lhs);
        const int64_t rhsHash = DirectSortedHash(secondary, rhs);
        if (lhsHash != rhsHash) {
            return lhsHash < rhsHash;
        }
        const uint64_t lhsRoute = static_cast<uint64_t>(DirectSortedRoute(secondary, lhs));
        const uint64_t rhsRoute = static_cast<uint64_t>(DirectSortedRoute(secondary, rhs));
        const int64_t lhsOrdinal = sampleOrdinalsGm.GetValue(lhsRoute / tiling.topK);
        const int64_t rhsOrdinal = sampleOrdinalsGm.GetValue(rhsRoute / tiling.topK);
        return lhsOrdinal != rhsOrdinal ? lhsOrdinal < rhsOrdinal : lhsRoute <= rhsRoute;
    }

    __aicore__ inline void PrepareDirectSortedLogicalScalar(uint32_t logical, uint32_t count)
    {
        uint32_t blockStarts[kMaxRanks];
        uint32_t blockCounts[kMaxRanks];
        for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
            blockStarts[block] = static_cast<uint32_t>(
                intWorkspaceGm.GetValue(DirectBlockStartOffset(block, logical)));
            blockCounts[block] = static_cast<uint32_t>(
                intWorkspaceGm.GetValue(DirectBlockCountOffset(block, logical)));
        }
        uint32_t cursor = 0U;
        for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
            const uint32_t blockStart = blockStarts[block];
            const uint32_t blockCount = blockCounts[block];
            for (uint32_t index = 0U; index < blockCount && cursor < count; ++index) {
                SetDirectSortedPair(
                    false,
                    cursor++,
                    intWorkspaceGm.GetValue(DirectRawRecordsOffset() + blockStart + index),
                    intWorkspaceGm.GetValue(DirectRawHashesOffset() + blockStart + index));
            }
        }
        if (cursor != count) {
            return;
        }
        bool sourceSecondary = false;
        for (uint32_t width = 1U; width < count; width <<= 1U) {
            const bool destinationSecondary = !sourceSecondary;
            for (uint32_t left = 0U; left < count; left += 2U * width) {
                const uint32_t middle = left + width < count ? left + width : count;
                const uint32_t right = left + 2U * width < count ? left + 2U * width : count;
                uint32_t lhs = left;
                uint32_t rhs = middle;
                uint32_t output = left;
                while (lhs < middle || rhs < right) {
                    bool takeLeft = rhs >= right;
                    if (lhs < middle && rhs < right) {
                        takeLeft = DirectPairLessEqual(sourceSecondary, lhs, rhs);
                    }
                    const uint32_t selected = takeLeft ? lhs++ : rhs++;
                    SetDirectSortedPair(
                        destinationSecondary,
                        output++,
                        DirectSortedRoute(sourceSecondary, selected),
                        DirectSortedHash(sourceSecondary, selected));
                }
            }
            sourceSecondary = destinationSecondary;
        }
        const uint32_t start = SharedLogicalRecordStart(logical);
        for (uint32_t index = 0U; index < count; ++index) {
            intWorkspaceGm.SetValue(
                SharedSortedRecordsOffset() + start + index,
                DirectSortedRoute(sourceSecondary, index));
            intWorkspaceGm.SetValue(
                SharedSortedHashesOffset() + start + index,
                DirectSortedHash(sourceSecondary, index));
        }
    }

    __aicore__ inline bool DirectRouteLess(uint32_t lhs, uint32_t rhs) const
    {
        const int64_t lhsOrdinal = sampleOrdinalsGm.GetValue(lhs / tiling.topK);
        const int64_t rhsOrdinal = sampleOrdinalsGm.GetValue(rhs / tiling.topK);
        return lhsOrdinal != rhsOrdinal ? lhsOrdinal < rhsOrdinal : lhs < rhs;
    }

    __aicore__ inline bool PrepareDirectHashSortedLogical(uint32_t logical, uint32_t count)
    {
        if (count == 0U || count > kHardwareSortMaxRecords) {
            return false;
        }

        // The first 16 KiB of scratch holds score bits and the second 16 KiB
        // holds uint32 payloads.  Once Sort32 completes, the whole 32 KiB is
        // reused as MrgSort's ping-pong destination.  Together with the packed
        // buffer this path consumes 64 KiB of per-core UB.
        AscendC::LocalTensor<float> scratch = hashSortScratchBuf.Get<float>();
        AscendC::LocalTensor<float> scores = scratch;
        AscendC::LocalTensor<uint32_t> scoreBits = scores.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<uint32_t> payloads =
            scratch[kHardwareSortCapacity].ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> packed = hashSortPackedBuf.Get<float>();

        uint32_t cursor = 0U;
        for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
            const uint32_t blockStart = static_cast<uint32_t>(
                intWorkspaceGm.GetValue(DirectBlockStartOffset(block, logical)));
            const uint32_t blockCount = static_cast<uint32_t>(
                intWorkspaceGm.GetValue(DirectBlockCountOffset(block, logical)));
            for (uint32_t index = 0U; index < blockCount && cursor < count; ++index) {
                const int64_t routeIndex = intWorkspaceGm.GetValue(
                    DirectRawRecordsOffset() + blockStart + index);
                const int64_t hash = intWorkspaceGm.GetValue(
                    DirectRawHashesOffset() + blockStart + index);
                if (routeIndex < 0 || static_cast<uint64_t>(routeIndex) > 0xffffffffULL
                    || hash < 0 || hash > static_cast<int64_t>(kRouteHashMax)) {
                    return false;
                }
                const uint32_t score = kRouteHashMax - static_cast<uint32_t>(hash);
                // 0x4b000000 + score encodes exactly 2^23 + score.  A vector
                // subtraction below avoids the unsupported scalar int-to-float
                // conversion and produces every integer score exactly.
                scoreBits.SetValue(cursor, kFloatIntegerBiasBits + score);
                payloads.SetValue(cursor, static_cast<uint32_t>(routeIndex));
                ++cursor;
            }
        }
        if (cursor != count) {
            return false;
        }

        const uint32_t paddedCount = (count + 31U) / 32U * 32U;
        for (uint32_t index = count; index < paddedCount; ++index) {
            // 0x4afffffe is 2^23 - 1, hence the vector subtraction maps it
            // to -1.0F, below every valid score in [0, 1048572].
            scoreBits.SetValue(index, kPaddedScoreBiasBits);
            payloads.SetValue(index, 0xffffffffU);
        }
        AscendC::Adds<float>(scores, scores, -kFloatIntegerBias, paddedCount);
        AscendC::Sort32<float>(
            packed, scores, payloads, static_cast<int32_t>(paddedCount / 32U));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> source = packed;
        AscendC::LocalTensor<float> destination = scratch;
        for (uint32_t runWidth = 32U; runWidth < count; runWidth *= 4U) {
            const uint32_t groupWidth = 4U * runWidth;
            for (uint32_t groupStart = 0U; groupStart < count; groupStart += groupWidth) {
                const uint32_t remaining = count - groupStart;
                const uint32_t listCount = (remaining + runWidth - 1U) / runWidth < 4U
                    ? (remaining + runWidth - 1U) / runWidth
                    : 4U;
                uint16_t lengths[4] = {0U, 0U, 0U, 0U};
                for (uint32_t list = 0U; list < listCount; ++list) {
                    const uint32_t listStart = groupStart + list * runWidth;
                    const uint32_t listRemaining = count - listStart;
                    lengths[list] = static_cast<uint16_t>(
                        listRemaining < runWidth ? listRemaining : runWidth);
                }
                if (listCount == 1U) {
                    // A trailing singleton is already sorted.  It is copied
                    // only after prior vector merges in this stage complete.
                    AscendC::PipeBarrier<PIPE_V>();
                    const uint32_t words = 2U * static_cast<uint32_t>(lengths[0]);
                    for (uint32_t word = 0U; word < words; ++word) {
                        destination.SetValue(
                            2U * groupStart + word,
                            source.GetValue(2U * groupStart + word));
                    }
                    continue;
                }

                const uint32_t second = groupStart + runWidth;
                const uint32_t third = listCount > 2U ? groupStart + 2U * runWidth : groupStart;
                const uint32_t fourth = listCount > 3U ? groupStart + 3U * runWidth : groupStart;
                const AscendC::MrgSortSrcList<float> sources(
                    source[2U * groupStart],
                    source[2U * second],
                    source[2U * third],
                    source[2U * fourth]);
                const AscendC::MrgSort4Info params(
                    lengths,
                    false,
                    static_cast<uint16_t>((1U << listCount) - 1U),
                    1U);
                AscendC::MrgSort<float>(destination[2U * groupStart], sources, params);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> previous = source;
            source = destination;
            destination = previous;
        }

        // Sort32/MrgSort are stable and order by descending score, which is
        // ascending hash.  Repair only equal-hash runs using the full int64
        // sample ordinal and route index so no secondary-key precision is lost.
        AscendC::LocalTensor<uint32_t> sortedWords = source.ReinterpretCast<uint32_t>();
        uint32_t runStart = 0U;
        while (runStart < count) {
            const uint32_t scoreWord = sortedWords.GetValue(2U * runStart);
            uint32_t runEnd = runStart + 1U;
            while (runEnd < count && sortedWords.GetValue(2U * runEnd) == scoreWord) {
                ++runEnd;
            }
            for (uint32_t index = runStart + 1U; index < runEnd; ++index) {
                const uint32_t routeIndex = sortedWords.GetValue(2U * index + 1U);
                uint32_t insertion = index;
                while (insertion > runStart) {
                    const uint32_t previous = sortedWords.GetValue(2U * (insertion - 1U) + 1U);
                    if (!DirectRouteLess(routeIndex, previous)) {
                        break;
                    }
                    sortedWords.SetValue(2U * insertion + 1U, previous);
                    --insertion;
                }
                sortedWords.SetValue(2U * insertion + 1U, routeIndex);
            }
            runStart = runEnd;
        }

        const uint32_t outputStart = SharedLogicalRecordStart(logical);
        for (uint32_t index = 0U; index < count; ++index) {
            const uint32_t routeIndex = sortedWords.GetValue(2U * index + 1U);
            intWorkspaceGm.SetValue(
                SharedSortedRecordsOffset() + outputStart + index,
                static_cast<int64_t>(routeIndex));
            intWorkspaceGm.SetValue(
                SharedSortedHashesOffset() + outputStart + index,
                RouteHash(routeIndex / tiling.topK, logical));
        }
        return true;
    }

    __aicore__ inline void PrepareDirectSortedLogical(uint32_t logical, uint32_t count)
    {
        if (!PrepareDirectHashSortedLogical(logical, count)) {
            PrepareDirectSortedLogicalScalar(logical, count);
        }
    }

    __aicore__ inline void BuildDirectLogicalRecords(uint32_t block)
    {
        const uint32_t tokenBegin = BlockTokenBegin(block);
        const uint32_t tokenEnd = BlockTokenEnd(block);
        for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
            const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
            for (uint32_t position = 0U; position < tiling.topK; ++position) {
                const uint64_t routeIndex = routeOffset + position;
                if (sampleMultiplicityGm.GetValue(routeIndex) <= 0) {
                    continue;
                }
                const uint32_t logical = static_cast<uint32_t>(sampleRoutesGm.GetValue(routeIndex));
                if (SharedLogicalRecordCount(logical) == 0U) {
                    continue;
                }
                const uint64_t cursorOffset = DirectBlockCursorOffset(block, logical);
                const uint32_t cursor = static_cast<uint32_t>(
                    intWorkspaceGm.GetValue(cursorOffset));
                intWorkspaceGm.SetValue(DirectRawRecordsOffset() + cursor, routeIndex);
                intWorkspaceGm.SetValue(
                    DirectRawHashesOffset() + cursor,
                    RouteHash(token, logical));
                intWorkspaceGm.SetValue(cursorOffset, static_cast<int64_t>(cursor + 1U));
            }
        }
    }

    __aicore__ inline void BuildDirectSortedLogicalRecords(uint32_t block)
    {
        uint32_t workIndex = 0U;
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const uint32_t count = SharedLogicalRecordCount(logical);
            if (count == 0U) {
                continue;
            }
            const uint32_t assigned = workIndex++;
            if (assigned % tiling.blockCount != block || count > tiling.actionMaxRecords) {
                continue;
            }
            PrepareDirectSortedLogical(logical, count);
        }
    }

    __aicore__ inline bool MaskLexLess(uint64_t lhs, uint64_t rhs) const
    {
        uint32_t lhsCursor = 0U;
        uint32_t rhsCursor = 0U;
        while (true) {
            while (lhsCursor < tiling.epSize && (lhs & (1ULL << lhsCursor)) == 0ULL) {
                ++lhsCursor;
            }
            while (rhsCursor < tiling.epSize && (rhs & (1ULL << rhsCursor)) == 0ULL) {
                ++rhsCursor;
            }
            if (lhsCursor == tiling.epSize || rhsCursor == tiling.epSize) {
                return lhsCursor == tiling.epSize && rhsCursor != tiling.epSize;
            }
            if (lhsCursor != rhsCursor) {
                return lhsCursor < rhsCursor;
            }
            ++lhsCursor;
            ++rhsCursor;
        }
    }

    __aicore__ inline bool BucketBetter(uint32_t candidate, int32_t best) const
    {
        if (best < 0) {
            return true;
        }
        const float candidateTotal = floatWorkspaceGm.GetValue(bucketTotalOffset + candidate);
        const float bestTotal = floatWorkspaceGm.GetValue(bucketTotalOffset + static_cast<uint32_t>(best));
        if (candidateTotal != bestTotal) {
            return candidateTotal > bestTotal;
        }
        const int64_t candidateKey = intWorkspaceGm.GetValue(bucketKeyOffset + candidate);
        const int64_t bestKey = intWorkspaceGm.GetValue(bucketKeyOffset + static_cast<uint32_t>(best));
        if (candidateKey != bestKey) {
            return candidateKey < bestKey;
        }
        return MaskLexLess(
            static_cast<uint64_t>(intWorkspaceGm.GetValue(bucketMaskOffset + candidate)),
            static_cast<uint64_t>(intWorkspaceGm.GetValue(bucketMaskOffset + static_cast<uint32_t>(best))));
    }

    __aicore__ inline int64_t RoundEven(float value) const
    {
        const int64_t lower = static_cast<int64_t>(value);
        const float fraction = value - static_cast<float>(lower);
        if (fraction < 0.5F) {
            return lower;
        }
        if (fraction > 0.5F) {
            return lower + 1LL;
        }
        return (lower & 1LL) == 0LL ? lower : lower + 1LL;
    }

    __aicore__ inline int32_t NextBucketRecord(uint32_t bucket, bool recordsSorted)
    {
        if (recordsSorted) {
            int64_t cursor = intWorkspaceGm.GetValue(tableKeyOffset + bucket);
            while (cursor >= 0) {
                const uint32_t record = static_cast<uint32_t>(cursor);
                cursor = intWorkspaceGm.GetValue(recordMaskOffset + record);
                intWorkspaceGm.SetValue(tableKeyOffset + bucket, cursor);
                if (intWorkspaceGm.GetValue(recordChosenOffset + record) < 0) {
                    return static_cast<int32_t>(record);
                }
            }
            return -1;
        }
        int32_t best = -1;
        int64_t bestHash = 0;
        int64_t bestOrdinal = 0;
        int64_t cursor = intWorkspaceGm.GetValue(tableKeyOffset + bucket);
        while (cursor >= 0) {
            const uint32_t record = static_cast<uint32_t>(cursor);
            cursor = intWorkspaceGm.GetValue(recordMaskOffset + record);
            if (intWorkspaceGm.GetValue(recordChosenOffset + record) >= 0) {
                continue;
            }
            const int64_t hash = intWorkspaceGm.GetValue(recordHashOffset + record);
            const uint64_t position = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(recordPositionOffset + record));
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const int64_t ordinal = sampleOrdinalsGm.GetValue(token);
            if (best < 0 || hash < bestHash || (hash == bestHash && ordinal < bestOrdinal)) {
                best = static_cast<int32_t>(record);
                bestHash = hash;
                bestOrdinal = ordinal;
            }
        }
        return best;
    }

    __aicore__ inline int64_t CeilNonnegative(float value) const
    {
        const int64_t lower = static_cast<int64_t>(value);
        return static_cast<float>(lower) < value ? lower + 1LL : lower;
    }

    __aicore__ inline void BuildQuotas(uint64_t mask, int64_t total, int64_t (&quotas)[kMaxRanks]) const
    {
        uint32_t destinations[kMaxRanks];
        uint32_t ordered[kMaxRanks];
        uint32_t count = 0U;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            quotas[rank] = 0LL;
            if ((mask & (1ULL << rank)) != 0ULL) {
                destinations[count] = rank;
                ordered[count] = rank;
                ++count;
            }
        }
        for (uint32_t index = 1; index < count; ++index) {
            const uint32_t value = ordered[index];
            uint32_t position = index;
            while (position > 0U) {
                const uint32_t previous = ordered[position - 1U];
                const float valueLoad = floatWorkspaceGm.GetValue(rankLoadOffset + value);
                const float previousLoad = floatWorkspaceGm.GetValue(rankLoadOffset + previous);
                if (previousLoad < valueLoad || (previousLoad == valueLoad && previous < value)) {
                    break;
                }
                ordered[position] = previous;
                --position;
            }
            ordered[position] = value;
        }
        int64_t remaining = total > 0LL ? total : 0LL;
        uint32_t active = 1U;
        while (active < count) {
            const float nextLevel = floatWorkspaceGm.GetValue(rankLoadOffset + ordered[active]);
            const float currentLevel = floatWorkspaceGm.GetValue(rankLoadOffset + ordered[active - 1U]);
            const int64_t required = CeilNonnegative(nextLevel - currentLevel) * active;
            if (required > remaining) {
                break;
            }
            if (required > 0LL) {
                const int64_t increment = required / active;
                const int64_t extra = required % active;
                for (uint32_t index = 0; index < active; ++index) {
                    quotas[ordered[index]] += increment + static_cast<int64_t>(index < extra);
                }
                remaining -= required;
            }
            ++active;
        }
        const int64_t increment = active > 0U ? remaining / active : 0LL;
        const int64_t extra = active > 0U ? remaining % active : 0LL;
        for (uint32_t index = 0; index < active; ++index) {
            quotas[ordered[index]] += increment + static_cast<int64_t>(index < extra);
        }
    }

    __aicore__ inline void AssignBucket(uint32_t bucket, bool recordsSorted)
    {
        const uint64_t mask = static_cast<uint64_t>(intWorkspaceGm.GetValue(bucketMaskOffset + bucket));
        int64_t quotas[kMaxRanks];
        BuildQuotas(mask, RoundEven(floatWorkspaceGm.GetValue(bucketTotalOffset + bucket)), quotas);
        uint32_t destinations[kMaxRanks];
        float assigned[kMaxRanks];
        uint32_t destinationCount = 0U;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            assigned[rank] = 0.0F;
            if ((mask & (1ULL << rank)) != 0ULL) {
                destinations[destinationCount++] = rank;
            }
        }

        bool allUnit = true;
        uint32_t bucketRecords = 0U;
        int64_t cursor = intWorkspaceGm.GetValue(tableKeyOffset + bucket);
        while (cursor >= 0) {
            const uint32_t record = static_cast<uint32_t>(cursor);
            cursor = intWorkspaceGm.GetValue(recordMaskOffset + record);
            const uint64_t position = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(recordPositionOffset + record));
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const float units = static_cast<float>(sampleMultiplicityGm.GetValue(position))
                * sampleWeightsGm.GetValue(token);
            allUnit = allUnit && units > 0.999999F && units < 1.000001F;
            ++bucketRecords;
        }

        int64_t quotaTotal = 0LL;
        for (uint32_t index = 0; index < destinationCount; ++index) {
            quotaTotal += quotas[destinations[index]] > 0LL ? quotas[destinations[index]] : 0LL;
        }
        uint32_t destinationIndex = 0U;
        int64_t consumed = 0LL;
        for (uint32_t order = 0; order < bucketRecords; ++order) {
            const int32_t selected = NextBucketRecord(bucket, recordsSorted);
            if (selected < 0) {
                break;
            }
            const uint32_t record = static_cast<uint32_t>(selected);
            const uint64_t position = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(recordPositionOffset + record));
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const float units = static_cast<float>(sampleMultiplicityGm.GetValue(position))
                * sampleWeightsGm.GetValue(token);
            uint32_t chosen = destinations[0];
            if (allUnit) {
                while (destinationIndex + 1U < destinationCount
                       && static_cast<int64_t>(order) >= consumed + quotas[destinations[destinationIndex]]) {
                    consumed += quotas[destinations[destinationIndex]];
                    ++destinationIndex;
                }
                chosen = quotaTotal > 0LL
                    ? destinations[destinationIndex]
                    : destinations[order % destinationCount];
            } else {
                float bestDeficit = 0.0F;
                float bestProjected = 0.0F;
                bool initialized = false;
                for (uint32_t index = 0; index < destinationCount; ++index) {
                    const uint32_t rank = destinations[index];
                    const float deficit = assigned[rank] + units - static_cast<float>(quotas[rank]);
                    const float projected = floatWorkspaceGm.GetValue(rankLoadOffset + rank)
                        + assigned[rank] + units;
                    if (!initialized || deficit < bestDeficit
                        || (deficit == bestDeficit
                            && (projected < bestProjected || (projected == bestProjected && rank < chosen)))) {
                        initialized = true;
                        bestDeficit = deficit;
                        bestProjected = projected;
                        chosen = rank;
                    }
                }
            }
            intWorkspaceGm.SetValue(recordChosenOffset + record, chosen);
            assigned[chosen] += units;
        }
        for (uint32_t index = 0; index < destinationCount; ++index) {
            const uint32_t rank = destinations[index];
            floatWorkspaceGm.SetValue(
                rankLoadOffset + rank,
                floatWorkspaceGm.GetValue(rankLoadOffset + rank) + assigned[rank]);
        }
    }

    __aicore__ inline void ProcessBuckets(uint32_t bucketCount, bool recordsSorted)
    {
        for (uint32_t processed = 0; processed < bucketCount; ++processed) {
            int32_t best = -1;
            for (uint32_t bucket = 0; bucket < bucketCount; ++bucket) {
                if (intWorkspaceGm.GetValue(bucketProcessedOffset + bucket) == 0
                    && BucketBetter(bucket, best)) {
                    best = static_cast<int32_t>(bucket);
                }
            }
            if (best < 0) {
                break;
            }
            const uint32_t bucket = static_cast<uint32_t>(best);
            intWorkspaceGm.SetValue(bucketProcessedOffset + bucket, 1);
            AssignBucket(bucket, recordsSorted);
        }
    }

    __aicore__ inline void ComputeStats(
        uint32_t recordCount,
        float (&groupCounts)[kMaxGroups],
        float (&assignmentLoads)[kMaxRanks])
    {
        for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
            groupCounts[group] = 0.0F;
        }
        for (uint32_t token = 0; token < tiling.numSamples; ++token) {
            uint32_t hits[kMaxGroups];
            for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
                hits[group] = 0U;
            }
            const uint32_t begin = static_cast<uint32_t>(intWorkspaceGm.GetValue(tokenStartOffset + token));
            const uint32_t end = static_cast<uint32_t>(intWorkspaceGm.GetValue(tokenStartOffset + token + 1U));
            for (uint32_t record = begin; record < end; ++record) {
                const uint32_t rank = static_cast<uint32_t>(intWorkspaceGm.GetValue(recordChosenOffset + record));
                for (uint32_t level = 0; level < tiling.numLevels; ++level) {
                    hits[LevelOffset(level) + rank / LevelSize(level)] = 1U;
                }
            }
            const float weight = sampleWeightsGm.GetValue(token);
            for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
                groupCounts[group] += hits[group] != 0U ? weight : 0.0F;
            }
        }

        for (uint32_t record = 0; record < recordCount; ++record) {
            const uint64_t position = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(recordPositionOffset + record));
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const uint32_t source = static_cast<uint32_t>(sampleSourcesGm.GetValue(token));
            const uint32_t logical = static_cast<uint32_t>(sampleRoutesGm.GetValue(position));
            const uint32_t rank = static_cast<uint32_t>(intWorkspaceGm.GetValue(recordChosenOffset + record));
            const uint32_t index = (source * tiling.numExperts + logical) * tiling.epSize + rank;
            intWorkspaceGm.SetValue(
                distributionOffset + index,
                intWorkspaceGm.GetValue(distributionOffset + index)
                    + sampleMultiplicityGm.GetValue(position));
        }

        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            assignmentLoads[rank] = 0.0F;
        }
        for (uint32_t source = 0; source < tiling.epSize; ++source) {
            for (uint32_t logical = 0; logical < tiling.numExperts; ++logical) {
                const uint32_t base = (source * tiling.numExperts + logical) * tiling.epSize;
                int64_t total = 0LL;
                for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                    total += intWorkspaceGm.GetValue(distributionOffset + base + rank);
                }
                const int64_t exact = assignmentCountsGm.GetValue(source * tiling.numExperts + logical);
                if (total <= 0LL || exact <= 0LL) {
                    continue;
                }
                int64_t rounded[kMaxRanks];
                float fractions[kMaxRanks];
                uint32_t extra[kMaxRanks];
                int64_t roundedTotal = 0LL;
                for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                    const int64_t count = intWorkspaceGm.GetValue(distributionOffset + base + rank);
                    const float raw = static_cast<float>(count) * static_cast<float>(exact)
                        / static_cast<float>(total);
                    rounded[rank] = static_cast<int64_t>(raw);
                    fractions[rank] = raw - static_cast<float>(rounded[rank]);
                    extra[rank] = 0U;
                    roundedTotal += rounded[rank];
                }
                int64_t remainder = exact - roundedTotal;
                while (remainder > 0LL) {
                    int32_t best = -1;
                    for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                        if (extra[rank] != 0U) {
                            continue;
                        }
                        if (best < 0 || fractions[rank] > fractions[static_cast<uint32_t>(best)]
                            || (fractions[rank] == fractions[static_cast<uint32_t>(best)]
                                && rank < static_cast<uint32_t>(best))) {
                            best = static_cast<int32_t>(rank);
                        }
                    }
                    if (best < 0) {
                        break;
                    }
                    extra[static_cast<uint32_t>(best)] = 1U;
                    --remainder;
                }
                for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                    assignmentLoads[rank] += static_cast<float>(rounded[rank] + extra[rank]);
                }
            }
        }
    }

    __aicore__ inline void ResetActionScratch(uint32_t recordCount)
    {
        activeHashCapacity = 2U;
        const uint32_t desired = recordCount > 0U ? 2U * recordCount : 2U;
        while (activeHashCapacity < desired) {
            activeHashCapacity <<= 1U;
        }
        for (uint32_t index = 0; index < activeHashCapacity; ++index) {
            intWorkspaceGm.SetValue(tableKeyOffset + index, -1);
        }
        const uint32_t actionDistribution = 2U * tiling.epSize * tiling.epSize;
        for (uint32_t index = 0; index < actionDistribution; ++index) {
            intWorkspaceGm.SetValue(distributionOffset + index, 0);
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            floatWorkspaceGm.SetValue(rankLoadOffset + rank, floatWorkspaceGm.GetValue(rank));
        }
    }

    __aicore__ inline void RemoveBaselineLogicalLoad(uint32_t logical, uint32_t recordCount)
    {
        if (OneCopyMode()) {
            const uint32_t owner = OwnerRank(logical);
            floatWorkspaceGm.SetValue(
                rankLoadOffset + owner,
                floatWorkspaceGm.GetValue(rankLoadOffset + owner)
                    - floatWorkspaceGm.GetValue(BaselineExpertLoadOffset(logical)));
            return;
        }
        for (uint32_t index = 0U; index < recordCount; ++index) {
            const uint32_t baselineRecord = SharedLogicalRecord(logical, index);
            const uint64_t position = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(BaselineRecordPositionOffset() + baselineRecord));
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const float units = static_cast<float>(sampleMultiplicityGm.GetValue(position))
                * sampleWeightsGm.GetValue(token);
            const uint32_t rank = static_cast<uint32_t>(
                intWorkspaceGm.GetValue(BaselineRecordChosenOffset() + baselineRecord));
            floatWorkspaceGm.SetValue(
                rankLoadOffset + rank,
                floatWorkspaceGm.GetValue(rankLoadOffset + rank) - units);
        }
    }

    __aicore__ inline uint32_t BuildActionRecords(
        uint32_t logical,
        uint32_t baselineRecordCount,
        uint64_t copies,
        uint32_t &bucketCount)
    {
        uint32_t recordCount = 0U;
        bucketCount = 0U;
        const uint32_t logicalOwner = OwnerRank(logical);
        for (uint32_t index = 0U; index < baselineRecordCount; ++index) {
            const uint64_t position = BaselineRouteIndex(logical, index);
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const uint64_t mask = EligibleMask(token, logicalOwner, copies);
            intWorkspaceGm.SetValue(recordPositionOffset + recordCount, position);
            intWorkspaceGm.SetValue(recordMaskOffset + recordCount, static_cast<int64_t>(mask));
            intWorkspaceGm.SetValue(
                recordHashOffset + recordCount,
                BaselineRouteHash(logical, index));
            intWorkspaceGm.SetValue(recordBucketOffset + recordCount, -1);
            const float units = static_cast<float>(sampleMultiplicityGm.GetValue(position))
                * sampleWeightsGm.GetValue(token);
            if (SingleRank(mask)) {
                const uint32_t rank = FirstRank(mask);
                intWorkspaceGm.SetValue(recordChosenOffset + recordCount, rank);
                floatWorkspaceGm.SetValue(
                    rankLoadOffset + rank,
                    floatWorkspaceGm.GetValue(rankLoadOffset + rank) + units);
            } else {
                intWorkspaceGm.SetValue(recordChosenOffset + recordCount, -1);
                const int64_t source = sampleSourcesGm.GetValue(token);
                const int64_t key = source * static_cast<int64_t>(tiling.numExperts) + logical;
                const uint32_t bucket = FindOrCreateBucket(key, mask, bucketCount);
                intWorkspaceGm.SetValue(recordBucketOffset + recordCount, bucket);
                floatWorkspaceGm.SetValue(
                    bucketTotalOffset + bucket,
                    floatWorkspaceGm.GetValue(bucketTotalOffset + bucket) + units);
            }
            ++recordCount;
        }
        return recordCount;
    }

    __aicore__ inline void ProjectRoundedDistribution(
        uint64_t offset,
        int64_t exact,
        int64_t (&rounded)[kMaxRanks]) const
    {
        int64_t total = 0LL;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            rounded[rank] = 0LL;
            total += intWorkspaceGm.GetValue(offset + rank);
        }
        if (total <= 0LL || exact <= 0LL) {
            return;
        }
        float fractions[kMaxRanks];
        uint32_t extra[kMaxRanks];
        int64_t roundedTotal = 0LL;
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            const int64_t count = intWorkspaceGm.GetValue(offset + rank);
            const float raw = static_cast<float>(count) * static_cast<float>(exact)
                / static_cast<float>(total);
            rounded[rank] = static_cast<int64_t>(raw);
            fractions[rank] = raw - static_cast<float>(rounded[rank]);
            extra[rank] = 0U;
            roundedTotal += rounded[rank];
        }
        int64_t remainder = exact - roundedTotal;
        while (remainder > 0LL) {
            int32_t best = -1;
            for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                if (extra[rank] != 0U) {
                    continue;
                }
                if (best < 0 || fractions[rank] > fractions[static_cast<uint32_t>(best)]
                    || (fractions[rank] == fractions[static_cast<uint32_t>(best)]
                        && rank < static_cast<uint32_t>(best))) {
                    best = static_cast<int32_t>(rank);
                }
            }
            if (best < 0) {
                break;
            }
            extra[static_cast<uint32_t>(best)] = 1U;
            ++rounded[static_cast<uint32_t>(best)];
            --remainder;
        }
    }

    __aicore__ inline void ComputeActionDeltas(
        uint32_t logical,
        uint32_t recordCount,
        float (&groupDeltas)[kMaxGroups],
        int64_t (&assignmentDeltas)[kMaxRanks])
    {
        for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
            groupDeltas[group] = 0.0F;
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            assignmentDeltas[rank] = 0LL;
        }

        for (uint32_t record = 0; record < recordCount; ++record) {
            const uint64_t position = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(recordPositionOffset + record));
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const uint32_t oldRank = BaselineChosenRank(logical, record);
            const uint32_t newRank = static_cast<uint32_t>(
                intWorkspaceGm.GetValue(recordChosenOffset + record));
            if (oldRank != newRank) {
                const float weight = sampleWeightsGm.GetValue(token);
                for (uint32_t level = 0; level < tiling.numLevels; ++level) {
                    const uint32_t size = LevelSize(level);
                    const uint32_t oldGroup = oldRank / size;
                    const uint32_t newGroup = newRank / size;
                    if (oldGroup == newGroup) {
                        continue;
                    }
                    bool oldShared = false;
                    bool newShared = false;
                    if (OneCopyMode()) {
                        const uint64_t tokenRoute = static_cast<uint64_t>(token) * tiling.topK;
                        for (uint32_t other = 0U; other < tiling.topK; ++other) {
                            const uint64_t otherRoute = tokenRoute + other;
                            if (otherRoute == position
                                || sampleMultiplicityGm.GetValue(otherRoute) <= 0) {
                                continue;
                            }
                            const uint32_t otherLogical = static_cast<uint32_t>(
                                sampleRoutesGm.GetValue(otherRoute));
                            const uint32_t otherRank = OwnerRank(otherLogical);
                            oldShared = oldShared || otherRank / size == oldGroup;
                            newShared = newShared || otherRank / size == newGroup;
                        }
                    } else {
                        const uint32_t baselineRecord = SharedLogicalRecord(logical, record);
                        const uint32_t tokenBegin = static_cast<uint32_t>(
                            intWorkspaceGm.GetValue(BaselineTokenStartOffset() + token));
                        const uint32_t tokenEnd = static_cast<uint32_t>(
                            intWorkspaceGm.GetValue(BaselineTokenStartOffset() + token + 1U));
                        for (uint32_t other = tokenBegin; other < tokenEnd; ++other) {
                            if (other == baselineRecord) {
                                continue;
                            }
                            const uint32_t otherRank = static_cast<uint32_t>(
                                intWorkspaceGm.GetValue(BaselineRecordChosenOffset() + other));
                            oldShared = oldShared || otherRank / size == oldGroup;
                            newShared = newShared || otherRank / size == newGroup;
                        }
                    }
                    if (!oldShared) {
                        groupDeltas[LevelOffset(level) + oldGroup] -= weight;
                    }
                    if (!newShared) {
                        groupDeltas[LevelOffset(level) + newGroup] += weight;
                    }
                }
            }

            const uint32_t source = static_cast<uint32_t>(sampleSourcesGm.GetValue(token));
            const int64_t multiplicity = sampleMultiplicityGm.GetValue(position);
            const uint64_t sourceOffset = static_cast<uint64_t>(source) * tiling.epSize;
            intWorkspaceGm.SetValue(
                distributionOffset + sourceOffset + oldRank,
                intWorkspaceGm.GetValue(distributionOffset + sourceOffset + oldRank) + multiplicity);
            const uint64_t candidateOffset = static_cast<uint64_t>(tiling.epSize) * tiling.epSize;
            intWorkspaceGm.SetValue(
                distributionOffset + candidateOffset + sourceOffset + newRank,
                intWorkspaceGm.GetValue(distributionOffset + candidateOffset + sourceOffset + newRank)
                    + multiplicity);
        }

        const uint64_t candidateOffset = static_cast<uint64_t>(tiling.epSize) * tiling.epSize;
        for (uint32_t source = 0; source < tiling.epSize; ++source) {
            const int64_t exact = assignmentCountsGm.GetValue(source * tiling.numExperts + logical);
            int64_t baselineRounded[kMaxRanks];
            int64_t candidateRounded[kMaxRanks];
            const uint64_t sourceOffset = static_cast<uint64_t>(source) * tiling.epSize;
            ProjectRoundedDistribution(
                distributionOffset + sourceOffset,
                exact,
                baselineRounded);
            ProjectRoundedDistribution(
                distributionOffset + candidateOffset + sourceOffset,
                exact,
                candidateRounded);
            for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                assignmentDeltas[rank] += candidateRounded[rank] - baselineRounded[rank];
            }
        }
    }

    __aicore__ inline void ResetStrictOneCopyAddScratch(uint32_t logical)
    {
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            floatWorkspaceGm.SetValue(rankLoadOffset + rank, floatWorkspaceGm.GetValue(rank));
        }
        const uint32_t owner = OwnerRank(logical);
        floatWorkspaceGm.SetValue(
            rankLoadOffset + owner,
            floatWorkspaceGm.GetValue(rankLoadOffset + owner)
                - floatWorkspaceGm.GetValue(BaselineExpertLoadOffset(logical)));
    }

    __aicore__ inline void BuildStrictOneCopyAddBuckets(
        uint32_t logical,
        uint32_t destination,
        uint32_t recordCount,
        int32_t (&heads)[kMaxRanks],
        int32_t (&tails)[kMaxRanks],
        uint32_t (&sourceOrder)[kMaxRanks],
        uint32_t &bucketCount,
        float (&bucketTotals)[kMaxRanks],
        uint32_t (&bucketRecords)[kMaxRanks],
        uint32_t (&bucketFlags)[kMaxRanks],
        float (&bucketUnits)[kMaxRanks],
        int64_t (&sourceMultiplicities)[kMaxRanks],
        int64_t (&movedMultiplicities)[kMaxRanks])
    {
        for (uint32_t source = 0U; source < tiling.epSize; ++source) {
            heads[source] = -1;
            tails[source] = -1;
            bucketTotals[source] = 0.0F;
            bucketRecords[source] = 0U;
            bucketFlags[source] = 7U;
            bucketUnits[source] = 0.0F;
        }
        bucketCount = 0U;
        const uint32_t owner = OwnerRank(logical);
        const uint32_t sharedStart = SharedLogicalRecordStart(logical);
        const uint64_t destinationBit = 1ULL << destination;
        for (uint32_t record = 0U; record < recordCount; ++record) {
            const uint64_t position = BaselineRouteIndex(logical, record);
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const uint32_t source = static_cast<uint32_t>(sampleSourcesGm.GetValue(token));
            const int64_t multiplicity = sampleMultiplicityGm.GetValue(position);
            sourceMultiplicities[source] += multiplicity;
            const uint32_t sharedRecord = sharedStart + record;
            const uint64_t move = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(AllDestinationMoveMasksOffset() + sharedRecord));
            const bool deterministicDestination = (move & destinationBit) != 0ULL;
            bool pairTie = false;
            if (!deterministicDestination) {
                const uint64_t tie = static_cast<uint64_t>(
                    intWorkspaceGm.GetValue(AllDestinationTieMasksOffset() + sharedRecord));
                pairTie = (tie & destinationBit) != 0ULL;
            }
            const float units = static_cast<float>(multiplicity) * sampleWeightsGm.GetValue(token);
            if (!pairTie) {
                const uint32_t rank = deterministicDestination ? destination : owner;
                if (rank == destination) {
                    movedMultiplicities[source] += multiplicity;
                }
                floatWorkspaceGm.SetValue(
                    rankLoadOffset + rank,
                    floatWorkspaceGm.GetValue(rankLoadOffset + rank) + units);
                continue;
            }

            if (heads[source] < 0) {
                heads[source] = static_cast<int32_t>(record);
                sourceOrder[bucketCount++] = source;
                bucketUnits[source] = units;
            } else {
                intWorkspaceGm.SetValue(
                    recordMaskOffset + static_cast<uint32_t>(tails[source]),
                    static_cast<int64_t>(record));
                if (units != bucketUnits[source]) {
                    bucketFlags[source] &= ~2U;
                }
            }
            tails[source] = static_cast<int32_t>(record);
            bucketTotals[source] += units;
            ++bucketRecords[source];
            if (!(units > 0.999999F && units < 1.000001F)) {
                bucketFlags[source] &= ~1U;
            }
            if (units == 0.0F) {
                bucketFlags[source] &= ~2U;
            }
            if (multiplicity != 1LL) {
                bucketFlags[source] &= ~4U;
            }
        }
        for (uint32_t index = 0U; index < bucketCount; ++index) {
            const uint32_t source = sourceOrder[index];
            intWorkspaceGm.SetValue(
                recordMaskOffset + static_cast<uint32_t>(tails[source]), -1LL);
        }
    }

    __aicore__ inline void BuildStrictPairQuotas(
        uint32_t owner,
        uint32_t destination,
        int64_t total,
        int64_t &ownerQuota,
        int64_t &destinationQuota) const
    {
        ownerQuota = 0LL;
        destinationQuota = 0LL;
        const uint32_t rankFirst = owner < destination ? owner : destination;
        const uint32_t rankSecond = owner < destination ? destination : owner;
        const float firstRankLoad = floatWorkspaceGm.GetValue(rankLoadOffset + rankFirst);
        const float secondRankLoad = floatWorkspaceGm.GetValue(rankLoadOffset + rankSecond);
        const bool keepRankOrder = firstRankLoad < secondRankLoad
            || (firstRankLoad == secondRankLoad && rankFirst < rankSecond);
        const uint32_t orderedFirst = keepRankOrder ? rankFirst : rankSecond;
        const uint32_t orderedSecond = keepRankOrder ? rankSecond : rankFirst;
        const float firstLoad = keepRankOrder ? firstRankLoad : secondRankLoad;
        const float secondLoad = keepRankOrder ? secondRankLoad : firstRankLoad;

        int64_t remaining = total > 0LL ? total : 0LL;
        int64_t firstQuota = 0LL;
        int64_t secondQuota = 0LL;
        const int64_t required = CeilNonnegative(secondLoad - firstLoad);
        if (required <= remaining) {
            if (required > 0LL) {
                firstQuota += required;
                remaining -= required;
            }
            const int64_t increment = remaining / 2LL;
            const int64_t extra = remaining % 2LL;
            firstQuota += increment + static_cast<int64_t>(extra > 0LL);
            secondQuota += increment;
        } else {
            firstQuota += remaining;
        }

        if (orderedFirst == owner) {
            ownerQuota = firstQuota;
            destinationQuota = secondQuota;
        } else {
            ownerQuota = secondQuota;
            destinationQuota = firstQuota;
        }
    }

    __aicore__ inline bool StrictSourceBucketBetter(
        uint32_t candidate,
        int32_t best,
        const float (&bucketTotals)[kMaxRanks]) const
    {
        if (best < 0) {
            return true;
        }
        const float candidateTotal = bucketTotals[candidate];
        const float bestTotal = bucketTotals[static_cast<uint32_t>(best)];
        if (candidateTotal != bestTotal) {
            return candidateTotal > bestTotal;
        }
        return candidate < static_cast<uint32_t>(best);
    }

    __aicore__ inline void AssignStrictSourceBucket(
        uint32_t logical,
        uint32_t destination,
        uint32_t source,
        int32_t firstRecord,
        uint32_t recordCount,
        float bucketTotal,
        bool allUnit,
        bool uniformUnits,
        bool allMultiplicityOne,
        float uniformUnit,
        int64_t (&movedMultiplicities)[kMaxRanks])
    {
        const uint32_t owner = OwnerRank(logical);
        int64_t ownerQuota = 0LL;
        int64_t destinationQuota = 0LL;
        BuildStrictPairQuotas(
            owner,
            destination,
            RoundEven(bucketTotal),
            ownerQuota,
            destinationQuota);
        const uint32_t rankFirst = owner < destination ? owner : destination;
        const uint32_t rankSecond = owner < destination ? destination : owner;
        const int64_t firstQuota = rankFirst == owner ? ownerQuota : destinationQuota;
        const int64_t secondQuota = rankSecond == owner ? ownerQuota : destinationQuota;
        float firstAssigned = 0.0F;
        float secondAssigned = 0.0F;

        const int64_t quotaTotal = (firstQuota > 0LL ? firstQuota : 0LL)
            + (secondQuota > 0LL ? secondQuota : 0LL);
        bool useSecond = false;
        int64_t consumed = 0LL;
        int64_t cursor = firstRecord;
        for (uint32_t order = 0U; order < recordCount && cursor >= 0LL; ++order) {
            const uint32_t record = static_cast<uint32_t>(cursor);
            cursor = intWorkspaceGm.GetValue(recordMaskOffset + record);
            const uint64_t position = BaselineRouteIndex(logical, record);
            float units = uniformUnit;
            if (!uniformUnits) {
                const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
                units = static_cast<float>(sampleMultiplicityGm.GetValue(position))
                    * sampleWeightsGm.GetValue(token);
            }
            uint32_t chosen = rankFirst;
            if (allUnit) {
                if (!useSecond && static_cast<int64_t>(order) >= consumed + firstQuota) {
                    consumed += firstQuota;
                    useSecond = true;
                }
                chosen = quotaTotal > 0LL
                    ? (useSecond ? rankSecond : rankFirst)
                    : ((order & 1U) != 0U ? rankSecond : rankFirst);
            } else {
                const float firstDeficit = firstAssigned + units - static_cast<float>(firstQuota);
                const float firstProjected = floatWorkspaceGm.GetValue(rankLoadOffset + rankFirst)
                    + firstAssigned + units;
                const float secondDeficit = secondAssigned + units - static_cast<float>(secondQuota);
                const float secondProjected = floatWorkspaceGm.GetValue(rankLoadOffset + rankSecond)
                    + secondAssigned + units;
                if (secondDeficit < firstDeficit
                    || (secondDeficit == firstDeficit
                        && (secondProjected < firstProjected
                            || (secondProjected == firstProjected && rankSecond < rankFirst)))) {
                    chosen = rankSecond;
                }
            }
            intWorkspaceGm.SetValue(recordChosenOffset + record, chosen);
            if (chosen == destination) {
                movedMultiplicities[source] += allMultiplicityOne
                    ? 1LL
                    : sampleMultiplicityGm.GetValue(position);
            }
            if (chosen == rankFirst) {
                firstAssigned += units;
            } else {
                secondAssigned += units;
            }
        }
        floatWorkspaceGm.SetValue(
            rankLoadOffset + rankFirst,
            floatWorkspaceGm.GetValue(rankLoadOffset + rankFirst) + firstAssigned);
        floatWorkspaceGm.SetValue(
            rankLoadOffset + rankSecond,
            floatWorkspaceGm.GetValue(rankLoadOffset + rankSecond) + secondAssigned);
    }

    __aicore__ inline void ProcessStrictOneCopyAddBuckets(
        uint32_t logical,
        uint32_t destination,
        const int32_t (&heads)[kMaxRanks],
        const uint32_t (&sourceOrder)[kMaxRanks],
        uint32_t bucketCount,
        const float (&bucketTotals)[kMaxRanks],
        const uint32_t (&bucketRecords)[kMaxRanks],
        const uint32_t (&bucketFlags)[kMaxRanks],
        const float (&bucketUnits)[kMaxRanks],
        int64_t (&movedMultiplicities)[kMaxRanks])
    {
        uint32_t processed[kMaxRanks];
        for (uint32_t source = 0U; source < tiling.epSize; ++source) {
            processed[source] = 0U;
        }
        for (uint32_t completed = 0U; completed < bucketCount; ++completed) {
            int32_t best = -1;
            for (uint32_t index = 0U; index < bucketCount; ++index) {
                const uint32_t source = sourceOrder[index];
                if (processed[source] == 0U && StrictSourceBucketBetter(source, best, bucketTotals)) {
                    best = static_cast<int32_t>(source);
                }
            }
            if (best < 0) {
                break;
            }
            const uint32_t source = static_cast<uint32_t>(best);
            processed[source] = 1U;
            AssignStrictSourceBucket(
                logical,
                destination,
                source,
                heads[source],
                bucketRecords[source],
                bucketTotals[source],
                (bucketFlags[source] & 1U) != 0U,
                (bucketFlags[source] & 2U) != 0U,
                (bucketFlags[source] & 4U) != 0U,
                bucketUnits[source],
                movedMultiplicities);
        }
    }

    __aicore__ inline void ProjectStrictPairDistribution(
        uint32_t owner,
        uint32_t destination,
        int64_t total,
        int64_t moved,
        int64_t exact,
        int64_t (&rounded)[kMaxRanks]) const
    {
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            rounded[rank] = 0LL;
        }
        if (total <= 0LL || exact <= 0LL) {
            return;
        }
        float fractions[kMaxRanks];
        uint32_t extra[kMaxRanks];
        int64_t roundedTotal = 0LL;
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            int64_t count = 0LL;
            if (rank == owner) {
                count = total - moved;
            } else if (rank == destination) {
                count = moved;
            }
            const float raw = static_cast<float>(count) * static_cast<float>(exact)
                / static_cast<float>(total);
            rounded[rank] = static_cast<int64_t>(raw);
            fractions[rank] = raw - static_cast<float>(rounded[rank]);
            extra[rank] = 0U;
            roundedTotal += rounded[rank];
        }
        int64_t remainder = exact - roundedTotal;
        while (remainder > 0LL) {
            int32_t best = -1;
            for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
                if (extra[rank] != 0U) {
                    continue;
                }
                if (best < 0 || fractions[rank] > fractions[static_cast<uint32_t>(best)]
                    || (fractions[rank] == fractions[static_cast<uint32_t>(best)]
                        && rank < static_cast<uint32_t>(best))) {
                    best = static_cast<int32_t>(rank);
                }
            }
            if (best < 0) {
                break;
            }
            extra[static_cast<uint32_t>(best)] = 1U;
            ++rounded[static_cast<uint32_t>(best)];
            --remainder;
        }
    }

    __aicore__ inline bool TryStrictPairMovedExact(
        uint32_t owner,
        uint32_t destination,
        int64_t total,
        int64_t moved,
        int64_t exact,
        int64_t &movedExact) const
    {
        movedExact = 0LL;
        if (total <= 0LL || exact <= 0LL) {
            return true;
        }
        const float totalFloat = static_cast<float>(total);
        const float exactFloat = static_cast<float>(exact);
        const float baselineRaw = totalFloat * exactFloat / totalFloat;
        const int64_t baselineFloor = static_cast<int64_t>(baselineRaw);
        const float baselineFraction = baselineRaw - static_cast<float>(baselineFloor);
        const int64_t baselineRemainder = exact - baselineFloor;
        bool baselineSafe = baselineRemainder == 0LL && baselineFloor == exact;
        if (baselineRemainder == 1LL && baselineFloor + 1LL == exact) {
            baselineSafe = baselineFraction > 0.0F || owner == 0U;
        }
        if (!baselineSafe) {
            return false;
        }

        const float ownerRaw = static_cast<float>(total - moved) * exactFloat / totalFloat;
        const float destinationRaw = static_cast<float>(moved) * exactFloat / totalFloat;
        int64_t ownerRounded = static_cast<int64_t>(ownerRaw);
        int64_t destinationRounded = static_cast<int64_t>(destinationRaw);
        const float ownerFraction = ownerRaw - static_cast<float>(ownerRounded);
        const float destinationFraction = destinationRaw - static_cast<float>(destinationRounded);
        const int64_t remainder = exact - ownerRounded - destinationRounded;
        if (remainder < 0LL || remainder > 2LL) {
            return false;
        }

        uint32_t lowestZero = 0U;
        while (lowestZero < tiling.epSize
               && (lowestZero == owner || lowestZero == destination)) {
            ++lowestZero;
        }
        bool usedOwner = false;
        bool usedDestination = false;
        for (int64_t step = 0LL; step < remainder; ++step) {
            bool initialized = false;
            uint32_t bestKind = 2U;
            uint32_t bestRank = lowestZero;
            float bestFraction = 0.0F;
            if (!usedOwner) {
                initialized = true;
                bestKind = 0U;
                bestRank = owner;
                bestFraction = ownerFraction;
            }
            if (!usedDestination
                && (!initialized || destinationFraction > bestFraction
                    || (destinationFraction == bestFraction && destination < bestRank))) {
                initialized = true;
                bestKind = 1U;
                bestRank = destination;
                bestFraction = destinationFraction;
            }
            if (lowestZero < tiling.epSize
                && (!initialized || 0.0F > bestFraction
                    || (0.0F == bestFraction && lowestZero < bestRank))) {
                bestKind = 2U;
            }
            if (bestKind == 2U) {
                return false;
            }
            if (bestKind == 0U) {
                usedOwner = true;
                ++ownerRounded;
            } else {
                usedDestination = true;
                ++destinationRounded;
            }
        }
        if (ownerRounded + destinationRounded != exact) {
            return false;
        }
        movedExact = destinationRounded;
        return true;
    }

    __aicore__ inline void AccumulateStrictPairAssignmentDelta(
        uint32_t owner,
        uint32_t destination,
        int64_t total,
        int64_t moved,
        int64_t exact,
        int64_t (&assignmentDeltas)[kMaxRanks]) const
    {
        int64_t movedExact = 0LL;
        if (TryStrictPairMovedExact(owner, destination, total, moved, exact, movedExact)) {
            assignmentDeltas[owner] -= movedExact;
            assignmentDeltas[destination] += movedExact;
            return;
        }
        int64_t baselineRounded[kMaxRanks];
        int64_t candidateRounded[kMaxRanks];
        ProjectStrictPairDistribution(
            owner, destination, total, 0LL, exact, baselineRounded);
        ProjectStrictPairDistribution(
            owner, destination, total, moved, exact, candidateRounded);
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            assignmentDeltas[rank] += candidateRounded[rank] - baselineRounded[rank];
        }
    }

    __aicore__ inline void ComputeStrictOneCopyAddDeltas(
        uint32_t logical,
        uint32_t destination,
        uint32_t recordCount,
        const int64_t (&sourceMultiplicities)[kMaxRanks],
        const int64_t (&movedMultiplicities)[kMaxRanks],
        float (&groupDeltas)[kMaxGroups],
        int64_t (&assignmentDeltas)[kMaxRanks])
    {
        for (uint32_t group = 0U; group < tiling.totalGroups; ++group) {
            groupDeltas[group] = 0.0F;
        }
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            assignmentDeltas[rank] = 0LL;
        }

        const uint32_t oldRank = OwnerRank(logical);
        const uint32_t sharedStart = SharedLogicalRecordStart(logical);
        const uint64_t destinationBit = 1ULL << destination;
        for (uint32_t record = 0U; record < recordCount; ++record) {
            const uint64_t position = BaselineRouteIndex(logical, record);
            const uint32_t token = static_cast<uint32_t>(position / tiling.topK);
            const uint32_t sharedRecord = sharedStart + record;
            const uint64_t move = static_cast<uint64_t>(
                intWorkspaceGm.GetValue(AllDestinationMoveMasksOffset() + sharedRecord));
            uint32_t newRank = oldRank;
            if ((move & destinationBit) != 0ULL) {
                newRank = destination;
            } else {
                const uint64_t tie = static_cast<uint64_t>(
                    intWorkspaceGm.GetValue(AllDestinationTieMasksOffset() + sharedRecord));
                if ((tie & destinationBit) != 0ULL) {
                    newRank = static_cast<uint32_t>(
                        intWorkspaceGm.GetValue(recordChosenOffset + record));
                }
            }
            if (oldRank != newRank) {
                const float weight = sampleWeightsGm.GetValue(token);
                for (uint32_t level = 0U; level < tiling.numLevels; ++level) {
                    const uint32_t size = LevelSize(level);
                    const uint32_t oldGroup = oldRank / size;
                    const uint32_t newGroup = newRank / size;
                    if (oldGroup == newGroup) {
                        continue;
                    }
                    const uint64_t coverage = static_cast<uint64_t>(
                        intWorkspaceGm.GetValue(ActualTokenCoverageIndex(token, level, false)));
                    const uint64_t duplicate = static_cast<uint64_t>(
                        intWorkspaceGm.GetValue(ActualTokenCoverageIndex(token, level, true)));
                    const bool oldShared = (duplicate & (1ULL << oldGroup)) != 0ULL;
                    const bool newShared = (coverage & (1ULL << newGroup)) != 0ULL;
                    if (!oldShared) {
                        groupDeltas[LevelOffset(level) + oldGroup] -= weight;
                    }
                    if (!newShared) {
                        groupDeltas[LevelOffset(level) + newGroup] += weight;
                    }
                }
            }

        }

        for (uint32_t source = 0U; source < tiling.epSize; ++source) {
            const int64_t exact = assignmentCountsGm.GetValue(source * tiling.numExperts + logical);
            AccumulateStrictPairAssignmentDelta(
                oldRank,
                destination,
                sourceMultiplicities[source],
                movedMultiplicities[source],
                exact,
                assignmentDeltas);
        }
    }

    __aicore__ inline void ProjectStrictOneCopyAdd(
        uint32_t logical,
        uint32_t actionRank,
        uint32_t actionIndex,
        uint32_t logicalRecords)
    {
        if (logicalRecords == 0U) {
            ZeroAdd(actionIndex);
            return;
        }
        ResetStrictOneCopyAddScratch(logical);

        int32_t heads[kMaxRanks];
        int32_t tails[kMaxRanks];
        uint32_t sourceOrder[kMaxRanks];
        uint32_t bucketCount = 0U;
        float bucketTotals[kMaxRanks];
        uint32_t bucketRecords[kMaxRanks];
        uint32_t bucketFlags[kMaxRanks];
        float bucketUnits[kMaxRanks];
        int64_t sourceMultiplicities[kMaxRanks];
        int64_t movedMultiplicities[kMaxRanks];
        for (uint32_t source = 0U; source < tiling.epSize; ++source) {
            sourceMultiplicities[source] = 0LL;
            movedMultiplicities[source] = 0LL;
        }
        BuildStrictOneCopyAddBuckets(
            logical,
            actionRank,
            logicalRecords,
            heads,
            tails,
            sourceOrder,
            bucketCount,
            bucketTotals,
            bucketRecords,
            bucketFlags,
            bucketUnits,
            sourceMultiplicities,
            movedMultiplicities);
        ProcessStrictOneCopyAddBuckets(
            logical,
            actionRank,
            heads,
            sourceOrder,
            bucketCount,
            bucketTotals,
            bucketRecords,
            bucketFlags,
            bucketUnits,
            movedMultiplicities);

        float groupDeltas[kMaxGroups];
        int64_t assignmentDeltas[kMaxRanks];
        ComputeStrictOneCopyAddDeltas(
            logical,
            actionRank,
            logicalRecords,
            sourceMultiplicities,
            movedMultiplicities,
            groupDeltas,
            assignmentDeltas);
        const uint64_t groupOffset = static_cast<uint64_t>(actionIndex) * tiling.groupOutputStride;
        const uint64_t assignmentOffset = static_cast<uint64_t>(actionIndex) * tiling.assignmentOutputStride;
        for (uint32_t group = 0U; group < tiling.totalGroups; ++group) {
            addGroupDeltasGm.SetValue(groupOffset + group, groupDeltas[group]);
        }
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            addAssignmentDeltasGm.SetValue(
                assignmentOffset + rank,
                static_cast<float>(assignmentDeltas[rank]));
        }
    }

    __aicore__ inline void ProjectAction(
        uint32_t kind,
        uint32_t logical,
        uint32_t actionRank,
        int64_t removeSlot,
        uint32_t actionIndex,
        uint32_t logicalRecords)
    {
        if (kind == kAdd && OneCopyMode()) {
            ProjectStrictOneCopyAdd(logical, actionRank, actionIndex, logicalRecords);
            return;
        }
        const uint64_t copies = CopyMask(logical, kind, logical, actionRank, removeSlot);
        ResetActionScratch(logicalRecords);
        RemoveBaselineLogicalLoad(logical, logicalRecords);
        uint32_t bucketCount = 0U;
        const uint32_t recordCount = BuildActionRecords(
            logical, logicalRecords, copies, bucketCount);
        BuildBucketRecordLists(bucketCount, recordCount);
        ProcessBuckets(bucketCount, true);
        float groupDeltas[kMaxGroups];
        int64_t assignmentDeltas[kMaxRanks];
        ComputeActionDeltas(logical, recordCount, groupDeltas, assignmentDeltas);

        const uint64_t groupOffset = static_cast<uint64_t>(actionIndex) * tiling.groupOutputStride;
        const uint64_t assignmentOffset = static_cast<uint64_t>(actionIndex) * tiling.assignmentOutputStride;
        if (kind == kAdd) {
            for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
                addGroupDeltasGm.SetValue(groupOffset + group, groupDeltas[group]);
            }
            for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                addAssignmentDeltasGm.SetValue(
                    assignmentOffset + rank,
                    static_cast<float>(assignmentDeltas[rank]));
            }
            return;
        }
        for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
            removeGroupDeltasGm.SetValue(groupOffset + group, groupDeltas[group]);
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            removeAssignmentDeltasGm.SetValue(
                assignmentOffset + rank,
                static_cast<float>(assignmentDeltas[rank]));
        }
    }

    __aicore__ inline void Project(
        uint32_t kind,
        uint32_t actionLogical,
        uint32_t actionRank,
        int64_t removeSlot,
        uint32_t actionIndex)
    {
        ResetProjection();
        uint32_t bucketCount = 0U;
        const uint32_t recordCount = BuildRecords(bucketCount);
        BuildBucketRecordLists(bucketCount, recordCount);
        ProcessBuckets(bucketCount, false);
        float groupCounts[kMaxGroups];
        float loads[kMaxRanks];
        ComputeStats(recordCount, groupCounts, loads);
        if (kind == kBaseline) {
            for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
                baseCountsGm.SetValue(group, groupCounts[group]);
            }
            for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                assignmentLoadsGm.SetValue(rank, loads[rank]);
            }
            return;
        }
        if (kind == kAdd) {
            for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
                addGroupDeltasGm.SetValue(
                    static_cast<uint64_t>(actionIndex) * tiling.groupOutputStride + group,
                    groupCounts[group] - baseCountsGm.GetValue(group));
            }
            for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
                addAssignmentDeltasGm.SetValue(
                    static_cast<uint64_t>(actionIndex) * tiling.assignmentOutputStride + rank,
                    loads[rank] - assignmentLoadsGm.GetValue(rank));
            }
            return;
        }
        for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
            removeGroupDeltasGm.SetValue(
                static_cast<uint64_t>(actionIndex) * tiling.groupOutputStride + group,
                groupCounts[group] - baseCountsGm.GetValue(group));
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            removeAssignmentDeltasGm.SetValue(
                static_cast<uint64_t>(actionIndex) * tiling.assignmentOutputStride + rank,
                loads[rank] - assignmentLoadsGm.GetValue(rank));
        }
    }

    __aicore__ inline void ZeroAdd(uint32_t action)
    {
        for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
            addGroupDeltasGm.SetValue(static_cast<uint64_t>(action) * tiling.groupOutputStride + group, 0.0F);
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            addAssignmentDeltasGm.SetValue(
                static_cast<uint64_t>(action) * tiling.assignmentOutputStride + rank, 0.0F);
        }
    }

    __aicore__ inline void ZeroRemove(uint32_t action)
    {
        for (uint32_t group = 0; group < tiling.totalGroups; ++group) {
            removeGroupDeltasGm.SetValue(static_cast<uint64_t>(action) * tiling.groupOutputStride + group, 0.0F);
        }
        for (uint32_t rank = 0; rank < tiling.epSize; ++rank) {
            removeAssignmentDeltasGm.SetValue(
                static_cast<uint64_t>(action) * tiling.assignmentOutputStride + rank, 0.0F);
        }
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hashSortScratchBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hashSortPackedBuf;
    AscendC::GlobalTensor<int64_t> sampleRoutesGm;
    AscendC::GlobalTensor<int64_t> sampleMultiplicityGm;
    AscendC::GlobalTensor<float> sampleWeightsGm;
    AscendC::GlobalTensor<int64_t> sampleSourcesGm;
    AscendC::GlobalTensor<int64_t> sampleOrdinalsGm;
    AscendC::GlobalTensor<int64_t> assignmentCountsGm;
    AscendC::GlobalTensor<float> seedBaseCountsGm;
    AscendC::GlobalTensor<int64_t> slotToLogicalGm;
    AscendC::GlobalTensor<int64_t> ownerSlotsGm;
    AscendC::GlobalTensor<int64_t> redundantSlotsGm;
    AscendC::GlobalTensor<int32_t> candidateExpertsGm;
    AscendC::GlobalTensor<float> baseCountsGm;
    AscendC::GlobalTensor<float> assignmentLoadsGm;
    AscendC::GlobalTensor<float> addGroupDeltasGm;
    AscendC::GlobalTensor<float> addAssignmentDeltasGm;
    AscendC::GlobalTensor<float> removeGroupDeltasGm;
    AscendC::GlobalTensor<float> removeAssignmentDeltasGm;
    AscendC::GlobalTensor<int64_t> intWorkspaceGm;
    AscendC::GlobalTensor<float> floatWorkspaceGm;
    HiermoeReplicaProjectTilingData tiling;
    uint64_t recordPositionOffset = 0;
    uint64_t recordMaskOffset = 0;
    uint64_t recordHashOffset = 0;
    uint64_t recordChosenOffset = 0;
    uint64_t recordBucketOffset = 0;
    uint64_t tableKeyOffset = 0;
    uint64_t tableMaskOffset = 0;
    uint64_t tableBucketOffset = 0;
    uint64_t bucketKeyOffset = 0;
    uint64_t bucketMaskOffset = 0;
    uint64_t bucketProcessedOffset = 0;
    uint64_t distributionOffset = 0;
    uint64_t tokenStartOffset = 0;
    uint64_t rankLoadOffset = 0;
    uint64_t bucketTotalOffset = 0;
    uint32_t workspaceMaxRecords = 0U;
    uint32_t workspaceHashCapacity = 2U;
    uint32_t workspaceDistributionSize = 0U;
    uint32_t activeHashCapacity = 2U;
};

extern "C" __global__ __aicore__ void hiermoe_replica_project(
    GM_ADDR sampleRoutes,
    GM_ADDR sampleMultiplicity,
    GM_ADDR sampleWeights,
    GM_ADDR sampleSources,
    GM_ADDR sampleOrdinals,
    GM_ADDR assignmentCounts,
    GM_ADDR seedBaseCounts,
    GM_ADDR slotToLogical,
    GM_ADDR ownerSlots,
    GM_ADDR redundantSlots,
    GM_ADDR candidateExperts,
    GM_ADDR baseCounts,
    GM_ADDR assignmentLoads,
    GM_ADDR addGroupDeltas,
    GM_ADDR addAssignmentDeltas,
    GM_ADDR removeGroupDeltas,
    GM_ADDR removeAssignmentDeltas,
    GM_ADDR intWorkspace,
    GM_ADDR floatWorkspace,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeReplicaProjectTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeReplicaProject op;
    op.Init(
        sampleRoutes,
        sampleMultiplicity,
        sampleWeights,
        sampleSources,
        sampleOrdinals,
        assignmentCounts,
        seedBaseCounts,
        slotToLogical,
        ownerSlots,
        redundantSlots,
        candidateExperts,
        baseCounts,
        assignmentLoads,
        addGroupDeltas,
        addAssignmentDeltas,
        removeGroupDeltas,
        removeAssignmentDeltas,
        intWorkspace,
        floatWorkspace,
        tilingData);
    op.Process();
}
