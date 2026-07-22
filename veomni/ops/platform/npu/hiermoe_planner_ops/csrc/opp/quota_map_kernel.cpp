#include "kernel_operator.h"
#include "quota_map_tiling.h"

class KernelHiermoeQuotaMap {
public:
    __aicore__ inline void Init(
        GM_ADDR selected,
        GM_ADDR copySlots,
        GM_ADDR copyCounts,
        GM_ADDR ownerRanks,
        GM_ADDR quotaWeights,
        GM_ADDR quotaConfigured,
        GM_ADDR tokenOrdinals,
        GM_ADDR physical,
        GM_ADDR groupCounts,
        GM_ADDR assignmentCounts,
        GM_ADDR intWorkspace,
        const HiermoeQuotaMapTilingData &tiling)
    {
        this->tiling = tiling;
        selectedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(selected));
        copySlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(copySlots));
        copyCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(copyCounts));
        ownerRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ownerRanks));
        quotaWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(quotaWeights));
        quotaConfiguredGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(quotaConfigured));
        tokenOrdinalsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tokenOrdinals));
        physicalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(physical));
        groupCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(groupCounts));
        assignmentCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(assignmentCounts));
        intWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(intWorkspace));
        pipe.InitBuffer(hashSortScratchBuf, 2U * kHardwareSortCapacity * sizeof(float));
        pipe.InitBuffer(hashSortPackedBuf, 2U * kHardwareSortCapacity * sizeof(float));

        recordStride = tiling.sortStride;
        recordCapacity = recordStride * 2ULL;
        logicalOffset = 0U;
        multiplicityOffset = logicalOffset + recordCapacity;
        maskOffset = multiplicityOffset + recordCapacity;
        hashOffset = maskOffset + recordCapacity;
        tupleKeyOffset = hashOffset + recordCapacity;
        orderOffset = tupleKeyOffset + recordCapacity;
        temporaryOffset = orderOffset + 2ULL * tiling.sortStride;
        privateBucketTotalOffset = temporaryOffset + 2ULL * tiling.sortStride;
        privateBucketUnitOffset = privateBucketTotalOffset
            + static_cast<uint64_t>(tiling.blockCount) * 2ULL * tiling.bucketStride;
        bucketTotalOffset = privateBucketUnitOffset
            + static_cast<uint64_t>(tiling.blockCount) * 2ULL * tiling.bucketStride;
        bucketUnitOffset = bucketTotalOffset + 2ULL * tiling.bucketStride;
        privateRankLoadOffset = bucketUnitOffset + 2ULL * tiling.bucketStride;
        rankLoadOffset = privateRankLoadOffset
            + static_cast<uint64_t>(tiling.blockCount) * 2ULL * tiling.rankStride;
        recordCountOffset = rankLoadOffset + 2ULL * tiling.rankStride;
        statsOffset = recordCountOffset + static_cast<uint64_t>(tiling.blockCount) * 16ULL;
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t firstToken = block * tiling.tokensPerBlock;
        const uint32_t tokenLimit = firstToken + tiling.tokensPerBlock;
        const uint32_t lastToken = tokenLimit < tiling.numTokens ? tokenLimit : tiling.numTokens;
        InitializeWorkspace(block, firstToken, lastToken);

        uint64_t recordCounts[2] = {0ULL, 0ULL};
        for (uint32_t layout = 0; layout < 2U; ++layout) {
            for (uint32_t token = firstToken; token < lastToken; ++token) {
                BuildTokenRecords(layout, token, block, recordCounts[layout]);
            }
            intWorkspaceGm.SetValue(
                recordCountOffset + static_cast<uint64_t>(block) * 16ULL + layout,
                static_cast<int64_t>(recordCounts[layout]));
        }
        FlushIntAndPhysical();
        AscendC::SyncAll<true>();

        ReduceBucketsAndLoads(block);
        FlushInt();
        AscendC::SyncAll<true>();

        for (uint32_t layout = 0U; layout < 2U; ++layout) {
            const uint32_t ownerBlock = layout < tiling.blockCount ? layout : 0U;
            if (block == ownerBlock) {
                SetHardwareSortFlag(layout, TryHardwareSortSingleBucket(layout));
            }
        }
        FlushInt();
        AscendC::SyncAll<true>();

        bool finalOrder = true;
        const bool allHardwareSorted = HardwareSortFlag(0U) && HardwareSortFlag(1U);
        if (!allHardwareSorted) {
            for (uint32_t layout = 0; layout < 2U; ++layout) {
                if (!HardwareSortFlag(layout)) {
                    SortLocalRun(layout, block, recordCounts[layout]);
                }
            }
            FlushInt();
            AscendC::SyncAll<true>();

            for (uint32_t span = 1U; span < tiling.blockCount; span *= 2U) {
                const uint32_t groups = (tiling.blockCount + 2U * span - 1U) / (2U * span);
                const uint32_t taskCount = 2U * groups;
                for (uint32_t task = block; task < taskCount; task += tiling.blockCount) {
                    const uint32_t layout = task / groups;
                    const uint32_t group = task % groups;
                    if (!HardwareSortFlag(layout)) {
                        MergeRuns(layout, group * 2U * span, span, finalOrder);
                    }
                }
                FlushInt();
                AscendC::SyncAll<true>();
                finalOrder = !finalOrder;
            }
        }

        if (block == 0U) {
            for (uint32_t layout = 0U; layout < 2U; ++layout) {
                const bool orderBuffer = HardwareSortFlag(layout) ? true : finalOrder;
                AssignBuckets(layout, TotalRecordCount(layout), orderBuffer);
            }
        }
        FlushIntAndPhysical();
        AscendC::SyncAll<true>();

        for (uint32_t layout = 0; layout < 2U; ++layout) {
            AccumulateStats(layout, block, firstToken, lastToken);
        }
        FlushInt();
        AscendC::SyncAll<true>();

        ReduceStats(block);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(groupCountsGm);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(assignmentCountsGm);
    }

private:
    static constexpr uint32_t kHardwareSortCapacity = 4096U;
    static constexpr uint32_t kHardwareSortMaxRecords = 4095U;
    static constexpr uint32_t kRouteHashMax = 1048572U;
    static constexpr uint32_t kFloatIntegerBiasBits = 0x4b000000U;
    static constexpr uint32_t kPaddedScoreBiasBits = 0x4afffffeU;
    static constexpr float kFloatIntegerBias = 8388608.0F;

    __aicore__ inline uint64_t TableBase(uint32_t layout) const
    {
        return static_cast<uint64_t>(layout) * tiling.numExperts;
    }

    __aicore__ inline uint64_t CopyBase(uint32_t layout, int64_t logical) const
    {
        return (TableBase(layout) + static_cast<uint64_t>(logical)) * tiling.maxCopies;
    }

    __aicore__ inline uint64_t BucketIndex(uint32_t layout, int64_t logical, uint32_t mask) const
    {
        return (TableBase(layout) + static_cast<uint64_t>(logical)) * tiling.maskCount + mask;
    }

    __aicore__ inline uint64_t BucketStorageIndex(uint32_t layout, int64_t logical, uint32_t mask) const
    {
        return static_cast<uint64_t>(layout) * tiling.bucketStride
            + static_cast<uint64_t>(logical) * tiling.maskCount + mask;
    }

    __aicore__ inline uint64_t PrivateBucketBase(uint32_t block) const
    {
        return static_cast<uint64_t>(block) * 2ULL * tiling.bucketStride;
    }

    __aicore__ inline uint64_t RankStorageIndex(uint32_t layout, uint32_t rank) const
    {
        return static_cast<uint64_t>(layout) * tiling.rankStride + rank;
    }

    __aicore__ inline uint64_t PrivateRankBase(uint32_t block) const
    {
        return static_cast<uint64_t>(block) * 2ULL * tiling.rankStride;
    }

    __aicore__ inline uint64_t WorkChunk(uint64_t elements) const
    {
        const uint64_t unaligned = (elements + tiling.blockCount - 1ULL) / tiling.blockCount;
        return (unaligned + 7ULL) / 8ULL * 8ULL;
    }

    __aicore__ inline uint64_t OutputBase(uint32_t layout, uint32_t token) const
    {
        return (static_cast<uint64_t>(token) * 2U + layout) * tiling.topK;
    }

    __aicore__ inline void FlushInt()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
    }

    __aicore__ inline void FlushIntAndPhysical()
    {
        FlushInt();
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(physicalGm);
    }

    __aicore__ inline void InitializeWorkspace(uint32_t block, uint32_t firstToken, uint32_t lastToken)
    {
        const uint64_t firstOutput = static_cast<uint64_t>(firstToken) * 2ULL * tiling.topK;
        const uint64_t lastOutput = static_cast<uint64_t>(lastToken) * 2ULL * tiling.topK;
        for (uint64_t index = firstOutput; index < lastOutput; ++index) {
            physicalGm.SetValue(index, -1LL);
        }

        const uint64_t privateBucketBase = PrivateBucketBase(block);
        for (uint32_t layout = 0; layout < 2U; ++layout) {
            for (uint32_t logical = 0; logical < tiling.numExperts; ++logical) {
                const int64_t rawCopies = copyCountsGm.GetValue(TableBase(layout) + logical);
                if (rawCopies <= 0LL || rawCopies > static_cast<int64_t>(tiling.maxCopies)) {
                    continue;
                }
                const uint32_t validMasks = 1U << static_cast<uint32_t>(rawCopies);
                const uint64_t bucketBase = privateBucketBase
                    + static_cast<uint64_t>(layout) * tiling.bucketStride
                    + static_cast<uint64_t>(logical) * tiling.maskCount;
                for (uint32_t mask = 1U; mask < validMasks; ++mask) {
                    intWorkspaceGm.SetValue(privateBucketTotalOffset + bucketBase + mask, 0LL);
                    intWorkspaceGm.SetValue(privateBucketUnitOffset + bucketBase + mask, 1LL);
                }
            }
        }

        const uint64_t privateRankBase = PrivateRankBase(block);
        for (uint64_t rank = 0; rank < 2ULL * tiling.rankStride; ++rank) {
            intWorkspaceGm.SetValue(privateRankLoadOffset + privateRankBase + rank, 0LL);
        }

        const uint64_t countBase = recordCountOffset + static_cast<uint64_t>(block) * 16ULL;
        for (uint32_t index = 0; index < 16U; ++index) {
            intWorkspaceGm.SetValue(countBase + index, 0LL);
        }

        const uint64_t statsBase = statsOffset + static_cast<uint64_t>(block) * tiling.statsStride;
        for (uint32_t index = 0; index < tiling.statsStride; ++index) {
            intWorkspaceGm.SetValue(statsBase + index, 0LL);
        }
    }

    __aicore__ inline void ReduceBucketsAndLoads(uint32_t block)
    {
        const uint32_t logicalAlignment = tiling.maskCount < 8U ? 8U / tiling.maskCount : 1U;
        const uint32_t unalignedLogicalChunk =
            (tiling.numExperts + tiling.blockCount - 1U) / tiling.blockCount;
        const uint32_t logicalChunk =
            (unalignedLogicalChunk + logicalAlignment - 1U) / logicalAlignment * logicalAlignment;
        const uint32_t firstLogical = block * logicalChunk;
        const uint32_t lastLogical = firstLogical + logicalChunk < tiling.numExperts
            ? firstLogical + logicalChunk
            : tiling.numExperts;
        for (uint32_t layout = 0; layout < 2U; ++layout) {
            for (uint32_t logical = firstLogical; logical < lastLogical; ++logical) {
                const int64_t rawCopies = copyCountsGm.GetValue(TableBase(layout) + logical);
                if (rawCopies <= 0LL || rawCopies > static_cast<int64_t>(tiling.maxCopies)) {
                    continue;
                }
                const uint32_t validMasks = 1U << static_cast<uint32_t>(rawCopies);
                const uint64_t bucketBase = static_cast<uint64_t>(layout) * tiling.bucketStride
                    + static_cast<uint64_t>(logical) * tiling.maskCount;
                for (uint32_t mask = 1U; mask < validMasks; ++mask) {
                    int64_t total = 0LL;
                    int64_t unit = 1LL;
                    for (uint32_t sourceBlock = 0; sourceBlock < tiling.blockCount; ++sourceBlock) {
                        const uint64_t source = PrivateBucketBase(sourceBlock) + bucketBase + mask;
                        total += intWorkspaceGm.GetValue(privateBucketTotalOffset + source);
                        unit &= intWorkspaceGm.GetValue(privateBucketUnitOffset + source);
                    }
                    intWorkspaceGm.SetValue(bucketTotalOffset + bucketBase + mask, total);
                    intWorkspaceGm.SetValue(bucketUnitOffset + bucketBase + mask, unit);
                }
            }
        }

        const uint64_t rankElements = 2ULL * tiling.rankStride;
        const uint64_t rankChunk = WorkChunk(rankElements);
        const uint64_t firstRank = static_cast<uint64_t>(block) * rankChunk;
        const uint64_t lastRank = firstRank + rankChunk < rankElements ? firstRank + rankChunk : rankElements;
        for (uint64_t rank = firstRank; rank < lastRank; ++rank) {
            int64_t load = 0LL;
            for (uint32_t sourceBlock = 0; sourceBlock < tiling.blockCount; ++sourceBlock) {
                load += intWorkspaceGm.GetValue(
                    privateRankLoadOffset + PrivateRankBase(sourceBlock) + rank);
            }
            intWorkspaceGm.SetValue(rankLoadOffset + rank, load);
        }
    }

    __aicore__ inline bool TokenDomainValid(uint32_t layout, uint32_t token)
    {
        if (tokenOrdinalsGm.GetValue(token) < 0) {
            return false;
        }
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        const uint64_t ownerOffset = TableBase(layout);
        for (uint32_t position = 0; position < tiling.topK; ++position) {
            const int64_t logical = selectedGm.GetValue(routeOffset + position);
            if (logical < 0 || logical >= static_cast<int64_t>(tiling.numExperts)) {
                return false;
            }
            const int64_t owner = ownerRanksGm.GetValue(ownerOffset + static_cast<uint64_t>(logical));
            if (owner < 0 || owner >= static_cast<int64_t>(tiling.epSize)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool CopyTableValid(uint32_t layout, int64_t logical)
    {
        const int64_t rawCopies = copyCountsGm.GetValue(TableBase(layout) + static_cast<uint64_t>(logical));
        if (rawCopies <= 0 || rawCopies > static_cast<int64_t>(tiling.maxCopies)) {
            return false;
        }
        const uint64_t copyBase = CopyBase(layout, logical);
        const int64_t slotLimit = static_cast<int64_t>(tiling.epSize) * tiling.slotsPerRank;
        for (uint32_t copy = 0; copy < static_cast<uint32_t>(rawCopies); ++copy) {
            const int64_t slot = copySlotsGm.GetValue(copyBase + copy);
            if (slot < 0 || slot >= slotLimit) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool IsCanonicalCopy(uint32_t layout, int64_t logical, uint32_t copy, uint32_t copies)
    {
        const uint64_t copyBase = CopyBase(layout, logical);
        const int64_t rank = copySlotsGm.GetValue(copyBase + copy) / static_cast<int64_t>(tiling.slotsPerRank);
        for (uint32_t later = copy + 1U; later < copies; ++later) {
            const int64_t laterRank =
                copySlotsGm.GetValue(copyBase + later) / static_cast<int64_t>(tiling.slotsPerRank);
            if (laterRank == rank) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool RankAlreadyVisited(
        uint32_t layout,
        uint32_t token,
        int64_t logical,
        int64_t destinationRank,
        uint32_t groupSize)
    {
        if (groupSize == 1U) {
            if (destinationRank == static_cast<int64_t>(tiling.sourceRank)) {
                return true;
            }
        } else if (
            destinationRank / static_cast<int64_t>(groupSize)
            == static_cast<int64_t>(tiling.sourceRank) / static_cast<int64_t>(groupSize)) {
            return true;
        }
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        const uint64_t ownerOffset = TableBase(layout);
        for (uint32_t position = 0; position < tiling.topK; ++position) {
            const int64_t other = selectedGm.GetValue(routeOffset + position);
            if (other == logical) {
                continue;
            }
            const int64_t owner = ownerRanksGm.GetValue(ownerOffset + static_cast<uint64_t>(other));
            if (groupSize == 1U) {
                if (owner == destinationRank) {
                    return true;
                }
            } else if (
                owner / static_cast<int64_t>(groupSize) == destinationRank / static_cast<int64_t>(groupSize)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint32_t CommunicationScore(
        uint32_t layout,
        uint32_t token,
        int64_t logical,
        int64_t destinationRank)
    {
        uint32_t score = 0U;
        if (tiling.numLevels > 1U) {
            score = score * 2U + static_cast<uint32_t>(
                !RankAlreadyVisited(layout, token, logical, destinationRank, tiling.levelSize1));
        }
        if (tiling.numLevels > 0U) {
            score = score * 2U + static_cast<uint32_t>(
                !RankAlreadyVisited(layout, token, logical, destinationRank, tiling.levelSize0));
        }
        score = score * 2U
            + static_cast<uint32_t>(!RankAlreadyVisited(layout, token, logical, destinationRank, 1U));
        return score;
    }

    __aicore__ inline int64_t RouteHash(uint32_t token, int64_t logical) const
    {
        const int64_t ordinal = tokenOrdinalsGm.GetValue(token);
        uint64_t wrapped = static_cast<uint64_t>(ordinal) * 1000003ULL
            + static_cast<uint64_t>(logical) * 65537ULL + static_cast<uint64_t>(tiling.step) * 131ULL
            + static_cast<uint64_t>(tiling.layerSeed) * 17ULL;
        wrapped = wrapped * 48271ULL + 1ULL;
        int64_t signedValue = 0;
        if (wrapped == (1ULL << 63U)) {
            signedValue = -9223372036854775807LL - 1LL;
        } else if ((wrapped & (1ULL << 63U)) != 0ULL) {
            signedValue = -static_cast<int64_t>((~wrapped) + 1ULL);
        } else {
            signedValue = static_cast<int64_t>(wrapped);
        }
        int64_t value = signedValue % 2147483647LL;
        if (value < 0) {
            value += 2147483647LL;
        }
        return value % 1048573LL;
    }

    __aicore__ inline uint32_t TieMask(uint32_t layout, uint32_t token, int64_t logical)
    {
        const uint32_t copies = static_cast<uint32_t>(
            copyCountsGm.GetValue(TableBase(layout) + static_cast<uint64_t>(logical)));
        const uint64_t copyBase = CopyBase(layout, logical);
        uint32_t bestScore = 0xffffffffU;
        uint32_t tieMask = 0U;
        for (uint32_t copy = 0; copy < copies; ++copy) {
            if (!IsCanonicalCopy(layout, logical, copy, copies)) {
                continue;
            }
            const int64_t rank =
                copySlotsGm.GetValue(copyBase + copy) / static_cast<int64_t>(tiling.slotsPerRank);
            const uint32_t score = CommunicationScore(layout, token, logical, rank);
            if (score < bestScore) {
                bestScore = score;
                tieMask = 1U << copy;
            } else if (score == bestScore) {
                tieMask |= 1U << copy;
            }
        }
        return tieMask;
    }

    __aicore__ inline uint32_t PopCount(uint32_t value) const
    {
        uint32_t count = 0U;
        while (value != 0U) {
            count += value & 1U;
            value >>= 1U;
        }
        return count;
    }

    __aicore__ inline uint32_t FirstCopy(uint32_t mask) const
    {
        for (uint32_t copy = 0; copy < tiling.maxCopies; ++copy) {
            if ((mask & (1U << copy)) != 0U) {
                return copy;
            }
        }
        return tiling.maxCopies;
    }

    __aicore__ inline void WriteChoice(uint32_t layout, uint32_t token, int64_t logical, int64_t slot)
    {
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        const uint64_t outputOffset = OutputBase(layout, token);
        for (uint32_t position = 0; position < tiling.topK; ++position) {
            if (selectedGm.GetValue(routeOffset + position) == logical) {
                physicalGm.SetValue(outputOffset + position, slot);
            }
        }
    }

    __aicore__ inline void BuildTokenRecords(
        uint32_t layout,
        uint32_t token,
        uint32_t block,
        uint64_t &recordCount)
    {
        if (!TokenDomainValid(layout, token)) {
            return;
        }
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        for (uint32_t position = 0; position < tiling.topK; ++position) {
            const int64_t logical = selectedGm.GetValue(routeOffset + position);
            bool first = true;
            for (uint32_t prior = 0; prior < position; ++prior) {
                if (selectedGm.GetValue(routeOffset + prior) == logical) {
                    first = false;
                    break;
                }
            }
            if (!first || !CopyTableValid(layout, logical)) {
                continue;
            }
            int64_t multiplicity = 0;
            for (uint32_t other = 0; other < tiling.topK; ++other) {
                multiplicity += static_cast<int64_t>(selectedGm.GetValue(routeOffset + other) == logical);
            }
            const int64_t rawCopies = copyCountsGm.GetValue(TableBase(layout) + static_cast<uint64_t>(logical));
            if (rawCopies == 1LL) {
                const int64_t slot = copySlotsGm.GetValue(CopyBase(layout, logical));
                WriteChoice(layout, token, logical, slot);
                const int64_t rank = slot / static_cast<int64_t>(tiling.slotsPerRank);
                const uint64_t loadIndex = privateRankLoadOffset + PrivateRankBase(block)
                    + RankStorageIndex(layout, static_cast<uint32_t>(rank));
                intWorkspaceGm.SetValue(loadIndex, intWorkspaceGm.GetValue(loadIndex) + multiplicity);
                continue;
            }
            const uint32_t mask = TieMask(layout, token, logical);
            const uint32_t tieCount = PopCount(mask);
            if (tieCount == 0U) {
                continue;
            }
            if (tieCount == 1U) {
                const uint32_t copy = FirstCopy(mask);
                const int64_t slot = copySlotsGm.GetValue(CopyBase(layout, logical) + copy);
                WriteChoice(layout, token, logical, slot);
                const int64_t rank = slot / static_cast<int64_t>(tiling.slotsPerRank);
                const uint64_t loadIndex = privateRankLoadOffset + PrivateRankBase(block)
                    + RankStorageIndex(layout, static_cast<uint32_t>(rank));
                intWorkspaceGm.SetValue(loadIndex, intWorkspaceGm.GetValue(loadIndex) + multiplicity);
                continue;
            }

            const uint64_t record = static_cast<uint64_t>(layout) * recordStride
                + static_cast<uint64_t>(token) * tiling.topK + position;
            intWorkspaceGm.SetValue(logicalOffset + record, logical);
            intWorkspaceGm.SetValue(multiplicityOffset + record, multiplicity);
            intWorkspaceGm.SetValue(maskOffset + record, static_cast<int64_t>(mask));
            intWorkspaceGm.SetValue(hashOffset + record, RouteHash(token, logical));
            intWorkspaceGm.SetValue(
                tupleKeyOffset + record,
                static_cast<int64_t>(DestinationTupleKey(layout, logical, mask)));
            intWorkspaceGm.SetValue(
                orderOffset + static_cast<uint64_t>(layout) * tiling.sortStride
                    + static_cast<uint64_t>(block) * tiling.runCapacity + recordCount,
                record);
            ++recordCount;

            const uint64_t bucket = PrivateBucketBase(block) + BucketStorageIndex(layout, logical, mask);
            intWorkspaceGm.SetValue(
                privateBucketTotalOffset + bucket,
                intWorkspaceGm.GetValue(privateBucketTotalOffset + bucket) + multiplicity);
            if (multiplicity != 1) {
                intWorkspaceGm.SetValue(privateBucketUnitOffset + bucket, 0LL);
            }
        }
    }

    __aicore__ inline uint32_t CollectDestinations(
        uint32_t layout,
        int64_t logical,
        uint32_t mask,
        int64_t ranks[8],
        uint32_t copies[8])
    {
        uint32_t count = 0U;
        const uint64_t copyBase = CopyBase(layout, logical);
        for (uint32_t copy = 0; copy < tiling.maxCopies; ++copy) {
            if ((mask & (1U << copy)) == 0U) {
                continue;
            }
            const int64_t rank =
                copySlotsGm.GetValue(copyBase + copy) / static_cast<int64_t>(tiling.slotsPerRank);
            uint32_t insert = count;
            while (insert > 0U && ranks[insert - 1U] > rank) {
                ranks[insert] = ranks[insert - 1U];
                copies[insert] = copies[insert - 1U];
                --insert;
            }
            ranks[insert] = rank;
            copies[insert] = copy;
            ++count;
        }
        return count;
    }

    __aicore__ inline uint64_t DestinationTupleKey(uint32_t layout, int64_t logical, uint32_t mask)
    {
        int64_t ranks[8];
        uint32_t copies[8];
        const uint32_t count = CollectDestinations(layout, logical, mask, ranks, copies);
        uint64_t key = 0ULL;
        for (uint32_t index = 0; index < 8U; ++index) {
            const uint64_t value = index < count ? static_cast<uint64_t>(ranks[index] + 1LL) : 0ULL;
            key = (key << 7U) | value;
        }
        return (key << 8U) | static_cast<uint64_t>(mask);
    }

    __aicore__ inline uint32_t RecordToken(uint32_t layout, uint64_t record) const
    {
        const uint64_t local = record - static_cast<uint64_t>(layout) * recordStride;
        return static_cast<uint32_t>(local / tiling.topK);
    }

    __aicore__ inline bool RecordLess(uint32_t layout, uint64_t recordA, uint64_t recordB)
    {
        const int64_t logicalA = intWorkspaceGm.GetValue(logicalOffset + recordA);
        const int64_t logicalB = intWorkspaceGm.GetValue(logicalOffset + recordB);
        const uint32_t maskA = static_cast<uint32_t>(intWorkspaceGm.GetValue(maskOffset + recordA));
        const uint32_t maskB = static_cast<uint32_t>(intWorkspaceGm.GetValue(maskOffset + recordB));
        const int64_t totalA = intWorkspaceGm.GetValue(
            bucketTotalOffset + BucketStorageIndex(layout, logicalA, maskA));
        const int64_t totalB = intWorkspaceGm.GetValue(
            bucketTotalOffset + BucketStorageIndex(layout, logicalB, maskB));
        if (totalA != totalB) {
            return totalA > totalB;
        }
        if (logicalA != logicalB) {
            return logicalA < logicalB;
        }
        const uint64_t tupleKeyA = static_cast<uint64_t>(intWorkspaceGm.GetValue(tupleKeyOffset + recordA));
        const uint64_t tupleKeyB = static_cast<uint64_t>(intWorkspaceGm.GetValue(tupleKeyOffset + recordB));
        if (tupleKeyA != tupleKeyB) {
            return tupleKeyA < tupleKeyB;
        }
        const int64_t hashA = intWorkspaceGm.GetValue(hashOffset + recordA);
        const int64_t hashB = intWorkspaceGm.GetValue(hashOffset + recordB);
        if (hashA != hashB) {
            return hashA < hashB;
        }
        const int64_t ordinalA = tokenOrdinalsGm.GetValue(RecordToken(layout, recordA));
        const int64_t ordinalB = tokenOrdinalsGm.GetValue(RecordToken(layout, recordB));
        if (ordinalA != ordinalB) {
            return ordinalA < ordinalB;
        }
        return recordA < recordB;
    }

    __aicore__ inline uint64_t SortBufferOffset(bool orderBuffer) const
    {
        return orderBuffer ? orderOffset : temporaryOffset;
    }

    __aicore__ inline void MergeRanges(
        uint32_t layout,
        uint64_t sourceLeft,
        uint64_t leftCount,
        uint64_t sourceRight,
        uint64_t rightCount,
        uint64_t destination)
    {
        uint64_t first = 0ULL;
        uint64_t second = 0ULL;
        uint64_t output = 0ULL;
        while (first < leftCount && second < rightCount) {
            const uint64_t firstRecord = static_cast<uint64_t>(intWorkspaceGm.GetValue(sourceLeft + first));
            const uint64_t secondRecord = static_cast<uint64_t>(intWorkspaceGm.GetValue(sourceRight + second));
            if (!RecordLess(layout, secondRecord, firstRecord)) {
                intWorkspaceGm.SetValue(destination + output, static_cast<int64_t>(firstRecord));
                ++first;
            } else {
                intWorkspaceGm.SetValue(destination + output, static_cast<int64_t>(secondRecord));
                ++second;
            }
            ++output;
        }
        while (first < leftCount) {
            intWorkspaceGm.SetValue(destination + output, intWorkspaceGm.GetValue(sourceLeft + first));
            ++first;
            ++output;
        }
        while (second < rightCount) {
            intWorkspaceGm.SetValue(destination + output, intWorkspaceGm.GetValue(sourceRight + second));
            ++second;
            ++output;
        }
    }

    __aicore__ inline void SortLocalRun(uint32_t layout, uint32_t block, uint64_t recordCount)
    {
        if (recordCount < 2ULL) {
            return;
        }
        const uint64_t runOffset = static_cast<uint64_t>(layout) * tiling.sortStride
            + static_cast<uint64_t>(block) * tiling.runCapacity;
        bool sourceOrder = true;
        for (uint64_t width = 1ULL; width < recordCount; width *= 2ULL) {
            const uint64_t sourceBase = SortBufferOffset(sourceOrder) + runOffset;
            const uint64_t destinationBase = SortBufferOffset(!sourceOrder) + runOffset;
            for (uint64_t left = 0ULL; left < recordCount; left += 2ULL * width) {
                const uint64_t middle = left + width < recordCount ? left + width : recordCount;
                const uint64_t right = left + 2ULL * width < recordCount ? left + 2ULL * width : recordCount;
                MergeRanges(
                    layout,
                    sourceBase + left,
                    middle - left,
                    sourceBase + middle,
                    right - middle,
                    destinationBase + left);
            }
            sourceOrder = !sourceOrder;
        }
        if (!sourceOrder) {
            const uint64_t sourceBase = temporaryOffset + runOffset;
            const uint64_t destinationBase = orderOffset + runOffset;
            for (uint64_t index = 0ULL; index < recordCount; ++index) {
                intWorkspaceGm.SetValue(
                    destinationBase + index,
                    intWorkspaceGm.GetValue(sourceBase + index));
            }
        }
    }

    __aicore__ inline uint64_t RunCount(uint32_t layout, uint32_t firstBlock, uint32_t blockSpan) const
    {
        uint64_t count = 0ULL;
        const uint32_t lastBlock = firstBlock + blockSpan < tiling.blockCount
            ? firstBlock + blockSpan
            : tiling.blockCount;
        for (uint32_t block = firstBlock; block < lastBlock; ++block) {
            count += static_cast<uint64_t>(intWorkspaceGm.GetValue(
                recordCountOffset + static_cast<uint64_t>(block) * 16ULL + layout));
        }
        return count;
    }

    __aicore__ inline uint64_t TotalRecordCount(uint32_t layout) const
    {
        return RunCount(layout, 0U, tiling.blockCount);
    }

    __aicore__ inline uint64_t HardwareSortFlagOffset(uint32_t layout) const
    {
        if (tiling.blockCount > 1U) {
            return recordCountOffset + static_cast<uint64_t>(layout) * 16ULL + 8ULL;
        }
        return recordCountOffset + 8ULL + layout;
    }

    __aicore__ inline void SetHardwareSortFlag(uint32_t layout, bool enabled)
    {
        intWorkspaceGm.SetValue(HardwareSortFlagOffset(layout), enabled ? 1LL : 0LL);
    }

    __aicore__ inline bool HardwareSortFlag(uint32_t layout) const
    {
        return intWorkspaceGm.GetValue(HardwareSortFlagOffset(layout)) != 0LL;
    }

    __aicore__ inline bool HardwareRecordLess(uint32_t layout, uint32_t lhs, uint32_t rhs) const
    {
        const int64_t lhsOrdinal = tokenOrdinalsGm.GetValue(RecordToken(layout, lhs));
        const int64_t rhsOrdinal = tokenOrdinalsGm.GetValue(RecordToken(layout, rhs));
        return lhsOrdinal != rhsOrdinal ? lhsOrdinal < rhsOrdinal : lhs < rhs;
    }

    __aicore__ inline bool TryHardwareSortSingleBucket(uint32_t layout)
    {
        const uint64_t count64 = TotalRecordCount(layout);
        if (count64 > static_cast<uint64_t>(kHardwareSortMaxRecords)) {
            return false;
        }
        if (count64 == 0ULL) {
            return true;
        }
        const uint32_t count = static_cast<uint32_t>(count64);
        AscendC::LocalTensor<float> scratch = hashSortScratchBuf.Get<float>();
        AscendC::LocalTensor<float> scores = scratch;
        AscendC::LocalTensor<uint32_t> scoreBits = scores.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<uint32_t> payloads =
            scratch[kHardwareSortCapacity].ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> packed = hashSortPackedBuf.Get<float>();

        bool first = true;
        int64_t firstLogical = -1LL;
        int64_t firstMask = -1LL;
        int64_t firstTupleKey = 0LL;
        uint32_t cursor = 0U;
        const uint64_t layoutBase = static_cast<uint64_t>(layout) * tiling.sortStride;
        for (uint32_t sourceBlock = 0U; sourceBlock < tiling.blockCount; ++sourceBlock) {
            const uint64_t sourceCount = static_cast<uint64_t>(intWorkspaceGm.GetValue(
                recordCountOffset + static_cast<uint64_t>(sourceBlock) * 16ULL + layout));
            const uint64_t sourceBase = orderOffset + layoutBase
                + static_cast<uint64_t>(sourceBlock) * tiling.runCapacity;
            for (uint64_t index = 0ULL; index < sourceCount; ++index) {
                const int64_t storedRecord = intWorkspaceGm.GetValue(sourceBase + index);
                if (storedRecord < 0LL || static_cast<uint64_t>(storedRecord) > 0xffffffffULL) {
                    return false;
                }
                const uint32_t record = static_cast<uint32_t>(storedRecord);
                const int64_t logical = intWorkspaceGm.GetValue(logicalOffset + record);
                const int64_t mask = intWorkspaceGm.GetValue(maskOffset + record);
                const int64_t tupleKey = intWorkspaceGm.GetValue(tupleKeyOffset + record);
                if (first) {
                    firstLogical = logical;
                    firstMask = mask;
                    firstTupleKey = tupleKey;
                    first = false;
                } else if (
                    logical != firstLogical || mask != firstMask || tupleKey != firstTupleKey) {
                    return false;
                }
                const int64_t hash = intWorkspaceGm.GetValue(hashOffset + record);
                if (hash < 0LL || hash > static_cast<int64_t>(kRouteHashMax)) {
                    return false;
                }
                const uint32_t score = kRouteHashMax - static_cast<uint32_t>(hash);
                scoreBits.SetValue(cursor, kFloatIntegerBiasBits + score);
                payloads.SetValue(cursor, record);
                ++cursor;
            }
        }
        if (cursor != count) {
            return false;
        }

        const uint32_t paddedCount = (count + 31U) / 32U * 32U;
        for (uint32_t index = count; index < paddedCount; ++index) {
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

        AscendC::LocalTensor<uint32_t> sortedWords = source.ReinterpretCast<uint32_t>();
        uint32_t runStart = 0U;
        while (runStart < count) {
            const uint32_t scoreWord = sortedWords.GetValue(2U * runStart);
            uint32_t runEnd = runStart + 1U;
            while (runEnd < count && sortedWords.GetValue(2U * runEnd) == scoreWord) {
                ++runEnd;
            }
            for (uint32_t index = runStart + 1U; index < runEnd; ++index) {
                const uint32_t record = sortedWords.GetValue(2U * index + 1U);
                uint32_t insertion = index;
                while (insertion > runStart) {
                    const uint32_t previous = sortedWords.GetValue(2U * (insertion - 1U) + 1U);
                    if (!HardwareRecordLess(layout, record, previous)) {
                        break;
                    }
                    sortedWords.SetValue(2U * insertion + 1U, previous);
                    --insertion;
                }
                sortedWords.SetValue(2U * insertion + 1U, record);
            }
            runStart = runEnd;
        }

        const uint64_t destinationBase = orderOffset + layoutBase;
        for (uint32_t index = 0U; index < count; ++index) {
            intWorkspaceGm.SetValue(
                destinationBase + index,
                static_cast<int64_t>(sortedWords.GetValue(2U * index + 1U)));
        }
        return true;
    }

    __aicore__ inline void MergeRuns(uint32_t layout, uint32_t firstBlock, uint32_t span, bool sourceOrder)
    {
        if (firstBlock >= tiling.blockCount) {
            return;
        }
        const uint64_t leftCount = RunCount(layout, firstBlock, span);
        const uint64_t rightCount = RunCount(layout, firstBlock + span, span);
        const uint64_t layoutBase = static_cast<uint64_t>(layout) * tiling.sortStride;
        const uint64_t leftBase = layoutBase + static_cast<uint64_t>(firstBlock) * tiling.runCapacity;
        const uint64_t rightBase = layoutBase + static_cast<uint64_t>(firstBlock + span) * tiling.runCapacity;
        MergeRanges(
            layout,
            SortBufferOffset(sourceOrder) + leftBase,
            leftCount,
            SortBufferOffset(sourceOrder) + rightBase,
            rightCount,
            SortBufferOffset(!sourceOrder) + leftBase);
    }

    __aicore__ inline void WaterfillQuotas(
        uint32_t layout,
        const int64_t ranks[8],
        uint32_t count,
        int64_t total,
        int64_t quotas[8])
    {
        uint32_t order[8];
        for (uint32_t index = 0; index < count; ++index) {
            order[index] = index;
            uint32_t insert = index;
            while (insert > 0U) {
                const uint32_t current = order[insert];
                const uint32_t prior = order[insert - 1U];
                const int64_t currentLoad = intWorkspaceGm.GetValue(
                    rankLoadOffset + RankStorageIndex(layout, static_cast<uint32_t>(ranks[current])));
                const int64_t priorLoad = intWorkspaceGm.GetValue(
                    rankLoadOffset + RankStorageIndex(layout, static_cast<uint32_t>(ranks[prior])));
                if (priorLoad < currentLoad || (priorLoad == currentLoad && ranks[prior] < ranks[current])) {
                    break;
                }
                order[insert] = prior;
                order[insert - 1U] = current;
                --insert;
            }
        }

        int64_t remaining = total > 0 ? total : 0;
        uint32_t active = 1U;
        while (active < count) {
            const int64_t currentLoad = intWorkspaceGm.GetValue(
                rankLoadOffset
                    + RankStorageIndex(layout, static_cast<uint32_t>(ranks[order[active - 1U]])));
            const int64_t nextLoad = intWorkspaceGm.GetValue(
                rankLoadOffset + RankStorageIndex(layout, static_cast<uint32_t>(ranks[order[active]])));
            const int64_t difference = nextLoad > currentLoad ? nextLoad - currentLoad : 0;
            const int64_t required = difference * static_cast<int64_t>(active);
            if (required > remaining) {
                break;
            }
            if (required > 0) {
                const int64_t increment = required / static_cast<int64_t>(active);
                const int64_t extra = required % static_cast<int64_t>(active);
                for (uint32_t index = 0; index < active; ++index) {
                    quotas[order[index]] += increment + static_cast<int64_t>(index < static_cast<uint32_t>(extra));
                }
                remaining -= required;
            }
            ++active;
        }
        const int64_t increment = remaining / static_cast<int64_t>(active);
        const int64_t extra = remaining % static_cast<int64_t>(active);
        for (uint32_t index = 0; index < active; ++index) {
            quotas[order[index]] += increment + static_cast<int64_t>(index < static_cast<uint32_t>(extra));
        }
    }

    __aicore__ inline bool BuildQuotas(
        uint32_t layout,
        int64_t logical,
        uint32_t mask,
        const int64_t ranks[8],
        const uint32_t copies[8],
        uint32_t count,
        int64_t total,
        int64_t quotas[8])
    {
        for (uint32_t index = 0; index < count; ++index) {
            quotas[index] = 0;
        }
        const uint64_t bucket = BucketIndex(layout, logical, mask);
        const int64_t configured = quotaConfiguredGm.GetValue(bucket);
        if (configured != 0 && configured != 1) {
            return false;
        }
        if (configured == 0) {
            WaterfillQuotas(layout, ranks, count, total, quotas);
            return true;
        }

        const uint64_t quotaBase = bucket * tiling.maxCopies;
        int64_t weights[8];
        int64_t weightTotal = 0;
        for (uint32_t index = 0; index < count; ++index) {
            const int64_t weight = quotaWeightsGm.GetValue(quotaBase + copies[index]);
            if (weight < 0 || weight > 2147483647LL) {
                return false;
            }
            weights[index] = weight;
            weightTotal += weight;
        }
        if (weightTotal <= 0) {
            WaterfillQuotas(layout, ranks, count, total, quotas);
            return true;
        }

        int64_t fractions[8];
        int64_t roundedTotal = 0;
        for (uint32_t index = 0; index < count; ++index) {
            const int64_t product = total * weights[index];
            quotas[index] = product / weightTotal;
            fractions[index] = product % weightTotal;
            roundedTotal += quotas[index];
        }
        int64_t remainder = total - roundedTotal;
        bool receivedExtra[8] = {false, false, false, false, false, false, false, false};
        while (remainder > 0) {
            int32_t best = -1;
            for (uint32_t index = 0; index < count; ++index) {
                if (receivedExtra[index]) {
                    continue;
                }
                if (
                    best < 0 || fractions[index] > fractions[best]
                    || (fractions[index] == fractions[best] && ranks[index] < ranks[best])) {
                    best = static_cast<int32_t>(index);
                }
            }
            if (best < 0) {
                break;
            }
            ++quotas[best];
            receivedExtra[best] = true;
            --remainder;
        }
        return true;
    }

    __aicore__ inline void AssignRecord(
        uint32_t layout,
        uint64_t record,
        int64_t logical,
        uint32_t copy)
    {
        const uint32_t token = RecordToken(layout, record);
        const int64_t slot = copySlotsGm.GetValue(CopyBase(layout, logical) + copy);
        WriteChoice(layout, token, logical, slot);
    }

    __aicore__ inline void AssignBucket(
        uint32_t layout,
        uint64_t sortedBase,
        uint64_t start,
        uint64_t end)
    {
        const uint64_t firstRecord =
            static_cast<uint64_t>(intWorkspaceGm.GetValue(sortedBase + start));
        const int64_t logical = intWorkspaceGm.GetValue(logicalOffset + firstRecord);
        const uint32_t mask = static_cast<uint32_t>(intWorkspaceGm.GetValue(maskOffset + firstRecord));
        const uint64_t bucket = BucketStorageIndex(layout, logical, mask);
        const int64_t total = intWorkspaceGm.GetValue(bucketTotalOffset + bucket);

        int64_t ranks[8];
        uint32_t copies[8];
        const uint32_t destinationCount = CollectDestinations(layout, logical, mask, ranks, copies);
        if (destinationCount == 0U) {
            return;
        }
        int64_t quotas[8];
        if (!BuildQuotas(layout, logical, mask, ranks, copies, destinationCount, total, quotas)) {
            return;
        }
        int64_t assigned[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const bool unitRoutes = intWorkspaceGm.GetValue(bucketUnitOffset + bucket) != 0;

        if (unitRoutes) {
            int64_t quotaTotal = 0;
            for (uint32_t index = 0; index < destinationCount; ++index) {
                quotaTotal += quotas[index] > 0 ? quotas[index] : 0;
            }
            uint32_t destinationIndex = 0U;
            int64_t consumed = 0;
            for (uint64_t position = 0U; position < end - start; ++position) {
                while (
                    destinationIndex + 1U < destinationCount
                    && static_cast<int64_t>(position)
                        >= consumed + (quotas[destinationIndex] > 0 ? quotas[destinationIndex] : 0)) {
                    consumed += quotas[destinationIndex] > 0 ? quotas[destinationIndex] : 0;
                    ++destinationIndex;
                }
                const uint32_t chosen = quotaTotal > 0
                    ? destinationIndex
                    : static_cast<uint32_t>(position % destinationCount);
                const uint64_t record = static_cast<uint64_t>(
                    intWorkspaceGm.GetValue(sortedBase + start + position));
                AssignRecord(layout, record, logical, copies[chosen]);
                ++assigned[chosen];
            }
        } else {
            for (uint64_t position = start; position < end; ++position) {
                const uint64_t record =
                    static_cast<uint64_t>(intWorkspaceGm.GetValue(sortedBase + position));
                const int64_t units = intWorkspaceGm.GetValue(multiplicityOffset + record);
                uint32_t chosen = 0U;
                for (uint32_t index = 1U; index < destinationCount; ++index) {
                    const int64_t candidateDeficit = assigned[index] + units - quotas[index];
                    const int64_t chosenDeficit = assigned[chosen] + units - quotas[chosen];
                    const int64_t candidateLoad = intWorkspaceGm.GetValue(
                        rankLoadOffset + RankStorageIndex(layout, static_cast<uint32_t>(ranks[index])))
                        + assigned[index] + units;
                    const int64_t chosenLoad = intWorkspaceGm.GetValue(
                        rankLoadOffset + RankStorageIndex(layout, static_cast<uint32_t>(ranks[chosen])))
                        + assigned[chosen] + units;
                    if (
                        candidateDeficit < chosenDeficit
                        || (candidateDeficit == chosenDeficit && candidateLoad < chosenLoad)
                        || (candidateDeficit == chosenDeficit && candidateLoad == chosenLoad
                            && ranks[index] < ranks[chosen])) {
                        chosen = index;
                    }
                }
                AssignRecord(layout, record, logical, copies[chosen]);
                assigned[chosen] += units;
            }
        }

        for (uint32_t index = 0; index < destinationCount; ++index) {
            const uint64_t loadIndex = rankLoadOffset
                + RankStorageIndex(layout, static_cast<uint32_t>(ranks[index]));
            intWorkspaceGm.SetValue(loadIndex, intWorkspaceGm.GetValue(loadIndex) + assigned[index]);
        }
    }

    __aicore__ inline void AssignBuckets(uint32_t layout, uint64_t recordCount, bool orderBuffer)
    {
        const uint64_t sortedBase = SortBufferOffset(orderBuffer)
            + static_cast<uint64_t>(layout) * tiling.sortStride;
        uint64_t start = 0ULL;
        while (start < recordCount) {
            const uint64_t firstRecord =
                static_cast<uint64_t>(intWorkspaceGm.GetValue(sortedBase + start));
            const int64_t logical = intWorkspaceGm.GetValue(logicalOffset + firstRecord);
            const int64_t mask = intWorkspaceGm.GetValue(maskOffset + firstRecord);
            uint64_t end = start + 1ULL;
            while (end < recordCount) {
                const uint64_t record =
                    static_cast<uint64_t>(intWorkspaceGm.GetValue(sortedBase + end));
                if (
                    intWorkspaceGm.GetValue(logicalOffset + record) != logical
                    || intWorkspaceGm.GetValue(maskOffset + record) != mask) {
                    break;
                }
                ++end;
            }
            AssignBucket(layout, sortedBase, start, end);
            start = end;
        }
    }

    __aicore__ inline uint32_t GroupSize(uint32_t level) const
    {
        if (level == 0U && tiling.numLevels > 0U) {
            return tiling.levelSize0;
        }
        if (level == 1U && tiling.numLevels > 1U) {
            return tiling.levelSize1;
        }
        return 1U;
    }

    __aicore__ inline void AccumulateStats(
        uint32_t layout,
        uint32_t block,
        uint32_t firstToken,
        uint32_t lastToken)
    {
        const uint64_t layoutStride = static_cast<uint64_t>(tiling.groupWidth) + tiling.epSize;
        const uint64_t statsBase = statsOffset + static_cast<uint64_t>(block) * tiling.statsStride
            + static_cast<uint64_t>(layout) * layoutStride;
        for (uint32_t token = firstToken; token < lastToken; ++token) {
            const uint64_t outputOffset = OutputBase(layout, token);
            uint64_t visitedGroups[3] = {0ULL, 0ULL, 0ULL};
            for (uint32_t position = 0; position < tiling.topK; ++position) {
                const int64_t slot = physicalGm.GetValue(outputOffset + position);
                if (slot < 0) {
                    continue;
                }
                const int64_t rank = slot / static_cast<int64_t>(tiling.slotsPerRank);
                const uint64_t assignmentIndex = statsBase + tiling.groupWidth + static_cast<uint64_t>(rank);
                intWorkspaceGm.SetValue(
                    assignmentIndex,
                    intWorkspaceGm.GetValue(assignmentIndex) + 1LL);

                uint32_t groupOffset = 0U;
                for (uint32_t level = 0; level <= tiling.numLevels; ++level) {
                    const uint32_t groupSize = GroupSize(level);
                    const uint32_t numGroups = tiling.epSize / groupSize;
                    const int64_t group = rank / static_cast<int64_t>(groupSize);
                    const uint64_t bit = 1ULL << static_cast<uint32_t>(group);
                    if ((visitedGroups[level] & bit) == 0ULL) {
                        visitedGroups[level] |= bit;
                        const uint64_t groupIndex = statsBase + groupOffset + static_cast<uint64_t>(group);
                        intWorkspaceGm.SetValue(
                            groupIndex,
                            intWorkspaceGm.GetValue(groupIndex) + 1LL);
                    }
                    groupOffset += numGroups;
                }
            }
        }
    }

    __aicore__ inline uint64_t FloatOutputChunk(uint64_t elements) const
    {
        const uint64_t unaligned = (elements + tiling.blockCount - 1ULL) / tiling.blockCount;
        return (unaligned + 15ULL) / 16ULL * 16ULL;
    }

    __aicore__ inline void ReduceStats(uint32_t block)
    {
        const uint32_t layoutStride = tiling.groupWidth + tiling.epSize;
        const uint64_t groupElements = 2ULL * tiling.groupWidth;
        const uint64_t groupChunk = FloatOutputChunk(groupElements);
        const uint64_t firstGroup = static_cast<uint64_t>(block) * groupChunk;
        const uint64_t lastGroup = firstGroup + groupChunk < groupElements
            ? firstGroup + groupChunk
            : groupElements;
        for (uint64_t output = firstGroup; output < lastGroup; ++output) {
            const uint32_t layout = static_cast<uint32_t>(output / tiling.groupWidth);
            const uint32_t local = static_cast<uint32_t>(output % tiling.groupWidth);
            int64_t total = 0LL;
            for (uint32_t sourceBlock = 0; sourceBlock < tiling.blockCount; ++sourceBlock) {
                total += intWorkspaceGm.GetValue(
                    statsOffset + static_cast<uint64_t>(sourceBlock) * tiling.statsStride
                        + static_cast<uint64_t>(layout) * layoutStride + local);
            }
            groupCountsGm.SetValue(output, static_cast<float>(total));
        }

        const uint64_t assignmentElements = 2ULL * tiling.epSize;
        const uint64_t assignmentChunk = FloatOutputChunk(assignmentElements);
        const uint64_t firstAssignment = static_cast<uint64_t>(block) * assignmentChunk;
        const uint64_t lastAssignment = firstAssignment + assignmentChunk < assignmentElements
            ? firstAssignment + assignmentChunk
            : assignmentElements;
        for (uint64_t output = firstAssignment; output < lastAssignment; ++output) {
            const uint32_t layout = static_cast<uint32_t>(output / tiling.epSize);
            const uint32_t rank = static_cast<uint32_t>(output % tiling.epSize);
            int64_t total = 0LL;
            for (uint32_t sourceBlock = 0; sourceBlock < tiling.blockCount; ++sourceBlock) {
                total += intWorkspaceGm.GetValue(
                    statsOffset + static_cast<uint64_t>(sourceBlock) * tiling.statsStride
                        + static_cast<uint64_t>(layout) * layoutStride + tiling.groupWidth + rank);
            }
            assignmentCountsGm.SetValue(output, static_cast<float>(total));
        }
    }

    AscendC::GlobalTensor<int64_t> selectedGm;
    AscendC::GlobalTensor<int64_t> copySlotsGm;
    AscendC::GlobalTensor<int64_t> copyCountsGm;
    AscendC::GlobalTensor<int64_t> ownerRanksGm;
    AscendC::GlobalTensor<int64_t> quotaWeightsGm;
    AscendC::GlobalTensor<int64_t> quotaConfiguredGm;
    AscendC::GlobalTensor<int64_t> tokenOrdinalsGm;
    AscendC::GlobalTensor<int64_t> physicalGm;
    AscendC::GlobalTensor<float> groupCountsGm;
    AscendC::GlobalTensor<float> assignmentCountsGm;
    AscendC::GlobalTensor<int64_t> intWorkspaceGm;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hashSortScratchBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hashSortPackedBuf;
    HiermoeQuotaMapTilingData tiling;
    uint64_t recordStride;
    uint64_t recordCapacity;
    uint64_t logicalOffset;
    uint64_t multiplicityOffset;
    uint64_t maskOffset;
    uint64_t hashOffset;
    uint64_t tupleKeyOffset;
    uint64_t orderOffset;
    uint64_t temporaryOffset;
    uint64_t privateBucketTotalOffset;
    uint64_t privateBucketUnitOffset;
    uint64_t bucketTotalOffset;
    uint64_t bucketUnitOffset;
    uint64_t privateRankLoadOffset;
    uint64_t rankLoadOffset;
    uint64_t recordCountOffset;
    uint64_t statsOffset;
};

extern "C" __global__ __aicore__ void hiermoe_quota_map(
    GM_ADDR selected,
    GM_ADDR copySlots,
    GM_ADDR copyCounts,
    GM_ADDR ownerRanks,
    GM_ADDR quotaWeights,
    GM_ADDR quotaConfigured,
    GM_ADDR tokenOrdinals,
    GM_ADDR physical,
    GM_ADDR groupCounts,
    GM_ADDR assignmentCounts,
    GM_ADDR intWorkspace,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeQuotaMapTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeQuotaMap op;
    op.Init(
        selected,
        copySlots,
        copyCounts,
        ownerRanks,
        quotaWeights,
        quotaConfigured,
        tokenOrdinals,
        physical,
        groupCounts,
        assignmentCounts,
        intWorkspace,
        tilingData);
    op.Process();
}
