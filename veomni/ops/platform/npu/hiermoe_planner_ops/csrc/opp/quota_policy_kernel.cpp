#include "kernel_operator.h"

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

class KernelHiermoeQuotaPolicy {
public:
    __aicore__ inline void Init(
        GM_ADDR sampleRoutes,
        GM_ADDR sampleMultiplicity,
        GM_ADDR sampleSources,
        GM_ADDR sampleOrdinals,
        GM_ADDR assignmentCounts,
        GM_ADDR layouts,
        GM_ADDR ownerSlots,
        GM_ADDR quotaWeights,
        GM_ADDR quotaConfigured,
        GM_ADDR compactRows,
        GM_ADDR rowCounts,
        GM_ADDR digest,
        GM_ADDR intWorkspace,
        const HiermoeQuotaPolicyTilingData &tiling)
    {
        this->tiling = tiling;
        sampleRoutesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleRoutes));
        sampleMultiplicityGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleMultiplicity));
        sampleSourcesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleSources));
        sampleOrdinalsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(sampleOrdinals));
        assignmentCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(assignmentCounts));
        layoutsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(layouts));
        ownerSlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ownerSlots));
        quotaWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(quotaWeights));
        quotaConfiguredGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(quotaConfigured));
        compactRowsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(compactRows));
        rowCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(rowCounts));
        digestGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(digest));
        intWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(intWorkspace));

        sourceExpertCount = static_cast<uint64_t>(tiling.epSize) * tiling.numExperts;
        sourceExpertStride = Align8(tiling.numExperts);
        sourceBucketStride = Align8(static_cast<uint64_t>(tiling.numExperts) * tiling.maskCount);
        bucketCount = static_cast<uint64_t>(tiling.epSize) * sourceBucketStride;
        orderStride = Align8(tiling.rowCapacity);
        orderCapacity = static_cast<uint64_t>(tiling.epSize) * orderStride;
        rankStride = Align8(tiling.epSize);
        copySlotOffset = 0U;
        copyCountOffset = Align8(copySlotOffset + static_cast<uint64_t>(tiling.numExperts) * tiling.maxCopies);
        canonicalCountOffset = Align8(copyCountOffset + tiling.numExperts);
        singletonRankOffset = Align8(canonicalCountOffset + tiling.numExperts);
        sampleBucketOffset = Align8(singletonRankOffset + tiling.numExperts);
        projectedBucketOffset = Align8(sampleBucketOffset + bucketCount);
        activeMaskListOffset = Align8(projectedBucketOffset + bucketCount);
        activeMaskCountOffset = Align8(activeMaskListOffset + bucketCount);
        observedCountOffset = Align8(
            activeMaskCountOffset + static_cast<uint64_t>(tiling.epSize) * sourceExpertStride);
        orderOffset = Align8(observedCountOffset + static_cast<uint64_t>(tiling.epSize) * sourceExpertStride);
        temporaryOffset = Align8(orderOffset + orderCapacity);
        sourceRankLoadOffset = Align8(temporaryOffset + orderCapacity);
        rankLoadOffset = Align8(sourceRankLoadOffset + static_cast<uint64_t>(tiling.epSize) * rankStride);
        sourceActiveOffset = Align8(rankLoadOffset + rankStride);
        statusOffset = Align8(sourceActiveOffset + static_cast<uint64_t>(tiling.epSize) * 8U);
    }

    __aicore__ inline void Process()
    {
        if (tiling.blockCount == 0U || AscendC::GetBlockIdx() >= tiling.blockCount) {
            return;
        }
        InitializeOutputs();
        FlushOutputs();
        AscendC::SyncAll<true>();

        const bool commonValid = ValidateCommonInputs();
        BuildDigestSegment();
        WriteStageStatus(commonValid);
        FlushWorkspace();
        AscendC::SyncAll<true>();
        if (!AllStagesValid()) {
            return;
        }
        if (AscendC::GetBlockIdx() == 0U) {
            InitializeCommonDigest();
        }
        FlushWorkspace();
        AscendC::SyncAll<true>();

        for (uint32_t layout = 0U; layout < 2U; ++layout) {
            ClearWorkspace();
            FlushWorkspace();
            AscendC::SyncAll<true>();

            bool valid = true;
            if (AscendC::GetBlockIdx() == 0U) {
                valid = BuildCopyTable(layout);
                if (valid) {
                    InitializeDigest(layout);
                }
            }
            WriteStageStatus(valid);
            FlushWorkspace();
            AscendC::SyncAll<true>();
            if (!AllStagesValid()) {
                InvalidateLayout(layout);
                continue;
            }

            valid = BuildSampleBuckets(layout);
            WriteStageStatus(valid);
            FlushWorkspace();
            AscendC::SyncAll<true>();
            if (!AllStagesValid()) {
                InvalidateLayout(layout);
                continue;
            }

            valid = ProjectBuckets();
            WriteStageStatus(valid);
            FlushWorkspace();
            AscendC::SyncAll<true>();
            if (!AllStagesValid()) {
                InvalidateLayout(layout);
                continue;
            }

            uint32_t rowCount = 0U;
            uint64_t activeCount = 0U;
            if (AscendC::GetBlockIdx() == 0U) {
                valid = FinalizeProjectedBuckets(activeCount);
                if (valid) {
                    SortActiveBuckets(activeCount);
                    valid = BuildPolicies(layout, activeCount, rowCount);
                }
                if (valid) {
                    rowCountsGm.SetValue(layout, static_cast<int64_t>(rowCount));
                    digestGm.SetValue(static_cast<uint64_t>(layout) * 2U, digestOne);
                    digestGm.SetValue(static_cast<uint64_t>(layout) * 2U + 1U, digestTwo);
                }
            }
            WriteStageStatus(valid);
            FlushWorkspace();
            FlushOutputs();
            AscendC::SyncAll<true>();
            if (!AllStagesValid()) {
                InvalidateLayout(layout);
                continue;
            }
        }
    }

private:
    __aicore__ inline uint64_t Align8(uint64_t value) const
    {
        return (value + 7U) / 8U * 8U;
    }

    __aicore__ inline void FlushWorkspace()
    {
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(intWorkspaceGm);
    }

    __aicore__ inline void FlushOutputs()
    {
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(quotaWeightsGm);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(quotaConfiguredGm);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(compactRowsGm);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(rowCountsGm);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(digestGm);
    }

    __aicore__ inline void WriteStageStatus(bool valid)
    {
        const uint64_t row = statusOffset + static_cast<uint64_t>(AscendC::GetBlockIdx()) * 8U;
        intWorkspaceGm.SetValue(row, valid ? 1LL : 0LL);
    }

    __aicore__ inline bool AllStagesValid()
    {
        for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
            if (intWorkspaceGm.GetValue(statusOffset + static_cast<uint64_t>(block) * 8U) == 0) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void InvalidateLayout(uint32_t layout)
    {
        if (AscendC::GetBlockIdx() == 0U) {
            ClearLayoutOutputs(layout);
        }
        FlushOutputs();
        AscendC::SyncAll<true>();
    }

    __aicore__ inline uint64_t SourceExpertInputIndex(uint32_t source, uint32_t logical) const
    {
        return static_cast<uint64_t>(source) * tiling.numExperts + logical;
    }

    __aicore__ inline uint64_t SourceExpertWorkspaceIndex(uint32_t source, uint32_t logical) const
    {
        return static_cast<uint64_t>(source) * sourceExpertStride + logical;
    }

    __aicore__ inline uint64_t SourceOrderBase(uint32_t source) const
    {
        return orderOffset + static_cast<uint64_t>(source) * orderStride;
    }

    __aicore__ inline uint64_t SourceRankLoadBase(uint32_t source) const
    {
        return sourceRankLoadOffset + static_cast<uint64_t>(source) * rankStride;
    }

    __aicore__ inline void SetOwnedLines(
        AscendC::GlobalTensor<int64_t> &tensor,
        uint64_t start,
        uint64_t elements,
        int64_t value)
    {
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t stride = static_cast<uint64_t>(tiling.blockCount) * 8U;
        for (uint64_t line = block * 8U; line < elements; line += stride) {
            const uint64_t end = line + 8U < elements ? line + 8U : elements;
            for (uint64_t index = line; index < end; ++index) {
                tensor.SetValue(start + index, value);
            }
        }
    }

    __aicore__ inline uint64_t BucketIndex(uint32_t source, uint32_t logical, uint32_t mask) const
    {
        return static_cast<uint64_t>(source) * sourceBucketStride
            + static_cast<uint64_t>(logical) * tiling.maskCount + mask;
    }

    __aicore__ inline uint64_t ActiveMaskListBase(uint32_t source, uint32_t logical) const
    {
        return activeMaskListOffset + static_cast<uint64_t>(source) * sourceBucketStride
            + static_cast<uint64_t>(logical) * tiling.maskCount;
    }

    __aicore__ inline uint64_t ActiveMaskCountIndex(uint32_t source, uint32_t logical) const
    {
        return activeMaskCountOffset + SourceExpertWorkspaceIndex(source, logical);
    }

    __aicore__ inline uint64_t ObservedCountIndex(uint32_t source, uint32_t logical) const
    {
        return observedCountOffset + SourceExpertWorkspaceIndex(source, logical);
    }

    __aicore__ inline uint64_t CopyBase(uint32_t logical) const
    {
        return copySlotOffset + static_cast<uint64_t>(logical) * tiling.maxCopies;
    }

    __aicore__ inline uint64_t DenseBucketIndex(uint32_t layout, uint32_t logical, uint32_t mask) const
    {
        return (static_cast<uint64_t>(layout) * tiling.numExperts + logical) * tiling.maskCount + mask;
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

    __aicore__ inline int64_t PositiveMod(int64_t value, int64_t modulus) const
    {
        int64_t result = value % modulus;
        if (result < 0) {
            result += modulus;
        }
        return result;
    }

    __aicore__ inline void DigestValue(int64_t value)
    {
        constexpr int64_t modulusOne = 1048573LL;
        constexpr int64_t modulusTwo = 1000003LL;
        digestOne = (digestOne * 131LL + PositiveMod(value, modulusOne) + 1LL) % modulusOne;
        digestTwo = (digestTwo * 257LL + PositiveMod(value, modulusTwo) + 1LL) % modulusTwo;
    }

    __aicore__ inline void InitializeOutputs()
    {
        const uint64_t weightElements =
            static_cast<uint64_t>(2U) * tiling.numExperts * tiling.maskCount * tiling.maxCopies;
        SetOwnedLines(quotaWeightsGm, 0U, weightElements, 0);
        const uint64_t configuredElements =
            static_cast<uint64_t>(2U) * tiling.numExperts * tiling.maskCount;
        SetOwnedLines(quotaConfiguredGm, 0U, configuredElements, 0);
        const uint64_t rowElements =
            static_cast<uint64_t>(2U) * tiling.rowCapacity * tiling.rowWidth;
        SetOwnedLines(compactRowsGm, 0U, rowElements, 0);
        SetOwnedLines(rowCountsGm, 0U, 2U, 0);
        SetOwnedLines(digestGm, 0U, 4U, -1);
    }

    __aicore__ inline void ClearLayoutOutputs(uint32_t layout)
    {
        const uint64_t weightStride =
            static_cast<uint64_t>(tiling.numExperts) * tiling.maskCount * tiling.maxCopies;
        const uint64_t weightBase = static_cast<uint64_t>(layout) * weightStride;
        for (uint64_t index = 0U; index < weightStride; ++index) {
            quotaWeightsGm.SetValue(weightBase + index, 0);
        }
        const uint64_t configuredStride = static_cast<uint64_t>(tiling.numExperts) * tiling.maskCount;
        const uint64_t configuredBase = static_cast<uint64_t>(layout) * configuredStride;
        for (uint64_t index = 0U; index < configuredStride; ++index) {
            quotaConfiguredGm.SetValue(configuredBase + index, 0);
        }
        const uint64_t rowStride = static_cast<uint64_t>(tiling.rowCapacity) * tiling.rowWidth;
        const uint64_t rowBase = static_cast<uint64_t>(layout) * rowStride;
        for (uint64_t index = 0U; index < rowStride; ++index) {
            compactRowsGm.SetValue(rowBase + index, 0);
        }
        rowCountsGm.SetValue(layout, 0);
        digestGm.SetValue(static_cast<uint64_t>(layout) * 2U, -1);
        digestGm.SetValue(static_cast<uint64_t>(layout) * 2U + 1U, -1);
    }

    __aicore__ inline void ClearWorkspace()
    {
        const uint64_t copyElements = static_cast<uint64_t>(tiling.numExperts) * tiling.maxCopies;
        SetOwnedLines(intWorkspaceGm, copySlotOffset, copyElements, -1);
        SetOwnedLines(intWorkspaceGm, copyCountOffset, tiling.numExperts, 0);
        SetOwnedLines(
            intWorkspaceGm,
            sourceRankLoadOffset,
            static_cast<uint64_t>(tiling.epSize) * rankStride,
            0);
        SetOwnedLines(intWorkspaceGm, rankLoadOffset, rankStride, 0);
        SetOwnedLines(intWorkspaceGm, sourceActiveOffset, static_cast<uint64_t>(tiling.epSize) * 8U, 0);
    }

    __aicore__ inline bool ValidateCommonInputs()
    {
        if (
            tiling.topK == 0U || tiling.topK > 16U || tiling.epSize == 0U || tiling.epSize > 64U
            || tiling.numExperts == 0U || tiling.numExperts > 256U || tiling.maxCopies == 0U
            || tiling.maxCopies > 8U || tiling.maskCount != (1U << tiling.maxCopies)
            || tiling.sourceRank >= tiling.epSize || tiling.slotsPerRank == 0U || tiling.numLevels > 2U
            || static_cast<uint64_t>(tiling.numSlots)
                != static_cast<uint64_t>(tiling.epSize) * tiling.slotsPerRank
            || static_cast<uint64_t>(tiling.numSamples)
                > static_cast<uint64_t>(tiling.epSize) * tiling.samplesPerSource) {
            return false;
        }
        if (
            (tiling.numLevels > 0U
             && (tiling.levelSize0 == 0U || tiling.epSize % tiling.levelSize0 != 0U))
            || (tiling.numLevels > 1U
                && (tiling.levelSize1 == 0U || tiling.epSize % tiling.levelSize1 != 0U))) {
            return false;
        }
        const uint32_t block = AscendC::GetBlockIdx();
        for (uint32_t sample = block; sample < tiling.numSamples; sample += tiling.blockCount) {
            const int64_t source = sampleSourcesGm.GetValue(sample);
            if (
                source < 0 || source >= static_cast<int64_t>(tiling.epSize)
                || sampleOrdinalsGm.GetValue(sample) < 0) {
                return false;
            }
        }
        for (uint32_t source = block; source < tiling.epSize; source += tiling.blockCount) {
            for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
                intWorkspaceGm.SetValue(ObservedCountIndex(source, logical), 0);
            }
            uint32_t sampleCount = 0U;
            for (uint32_t sample = 0U; sample < tiling.numSamples; ++sample) {
                if (sampleSourcesGm.GetValue(sample) != static_cast<int64_t>(source)) {
                    continue;
                }
                ++sampleCount;
                const uint64_t routeBase = static_cast<uint64_t>(sample) * tiling.topK;
                for (uint32_t position = 0U; position < tiling.topK; ++position) {
                    const int64_t logical = sampleRoutesGm.GetValue(routeBase + position);
                    if (logical < 0 || logical >= static_cast<int64_t>(tiling.numExperts)) {
                        return false;
                    }
                    uint32_t occurrences = 0U;
                    bool first = true;
                    for (uint32_t other = 0U; other < tiling.topK; ++other) {
                        occurrences += static_cast<uint32_t>(sampleRoutesGm.GetValue(routeBase + other) == logical);
                        if (other < position && sampleRoutesGm.GetValue(routeBase + other) == logical) {
                            first = false;
                        }
                    }
                    const int64_t expected = first ? static_cast<int64_t>(occurrences) : 0LL;
                    if (sampleMultiplicityGm.GetValue(routeBase + position) != expected) {
                        return false;
                    }
                    if (expected > 0) {
                        const uint32_t logicalIndex = static_cast<uint32_t>(logical);
                        const uint64_t observedIndex = ObservedCountIndex(source, logicalIndex);
                        const int64_t observed = intWorkspaceGm.GetValue(observedIndex);
                        if (observed > 9223372036854775807LL - expected) {
                            return false;
                        }
                        intWorkspaceGm.SetValue(observedIndex, observed + expected);
                    }
                }
            }
            if (sampleCount > tiling.samplesPerSource) {
                return false;
            }
            for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
                const int64_t count = assignmentCountsGm.GetValue(SourceExpertInputIndex(source, logical));
                if (count < 0 || count > 2147483647LL) {
                    return false;
                }
            }
        }
        return true;
    }

    __aicore__ inline bool BuildCopyTable(uint32_t layout)
    {
        const uint64_t layoutBase = static_cast<uint64_t>(layout) * tiling.numSlots;
        for (uint32_t slot = 0U; slot < tiling.numSlots; ++slot) {
            const int64_t logical = layoutsGm.GetValue(layoutBase + slot);
            if (logical == -1) {
                continue;
            }
            if (logical < 0 || logical >= static_cast<int64_t>(tiling.numExperts)) {
                return false;
            }
            const uint64_t countIndex = copyCountOffset + static_cast<uint64_t>(logical);
            const int64_t count = intWorkspaceGm.GetValue(countIndex);
            if (count < 0 || count >= static_cast<int64_t>(tiling.maxCopies)) {
                return false;
            }
            intWorkspaceGm.SetValue(CopyBase(static_cast<uint32_t>(logical)) + count, slot);
            intWorkspaceGm.SetValue(countIndex, count + 1LL);
        }
        const uint64_t ownerBase = static_cast<uint64_t>(layout) * tiling.numExperts;
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const int64_t copies = intWorkspaceGm.GetValue(copyCountOffset + logical);
            const int64_t owner = ownerSlotsGm.GetValue(ownerBase + logical);
            if (
                copies <= 0 || copies > static_cast<int64_t>(tiling.maxCopies) || owner < 0
                || owner >= static_cast<int64_t>(tiling.numSlots)
                || layoutsGm.GetValue(layoutBase + static_cast<uint64_t>(owner))
                    != static_cast<int64_t>(logical)) {
                return false;
            }
        }
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            const uint32_t copies = static_cast<uint32_t>(
                intWorkspaceGm.GetValue(copyCountOffset + logical));
            uint32_t canonicalCount = 0U;
            int64_t singletonRank = -1LL;
            for (uint32_t copy = 0U; copy < copies; ++copy) {
                if (!IsCanonicalCopy(logical, copy, copies)) {
                    continue;
                }
                ++canonicalCount;
                singletonRank = intWorkspaceGm.GetValue(CopyBase(logical) + copy)
                    / static_cast<int64_t>(tiling.slotsPerRank);
            }
            if (canonicalCount == 0U || canonicalCount > tiling.maxCopies) {
                return false;
            }
            intWorkspaceGm.SetValue(canonicalCountOffset + logical, canonicalCount);
            intWorkspaceGm.SetValue(
                singletonRankOffset + logical,
                canonicalCount == 1U ? singletonRank : -1LL);
        }
        return true;
    }

    __aicore__ inline bool IsCanonicalCopy(uint32_t logical, uint32_t copy, uint32_t copies)
    {
        const uint64_t copyBase = CopyBase(logical);
        const int64_t rank =
            intWorkspaceGm.GetValue(copyBase + copy) / static_cast<int64_t>(tiling.slotsPerRank);
        for (uint32_t later = copy + 1U; later < copies; ++later) {
            const int64_t laterRank =
                intWorkspaceGm.GetValue(copyBase + later) / static_cast<int64_t>(tiling.slotsPerRank);
            if (laterRank == rank) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool RankAlreadyVisited(
        uint32_t layout,
        uint32_t sample,
        uint32_t logical,
        int64_t destinationRank,
        uint32_t groupSize)
    {
        const int64_t source = sampleSourcesGm.GetValue(sample);
        if (groupSize == 1U) {
            if (destinationRank == source) {
                return true;
            }
        } else if (
            destinationRank / static_cast<int64_t>(groupSize)
            == source / static_cast<int64_t>(groupSize)) {
            return true;
        }
        const uint64_t routeBase = static_cast<uint64_t>(sample) * tiling.topK;
        const uint64_t ownerBase = static_cast<uint64_t>(layout) * tiling.numExperts;
        for (uint32_t position = 0U; position < tiling.topK; ++position) {
            const int64_t other = sampleRoutesGm.GetValue(routeBase + position);
            if (other == static_cast<int64_t>(logical)) {
                continue;
            }
            const int64_t otherRank =
                ownerSlotsGm.GetValue(ownerBase + static_cast<uint64_t>(other))
                / static_cast<int64_t>(tiling.slotsPerRank);
            if (groupSize == 1U) {
                if (otherRank == destinationRank) {
                    return true;
                }
            } else if (
                otherRank / static_cast<int64_t>(groupSize)
                == destinationRank / static_cast<int64_t>(groupSize)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint32_t CommunicationScore(
        uint32_t layout,
        uint32_t sample,
        uint32_t logical,
        int64_t destinationRank)
    {
        uint32_t score = 0U;
        if (tiling.numLevels > 1U) {
            score = score * 2U + static_cast<uint32_t>(
                !RankAlreadyVisited(layout, sample, logical, destinationRank, tiling.levelSize1));
        }
        if (tiling.numLevels > 0U) {
            score = score * 2U + static_cast<uint32_t>(
                !RankAlreadyVisited(layout, sample, logical, destinationRank, tiling.levelSize0));
        }
        score = score * 2U
            + static_cast<uint32_t>(!RankAlreadyVisited(layout, sample, logical, destinationRank, 1U));
        return score;
    }

    __aicore__ inline uint32_t TieMask(uint32_t layout, uint32_t sample, uint32_t logical)
    {
        const uint32_t copies =
            static_cast<uint32_t>(intWorkspaceGm.GetValue(copyCountOffset + logical));
        const uint64_t copyBase = CopyBase(logical);
        const int64_t source = sampleSourcesGm.GetValue(sample);
        uint64_t visitedRanks = 1ULL << static_cast<uint32_t>(source);
        uint64_t visitedLevelZero = 0U;
        uint64_t visitedLevelOne = 0U;
        if (tiling.numLevels > 0U) {
            visitedLevelZero = 1ULL << (static_cast<uint32_t>(source) / tiling.levelSize0);
        }
        if (tiling.numLevels > 1U) {
            visitedLevelOne = 1ULL << (static_cast<uint32_t>(source) / tiling.levelSize1);
        }
        const uint64_t routeBase = static_cast<uint64_t>(sample) * tiling.topK;
        const uint64_t ownerBase = static_cast<uint64_t>(layout) * tiling.numExperts;
        for (uint32_t position = 0U; position < tiling.topK; ++position) {
            const int64_t other = sampleRoutesGm.GetValue(routeBase + position);
            if (other == static_cast<int64_t>(logical)) {
                continue;
            }
            const uint32_t rank = static_cast<uint32_t>(
                ownerSlotsGm.GetValue(ownerBase + static_cast<uint64_t>(other))
                / static_cast<int64_t>(tiling.slotsPerRank));
            visitedRanks |= 1ULL << rank;
            if (tiling.numLevels > 0U) {
                visitedLevelZero |= 1ULL << (rank / tiling.levelSize0);
            }
            if (tiling.numLevels > 1U) {
                visitedLevelOne |= 1ULL << (rank / tiling.levelSize1);
            }
        }
        uint32_t bestScore = 0xffffffffU;
        uint32_t mask = 0U;
        for (uint32_t copy = 0U; copy < copies; ++copy) {
            if (!IsCanonicalCopy(logical, copy, copies)) {
                continue;
            }
            const int64_t rank =
                intWorkspaceGm.GetValue(copyBase + copy) / static_cast<int64_t>(tiling.slotsPerRank);
            uint32_t score = 0U;
            if (tiling.numLevels > 1U) {
                score = score * 2U + static_cast<uint32_t>(
                    (visitedLevelOne & (1ULL << (static_cast<uint32_t>(rank) / tiling.levelSize1))) == 0U);
            }
            if (tiling.numLevels > 0U) {
                score = score * 2U + static_cast<uint32_t>(
                    (visitedLevelZero & (1ULL << (static_cast<uint32_t>(rank) / tiling.levelSize0))) == 0U);
            }
            score = score * 2U
                + static_cast<uint32_t>((visitedRanks & (1ULL << static_cast<uint32_t>(rank))) == 0U);
            if (score < bestScore) {
                bestScore = score;
                mask = 1U << copy;
            } else if (score == bestScore) {
                mask |= 1U << copy;
            }
        }
        return mask;
    }

    __aicore__ inline bool BuildSampleBuckets(uint32_t layout)
    {
        const uint32_t block = AscendC::GetBlockIdx();
        for (uint32_t source = block; source < tiling.epSize; source += tiling.blockCount) {
            bool hasMultipleCopies = false;
            for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
                const int64_t canonicalCount = intWorkspaceGm.GetValue(canonicalCountOffset + logical);
                if (canonicalCount <= 1) {
                    continue;
                }
                hasMultipleCopies = true;
                intWorkspaceGm.SetValue(ActiveMaskCountIndex(source, logical), 0);
                const uint64_t bucketBase = sampleBucketOffset + BucketIndex(source, logical, 0U);
                for (uint32_t mask = 0U; mask < tiling.maskCount; ++mask) {
                    intWorkspaceGm.SetValue(bucketBase + mask, 0);
                }
            }
            if (!hasMultipleCopies) {
                continue;
            }
            for (uint32_t sample = 0U; sample < tiling.numSamples; ++sample) {
                if (sampleSourcesGm.GetValue(sample) != static_cast<int64_t>(source)) {
                    continue;
                }
                const uint64_t routeBase = static_cast<uint64_t>(sample) * tiling.topK;
                for (uint32_t position = 0U; position < tiling.topK; ++position) {
                    const int64_t multiplicity = sampleMultiplicityGm.GetValue(routeBase + position);
                    if (multiplicity == 0) {
                        continue;
                    }
                    const uint32_t logical = static_cast<uint32_t>(sampleRoutesGm.GetValue(routeBase + position));
                    if (intWorkspaceGm.GetValue(canonicalCountOffset + logical) == 1) {
                        continue;
                    }
                    const uint32_t mask = TieMask(layout, sample, logical);
                    if (mask == 0U || mask >= tiling.maskCount) {
                        return false;
                    }
                    const uint64_t bucket = BucketIndex(source, logical, mask);
                    const int64_t previous = intWorkspaceGm.GetValue(sampleBucketOffset + bucket);
                    if (previous < 0 || previous > 9223372036854775807LL - multiplicity) {
                        return false;
                    }
                    if (previous == 0) {
                        const uint64_t countIndex = ActiveMaskCountIndex(source, logical);
                        const int64_t rawCount = intWorkspaceGm.GetValue(countIndex);
                        if (rawCount < 0 || rawCount >= static_cast<int64_t>(tiling.maskCount - 1U)) {
                            return false;
                        }
                        const uint64_t listBase = ActiveMaskListBase(source, logical);
                        uint32_t insert = static_cast<uint32_t>(rawCount);
                        while (
                            insert > 0U
                            && intWorkspaceGm.GetValue(listBase + insert - 1U) > static_cast<int64_t>(mask)) {
                            intWorkspaceGm.SetValue(
                                listBase + insert,
                                intWorkspaceGm.GetValue(listBase + insert - 1U));
                            --insert;
                        }
                        intWorkspaceGm.SetValue(listBase + insert, mask);
                        intWorkspaceGm.SetValue(countIndex, rawCount + 1LL);
                    }
                    intWorkspaceGm.SetValue(sampleBucketOffset + bucket, previous + multiplicity);
                }
            }
        }
        return true;
    }

    __aicore__ inline uint32_t CollectDestinations(
        uint32_t logical,
        uint32_t mask,
        int64_t ranks[8],
        uint32_t copies[8])
    {
        const uint32_t copyCount =
            static_cast<uint32_t>(intWorkspaceGm.GetValue(copyCountOffset + logical));
        const uint64_t copyBase = CopyBase(logical);
        uint32_t count = 0U;
        for (uint32_t copy = 0U; copy < copyCount; ++copy) {
            if ((mask & (1U << copy)) == 0U) {
                continue;
            }
            const int64_t rank =
                intWorkspaceGm.GetValue(copyBase + copy) / static_cast<int64_t>(tiling.slotsPerRank);
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

    __aicore__ inline int32_t CompareDestinationTuples(
        uint32_t logical,
        uint32_t maskA,
        uint32_t maskB)
    {
        int64_t ranksA[8];
        int64_t ranksB[8];
        uint32_t copiesA[8];
        uint32_t copiesB[8];
        const uint32_t countA = CollectDestinations(logical, maskA, ranksA, copiesA);
        const uint32_t countB = CollectDestinations(logical, maskB, ranksB, copiesB);
        const uint32_t common = countA < countB ? countA : countB;
        for (uint32_t index = 0U; index < common; ++index) {
            if (ranksA[index] < ranksB[index]) {
                return -1;
            }
            if (ranksA[index] > ranksB[index]) {
                return 1;
            }
        }
        if (countA < countB) {
            return -1;
        }
        if (countA > countB) {
            return 1;
        }
        return 0;
    }

    __aicore__ inline bool ProjectBuckets()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        for (uint32_t source = block; source < tiling.epSize; source += tiling.blockCount) {
            uint64_t activeCount = 0U;
            const uint64_t localRankLoadBase = SourceRankLoadBase(source);
            const uint64_t localOrderBase = SourceOrderBase(source);
            for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
                const uint64_t sourceExpert = SourceExpertInputIndex(source, logical);
                const int64_t sampleTotal = intWorkspaceGm.GetValue(ObservedCountIndex(source, logical));
                if (sampleTotal < 0) {
                    return false;
                }
                if (sampleTotal == 0) {
                    continue;
                }
                const int64_t exact = assignmentCountsGm.GetValue(sourceExpert);
                const int64_t canonicalCount = intWorkspaceGm.GetValue(canonicalCountOffset + logical);
                if (canonicalCount == 1) {
                    const int64_t rank = intWorkspaceGm.GetValue(singletonRankOffset + logical);
                    if (rank < 0 || rank >= static_cast<int64_t>(tiling.epSize)) {
                        return false;
                    }
                    const uint64_t loadIndex = localRankLoadBase + static_cast<uint64_t>(rank);
                    const int64_t load = intWorkspaceGm.GetValue(loadIndex);
                    if (load > 9223372036854775807LL - exact) {
                        return false;
                    }
                    intWorkspaceGm.SetValue(loadIndex, load + exact);
                    continue;
                }
                const int64_t rawActiveMasks = intWorkspaceGm.GetValue(ActiveMaskCountIndex(source, logical));
                if (
                    canonicalCount <= 1 || rawActiveMasks <= 0
                    || rawActiveMasks >= static_cast<int64_t>(tiling.maskCount)) {
                    return false;
                }
                const uint32_t activeMasks = static_cast<uint32_t>(rawActiveMasks);
                const uint64_t activeMaskBase = ActiveMaskListBase(source, logical);
                int64_t roundedTotal = 0;
                for (uint32_t active = 0U; active < activeMasks; ++active) {
                    const uint32_t mask = static_cast<uint32_t>(
                        intWorkspaceGm.GetValue(activeMaskBase + active));
                    const uint64_t bucket = BucketIndex(source, logical, mask);
                    const int64_t count = intWorkspaceGm.GetValue(sampleBucketOffset + bucket);
                    const int64_t product = exact * count;
                    const int64_t projected = product / sampleTotal;
                    intWorkspaceGm.SetValue(projectedBucketOffset + bucket, projected);
                    roundedTotal += projected;
                }
                int64_t remainder = exact - roundedTotal;
                while (remainder > 0) {
                    int32_t bestMask = -1;
                    int64_t bestFraction = -1;
                    for (uint32_t active = 0U; active < activeMasks; ++active) {
                        const uint32_t mask = static_cast<uint32_t>(
                            intWorkspaceGm.GetValue(activeMaskBase + active));
                        const uint64_t bucket = BucketIndex(source, logical, mask);
                        const int64_t count = intWorkspaceGm.GetValue(sampleBucketOffset + bucket);
                        const int64_t projected = intWorkspaceGm.GetValue(projectedBucketOffset + bucket);
                        if (projected < 0) {
                            continue;
                        }
                        const int64_t fraction = (exact * count) % sampleTotal;
                        const bool betterTuple = bestMask >= 0
                            && CompareDestinationTuples(logical, mask, static_cast<uint32_t>(bestMask)) < 0;
                        if (
                            bestMask < 0 || fraction > bestFraction
                            || (fraction == bestFraction && betterTuple)
                            || (fraction == bestFraction && bestMask >= 0
                                && CompareDestinationTuples(
                                       logical, mask, static_cast<uint32_t>(bestMask))
                                    == 0
                                && mask < static_cast<uint32_t>(bestMask))) {
                            bestMask = static_cast<int32_t>(mask);
                            bestFraction = fraction;
                        }
                    }
                    if (bestMask < 0) {
                        return false;
                    }
                    const uint64_t bucket =
                        BucketIndex(source, logical, static_cast<uint32_t>(bestMask));
                    const int64_t projected = intWorkspaceGm.GetValue(projectedBucketOffset + bucket);
                    intWorkspaceGm.SetValue(projectedBucketOffset + bucket, -(projected + 1LL));
                    --remainder;
                }
                for (uint32_t active = 0U; active < activeMasks; ++active) {
                    const uint32_t mask = static_cast<uint32_t>(
                        intWorkspaceGm.GetValue(activeMaskBase + active));
                    const uint64_t bucket = BucketIndex(source, logical, mask);
                    int64_t projected = intWorkspaceGm.GetValue(projectedBucketOffset + bucket);
                    if (projected < 0) {
                        projected = -projected;
                        intWorkspaceGm.SetValue(projectedBucketOffset + bucket, projected);
                    }
                    if (projected <= 0) {
                        continue;
                    }
                    const uint32_t destinationCount = PopCount(mask);
                    if (destinationCount == 1U) {
                        int64_t ranks[8];
                        uint32_t copies[8];
                        if (CollectDestinations(logical, mask, ranks, copies) != 1U) {
                            return false;
                        }
                        const uint64_t loadIndex = localRankLoadBase + static_cast<uint64_t>(ranks[0]);
                        const int64_t load = intWorkspaceGm.GetValue(loadIndex);
                        if (load > 9223372036854775807LL - projected) {
                            return false;
                        }
                        intWorkspaceGm.SetValue(loadIndex, load + projected);
                    } else if (destinationCount > 1U) {
                        if (activeCount >= tiling.rowCapacity) {
                            return false;
                        }
                        intWorkspaceGm.SetValue(localOrderBase + activeCount, static_cast<int64_t>(bucket));
                        ++activeCount;
                    } else {
                        return false;
                    }
                }
            }
            intWorkspaceGm.SetValue(sourceActiveOffset + static_cast<uint64_t>(source) * 8U, activeCount);
        }
        return true;
    }

    __aicore__ inline bool FinalizeProjectedBuckets(uint64_t &activeCount)
    {
        for (uint32_t rank = 0U; rank < tiling.epSize; ++rank) {
            int64_t total = 0;
            for (uint32_t source = 0U; source < tiling.epSize; ++source) {
                const int64_t value = intWorkspaceGm.GetValue(SourceRankLoadBase(source) + rank);
                if (value < 0 || total > 9223372036854775807LL - value) {
                    return false;
                }
                total += value;
            }
            intWorkspaceGm.SetValue(rankLoadOffset + rank, total);
        }
        activeCount = 0U;
        for (uint32_t source = 0U; source < tiling.epSize; ++source) {
            const int64_t rawCount = intWorkspaceGm.GetValue(
                sourceActiveOffset + static_cast<uint64_t>(source) * 8U);
            if (rawCount < 0 || rawCount > static_cast<int64_t>(tiling.rowCapacity)) {
                return false;
            }
            const uint64_t count = static_cast<uint64_t>(rawCount);
            if (activeCount > orderCapacity - count) {
                return false;
            }
            const uint64_t sourceBase = SourceOrderBase(source);
            for (uint64_t index = 0U; index < count; ++index) {
                intWorkspaceGm.SetValue(
                    orderOffset + activeCount + index,
                    intWorkspaceGm.GetValue(sourceBase + index));
            }
            activeCount += count;
        }
        return true;
    }

    __aicore__ inline void DecodeBucket(
        uint64_t bucket,
        uint32_t &source,
        uint32_t &logical,
        uint32_t &mask) const
    {
        source = static_cast<uint32_t>(bucket / sourceBucketStride);
        const uint64_t localBucket = bucket % sourceBucketStride;
        mask = static_cast<uint32_t>(localBucket % tiling.maskCount);
        logical = static_cast<uint32_t>(localBucket / tiling.maskCount);
    }

    __aicore__ inline bool BucketLess(uint64_t bucketA, uint64_t bucketB)
    {
        uint32_t sourceA = 0U;
        uint32_t logicalA = 0U;
        uint32_t maskA = 0U;
        uint32_t sourceB = 0U;
        uint32_t logicalB = 0U;
        uint32_t maskB = 0U;
        DecodeBucket(bucketA, sourceA, logicalA, maskA);
        DecodeBucket(bucketB, sourceB, logicalB, maskB);
        const int64_t totalA = intWorkspaceGm.GetValue(projectedBucketOffset + bucketA);
        const int64_t totalB = intWorkspaceGm.GetValue(projectedBucketOffset + bucketB);
        if (totalA != totalB) {
            return totalA > totalB;
        }
        if (sourceA != sourceB) {
            return sourceA < sourceB;
        }
        if (logicalA != logicalB) {
            return logicalA < logicalB;
        }
        const int32_t tupleComparison = CompareDestinationTuples(logicalA, maskA, maskB);
        if (tupleComparison != 0) {
            return tupleComparison < 0;
        }
        return maskA < maskB;
    }

    __aicore__ inline void SortActiveBuckets(uint64_t activeCount)
    {
        if (activeCount < 2U) {
            return;
        }
        for (uint64_t width = 1U; width < activeCount; width *= 2U) {
            for (uint64_t left = 0U; left < activeCount; left += 2U * width) {
                const uint64_t middle = left + width < activeCount ? left + width : activeCount;
                const uint64_t right = left + 2U * width < activeCount ? left + 2U * width : activeCount;
                uint64_t first = left;
                uint64_t second = middle;
                uint64_t output = left;
                while (first < middle && second < right) {
                    const uint64_t bucketFirst =
                        static_cast<uint64_t>(intWorkspaceGm.GetValue(orderOffset + first));
                    const uint64_t bucketSecond =
                        static_cast<uint64_t>(intWorkspaceGm.GetValue(orderOffset + second));
                    if (!BucketLess(bucketSecond, bucketFirst)) {
                        intWorkspaceGm.SetValue(temporaryOffset + output, static_cast<int64_t>(bucketFirst));
                        ++first;
                    } else {
                        intWorkspaceGm.SetValue(temporaryOffset + output, static_cast<int64_t>(bucketSecond));
                        ++second;
                    }
                    ++output;
                }
                while (first < middle) {
                    intWorkspaceGm.SetValue(
                        temporaryOffset + output, intWorkspaceGm.GetValue(orderOffset + first));
                    ++first;
                    ++output;
                }
                while (second < right) {
                    intWorkspaceGm.SetValue(
                        temporaryOffset + output, intWorkspaceGm.GetValue(orderOffset + second));
                    ++second;
                    ++output;
                }
            }
            for (uint64_t index = 0U; index < activeCount; ++index) {
                intWorkspaceGm.SetValue(
                    orderOffset + index, intWorkspaceGm.GetValue(temporaryOffset + index));
            }
        }
    }

    __aicore__ inline void WaterfillQuotas(
        const int64_t ranks[8],
        uint32_t count,
        int64_t total,
        int64_t quotas[8])
    {
        uint32_t order[8];
        for (uint32_t index = 0U; index < count; ++index) {
            quotas[index] = 0;
            order[index] = index;
            uint32_t insert = index;
            while (insert > 0U) {
                const uint32_t current = order[insert];
                const uint32_t prior = order[insert - 1U];
                const int64_t currentLoad =
                    intWorkspaceGm.GetValue(rankLoadOffset + static_cast<uint64_t>(ranks[current]));
                const int64_t priorLoad =
                    intWorkspaceGm.GetValue(rankLoadOffset + static_cast<uint64_t>(ranks[prior]));
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
                rankLoadOffset + static_cast<uint64_t>(ranks[order[active - 1U]]));
            const int64_t nextLoad = intWorkspaceGm.GetValue(
                rankLoadOffset + static_cast<uint64_t>(ranks[order[active]]));
            const int64_t difference = nextLoad > currentLoad ? nextLoad - currentLoad : 0;
            const int64_t required = difference * static_cast<int64_t>(active);
            if (required > remaining) {
                break;
            }
            if (required > 0) {
                const int64_t increment = required / static_cast<int64_t>(active);
                const int64_t extra = required % static_cast<int64_t>(active);
                for (uint32_t index = 0U; index < active; ++index) {
                    quotas[order[index]] +=
                        increment + static_cast<int64_t>(index < static_cast<uint32_t>(extra));
                }
                remaining -= required;
            }
            ++active;
        }
        const int64_t increment = remaining / static_cast<int64_t>(active);
        const int64_t extra = remaining % static_cast<int64_t>(active);
        for (uint32_t index = 0U; index < active; ++index) {
            quotas[order[index]] +=
                increment + static_cast<int64_t>(index < static_cast<uint32_t>(extra));
        }
    }

    __aicore__ inline void SegmentDigestValue(int64_t value, int64_t &one, int64_t &two)
    {
        constexpr int64_t modulusOne = 1048573LL;
        constexpr int64_t modulusTwo = 1000003LL;
        one = (one * 131LL + PositiveMod(value, modulusOne) + 1LL) % modulusOne;
        two = (two * 257LL + PositiveMod(value, modulusTwo) + 1LL) % modulusTwo;
    }

    __aicore__ inline int64_t ModularPower(int64_t base, uint64_t exponent, int64_t modulus)
    {
        int64_t result = 1LL;
        int64_t factor = base % modulus;
        while (exponent != 0U) {
            if ((exponent & 1U) != 0U) {
                result = (result * factor) % modulus;
            }
            factor = (factor * factor) % modulus;
            exponent >>= 1U;
        }
        return result;
    }

    __aicore__ inline void BuildDigestSegment()
    {
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t begin = static_cast<uint64_t>(tiling.numSamples) * block / tiling.blockCount;
        const uint64_t end = static_cast<uint64_t>(tiling.numSamples) * (block + 1U) / tiling.blockCount;
        int64_t one = 0LL;
        int64_t two = 0LL;
        for (uint64_t sample = begin; sample < end; ++sample) {
            SegmentDigestValue(sampleSourcesGm.GetValue(sample), one, two);
            SegmentDigestValue(sampleOrdinalsGm.GetValue(sample), one, two);
            const uint64_t routeBase = sample * tiling.topK;
            for (uint32_t position = 0U; position < tiling.topK; ++position) {
                SegmentDigestValue(sampleRoutesGm.GetValue(routeBase + position), one, two);
                SegmentDigestValue(sampleMultiplicityGm.GetValue(routeBase + position), one, two);
            }
        }
        const uint64_t statusBase = statusOffset + block * 8U;
        intWorkspaceGm.SetValue(statusBase + 1U, static_cast<int64_t>((end - begin) * (2U + 2U * tiling.topK)));
        intWorkspaceGm.SetValue(statusBase + 2U, one);
        intWorkspaceGm.SetValue(statusBase + 3U, two);
    }

    __aicore__ inline void InitializeCommonDigest()
    {
        digestOne = 17LL;
        digestTwo = 29LL;
        DigestValue(607543LL);
        DigestValue(tiling.numSamples);
        DigestValue(tiling.topK);
        DigestValue(tiling.epSize);
        DigestValue(tiling.numExperts);
        DigestValue(tiling.numSlots);
        DigestValue(tiling.maxCopies);
        DigestValue(tiling.samplesPerSource);
        DigestValue(tiling.numLevels);
        DigestValue(tiling.levelSize0);
        DigestValue(tiling.levelSize1);
        for (uint32_t block = 0U; block < tiling.blockCount; ++block) {
            const uint64_t statusBase = statusOffset + static_cast<uint64_t>(block) * 8U;
            const uint64_t length = static_cast<uint64_t>(intWorkspaceGm.GetValue(statusBase + 1U));
            digestOne = (
                digestOne * ModularPower(131LL, length, 1048573LL)
                + intWorkspaceGm.GetValue(statusBase + 2U)) % 1048573LL;
            digestTwo = (
                digestTwo * ModularPower(257LL, length, 1000003LL)
                + intWorkspaceGm.GetValue(statusBase + 3U)) % 1000003LL;
        }
        for (uint64_t index = 0U; index < sourceExpertCount; ++index) {
            DigestValue(assignmentCountsGm.GetValue(index));
        }
        intWorkspaceGm.SetValue(statusOffset + 4U, digestOne);
        intWorkspaceGm.SetValue(statusOffset + 5U, digestTwo);
    }

    __aicore__ inline void InitializeDigest(uint32_t layout)
    {
        digestOne = intWorkspaceGm.GetValue(statusOffset + 4U);
        digestTwo = intWorkspaceGm.GetValue(statusOffset + 5U);
        const uint64_t layoutBase = static_cast<uint64_t>(layout) * tiling.numSlots;
        for (uint32_t slot = 0U; slot < tiling.numSlots; ++slot) {
            DigestValue(layoutsGm.GetValue(layoutBase + slot));
        }
        const uint64_t ownerBase = static_cast<uint64_t>(layout) * tiling.numExperts;
        for (uint32_t logical = 0U; logical < tiling.numExperts; ++logical) {
            DigestValue(ownerSlotsGm.GetValue(ownerBase + logical));
        }
    }

    __aicore__ inline void DigestPolicy(
        uint32_t source,
        uint32_t logical,
        const int64_t ranks[8],
        const int64_t quotas[8],
        uint32_t count)
    {
        DigestValue(21073LL);
        DigestValue(source);
        DigestValue(logical);
        DigestValue(count);
        for (uint32_t index = 0U; index < count; ++index) {
            DigestValue(ranks[index]);
        }
        for (uint32_t index = 0U; index < count; ++index) {
            DigestValue(quotas[index]);
        }
    }

    __aicore__ inline bool BuildPolicies(
        uint32_t layout,
        uint64_t activeCount,
        uint32_t &rowCount)
    {
        rowCount = 0U;
        for (uint64_t position = 0U; position < activeCount; ++position) {
            const uint64_t bucket =
                static_cast<uint64_t>(intWorkspaceGm.GetValue(orderOffset + position));
            uint32_t source = 0U;
            uint32_t logical = 0U;
            uint32_t mask = 0U;
            DecodeBucket(bucket, source, logical, mask);
            const int64_t total = intWorkspaceGm.GetValue(projectedBucketOffset + bucket);
            if (total <= 0) {
                return false;
            }
            int64_t ranks[8];
            uint32_t copies[8];
            const uint32_t count = CollectDestinations(logical, mask, ranks, copies);
            if (count <= 1U || count > tiling.maxCopies) {
                return false;
            }
            int64_t quotas[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            WaterfillQuotas(ranks, count, total, quotas);
            int64_t quotaTotal = 0;
            for (uint32_t index = 0U; index < count; ++index) {
                if (quotas[index] < 0 || quotas[index] > 2147483647LL) {
                    return false;
                }
                quotaTotal += quotas[index];
                const uint64_t loadIndex = rankLoadOffset + static_cast<uint64_t>(ranks[index]);
                const int64_t load = intWorkspaceGm.GetValue(loadIndex);
                if (load > 9223372036854775807LL - quotas[index]) {
                    return false;
                }
                intWorkspaceGm.SetValue(loadIndex, load + quotas[index]);
            }
            if (quotaTotal != total) {
                return false;
            }
            DigestPolicy(source, logical, ranks, quotas, count);
            if (source != tiling.sourceRank) {
                continue;
            }
            if (rowCount >= tiling.rowCapacity) {
                return false;
            }
            const uint64_t denseBucket = DenseBucketIndex(layout, logical, mask);
            quotaConfiguredGm.SetValue(denseBucket, 1);
            const uint64_t weightBase = denseBucket * tiling.maxCopies;
            for (uint32_t index = 0U; index < count; ++index) {
                quotaWeightsGm.SetValue(weightBase + copies[index], quotas[index]);
            }
            const uint64_t rowBase =
                (static_cast<uint64_t>(layout) * tiling.rowCapacity + rowCount) * tiling.rowWidth;
            compactRowsGm.SetValue(rowBase, source);
            compactRowsGm.SetValue(rowBase + 1U, logical);
            compactRowsGm.SetValue(rowBase + 2U, count);
            for (uint32_t index = 0U; index < count; ++index) {
                compactRowsGm.SetValue(rowBase + 3U + index, ranks[index]);
                compactRowsGm.SetValue(rowBase + 3U + tiling.maxCopies + index, quotas[index]);
            }
            ++rowCount;
        }
        return true;
    }

    AscendC::GlobalTensor<int64_t> sampleRoutesGm;
    AscendC::GlobalTensor<int64_t> sampleMultiplicityGm;
    AscendC::GlobalTensor<int64_t> sampleSourcesGm;
    AscendC::GlobalTensor<int64_t> sampleOrdinalsGm;
    AscendC::GlobalTensor<int64_t> assignmentCountsGm;
    AscendC::GlobalTensor<int64_t> layoutsGm;
    AscendC::GlobalTensor<int64_t> ownerSlotsGm;
    AscendC::GlobalTensor<int64_t> quotaWeightsGm;
    AscendC::GlobalTensor<int64_t> quotaConfiguredGm;
    AscendC::GlobalTensor<int64_t> compactRowsGm;
    AscendC::GlobalTensor<int64_t> rowCountsGm;
    AscendC::GlobalTensor<int64_t> digestGm;
    AscendC::GlobalTensor<int64_t> intWorkspaceGm;
    HiermoeQuotaPolicyTilingData tiling;
    uint64_t sourceExpertCount;
    uint64_t sourceExpertStride;
    uint64_t sourceBucketStride;
    uint64_t bucketCount;
    uint64_t orderStride;
    uint64_t orderCapacity;
    uint64_t rankStride;
    uint64_t copySlotOffset;
    uint64_t copyCountOffset;
    uint64_t canonicalCountOffset;
    uint64_t singletonRankOffset;
    uint64_t sampleBucketOffset;
    uint64_t projectedBucketOffset;
    uint64_t activeMaskListOffset;
    uint64_t activeMaskCountOffset;
    uint64_t observedCountOffset;
    uint64_t orderOffset;
    uint64_t temporaryOffset;
    uint64_t sourceRankLoadOffset;
    uint64_t rankLoadOffset;
    uint64_t sourceActiveOffset;
    uint64_t statusOffset;
    int64_t digestOne;
    int64_t digestTwo;
};

extern "C" __global__ __aicore__ void hiermoe_quota_policy(
    GM_ADDR sampleRoutes,
    GM_ADDR sampleMultiplicity,
    GM_ADDR sampleSources,
    GM_ADDR sampleOrdinals,
    GM_ADDR assignmentCounts,
    GM_ADDR layouts,
    GM_ADDR ownerSlots,
    GM_ADDR quotaWeights,
    GM_ADDR quotaConfigured,
    GM_ADDR compactRows,
    GM_ADDR rowCounts,
    GM_ADDR digest,
    GM_ADDR intWorkspace,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeQuotaPolicyTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeQuotaPolicy op;
    op.Init(
        sampleRoutes,
        sampleMultiplicity,
        sampleSources,
        sampleOrdinals,
        assignmentCounts,
        layouts,
        ownerSlots,
        quotaWeights,
        quotaConfigured,
        compactRows,
        rowCounts,
        digest,
        intWorkspace,
        tilingData);
    op.Process();
}
