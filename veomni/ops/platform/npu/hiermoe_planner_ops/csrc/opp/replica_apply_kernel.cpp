#include "kernel_operator.h"
#include "replica_apply_tiling.h"

class KernelHiermoeReplicaApply {
public:
    __aicore__ inline void Init(
        GM_ADDR routeIndices,
        GM_ADDR tokenCounts,
        GM_ADDR flatLogical,
        GM_ADDR routeRanks,
        GM_ADDR routeScores,
        GM_ADDR minimumScores,
        GM_ADDR tieCount,
        GM_ADDR tiedRankOrder,
        GM_ADDR routeHashes,
        GM_ADDR tokenGroupCounts,
        GM_ADDR logicalExpert,
        GM_ADDR destinationRank,
        const HiermoeReplicaApplyTilingData &tiling)
    {
        this->tiling = tiling;
        routeIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(routeIndices));
        tokenCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenCounts));
        flatLogicalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(flatLogical));
        routeRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(routeRanks));
        routeScoresGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(routeScores));
        minimumScoresGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(minimumScores));
        tieCountGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tieCount));
        tiedRankOrderGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tiedRankOrder));
        routeHashesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(routeHashes));
        tokenGroupCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenGroupCounts));
        logicalExpertGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(logicalExpert));
        destinationRankGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(destinationRank));
    }

    __aicore__ inline void Process()
    {
        const int32_t logicalExpert = static_cast<int32_t>(logicalExpertGm.GetValue(0));
        const int32_t destinationRank = static_cast<int32_t>(destinationRankGm.GetValue(0));
        if (logicalExpert < 0 || logicalExpert >= static_cast<int32_t>(tiling.numExperts) || destinationRank < 0 ||
            destinationRank >= static_cast<int32_t>(tiling.epSize)) {
            return;
        }
        const int32_t rawCount = tokenCountsGm.GetValue(logicalExpert);
        const int32_t boundedCount = rawCount < 0 ? 0 :
            (rawCount > static_cast<int32_t>(tiling.tokenWidth) ? static_cast<int32_t>(tiling.tokenWidth) : rawCount);
        const uint32_t count = static_cast<uint32_t>(boundedCount);
        const uint64_t routeRow = static_cast<uint64_t>(logicalExpert) * tiling.tokenWidth;
        const uint32_t blockCount = static_cast<uint32_t>(AscendC::GetBlockNum());
        for (uint32_t item = AscendC::GetBlockIdx(); item < count; item += blockCount) {
            const int32_t firstRoute = routeIndicesGm.GetValue(routeRow + item);
            if (firstRoute < 0 || firstRoute >= static_cast<int32_t>(tiling.numRoutes)) {
                continue;
            }
            const int32_t tokenId = firstRoute / static_cast<int32_t>(tiling.topK);
            if (tokenId < 0 || tokenId >= static_cast<int32_t>(tiling.numTokens)) {
                continue;
            }
            const uint64_t tokenRoute = static_cast<uint64_t>(tokenId) * tiling.topK;
            for (uint32_t position = 0; position < tiling.topK; ++position) {
                const uint64_t routeIndex = tokenRoute + position;
                if (flatLogicalGm.GetValue(routeIndex) != logicalExpert) {
                    continue;
                }
                ApplyRoute(routeIndex, tokenId, destinationRank);
            }
        }
        FlushState();
    }

private:
    __aicore__ inline void ApplyRoute(uint64_t routeIndex, int32_t tokenId, int32_t destinationRank)
    {
        const int32_t currentRank = static_cast<int32_t>(routeRanksGm.GetValue(routeIndex));
        if (currentRank < 0 || currentRank >= static_cast<int32_t>(tiling.epSize)) {
            return;
        }
        const int32_t destinationScore = routeScoresGm.GetValue(routeIndex * tiling.epSize + destinationRank);
        const int32_t minimumScore = minimumScoresGm.GetValue(routeIndex);
        const int32_t rawTies = tieCountGm.GetValue(routeIndex);
        const int32_t ties = rawTies < 1 ? 1 :
            (rawTies > static_cast<int32_t>(tiling.epSize) ? static_cast<int32_t>(tiling.epSize) : rawTies);
        const uint64_t tieOffset = routeIndex * tiling.epSize;
        int32_t candidateRank = currentRank;
        int32_t insertion = 0;

        if (destinationScore < minimumScore) {
            candidateRank = destinationRank;
        } else if (destinationScore == minimumScore) {
            for (int32_t index = 0; index < ties; ++index) {
                insertion += tiedRankOrderGm.GetValue(tieOffset + index) < destinationRank ? 1 : 0;
            }
            if (insertion >= static_cast<int32_t>(tiling.epSize)) {
                return;
            }
            const int64_t hash = routeHashesGm.GetValue(routeIndex);
            const int32_t target = static_cast<int32_t>(hash % static_cast<int64_t>(ties + 1));
            if (target == insertion) {
                candidateRank = destinationRank;
            } else {
                const int32_t existing = target - (target > insertion ? 1 : 0);
                candidateRank = static_cast<int32_t>(tiedRankOrderGm.GetValue(tieOffset + existing));
            }
            if (candidateRank < 0 || candidateRank >= static_cast<int32_t>(tiling.epSize)) {
                return;
            }
        }

        if (destinationScore < minimumScore) {
            for (uint32_t index = 0; index < tiling.epSize; ++index) {
                tiedRankOrderGm.SetValue(tieOffset + index, static_cast<int64_t>(tiling.epSize));
            }
            tiedRankOrderGm.SetValue(tieOffset, static_cast<int64_t>(destinationRank));
            tieCountGm.SetValue(routeIndex, 1);
            minimumScoresGm.SetValue(routeIndex, destinationScore);
        } else if (destinationScore == minimumScore) {
            for (int32_t index = static_cast<int32_t>(tiling.epSize) - 1; index > insertion; --index) {
                tiedRankOrderGm.SetValue(tieOffset + index, tiedRankOrderGm.GetValue(tieOffset + index - 1));
            }
            tiedRankOrderGm.SetValue(tieOffset + insertion, static_cast<int64_t>(destinationRank));
            tieCountGm.SetValue(routeIndex, ties + 1);
        }

        if (candidateRank != currentRank) {
            UpdateLevel(tokenId, currentRank, candidateRank, tiling.levelSize0, tiling.levelOffset0);
            if (tiling.numLevels > 1) {
                UpdateLevel(tokenId, currentRank, candidateRank, tiling.levelSize1, tiling.levelOffset1);
            }
            if (tiling.numLevels > 2) {
                UpdateLevel(tokenId, currentRank, candidateRank, tiling.levelSize2, tiling.levelOffset2);
            }
            routeRanksGm.SetValue(routeIndex, static_cast<int64_t>(candidateRank));
        }
    }

    __aicore__ inline void UpdateLevel(
        int32_t tokenId, int32_t oldRank, int32_t newRank, uint32_t groupSize, uint32_t outputOffset)
    {
        const uint32_t oldGroup = static_cast<uint32_t>(oldRank) / groupSize;
        const uint32_t newGroup = static_cast<uint32_t>(newRank) / groupSize;
        if (oldGroup == newGroup) {
            return;
        }
        const uint64_t tokenOffset = static_cast<uint64_t>(tokenId) * tiling.totalGroups + outputOffset;
        const int32_t oldCount = tokenGroupCountsGm.GetValue(tokenOffset + oldGroup);
        const int32_t newCount = tokenGroupCountsGm.GetValue(tokenOffset + newGroup);
        tokenGroupCountsGm.SetValue(tokenOffset + oldGroup, oldCount - 1);
        tokenGroupCountsGm.SetValue(tokenOffset + newGroup, newCount + 1);
    }

    __aicore__ inline void FlushState()
    {
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(routeRanksGm);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(minimumScoresGm);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(tieCountGm);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(tiedRankOrderGm);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(tokenGroupCountsGm);
    }

    AscendC::GlobalTensor<int32_t> routeIndicesGm;
    AscendC::GlobalTensor<int32_t> tokenCountsGm;
    AscendC::GlobalTensor<int64_t> flatLogicalGm;
    AscendC::GlobalTensor<int64_t> routeRanksGm;
    AscendC::GlobalTensor<int32_t> routeScoresGm;
    AscendC::GlobalTensor<int32_t> minimumScoresGm;
    AscendC::GlobalTensor<int32_t> tieCountGm;
    AscendC::GlobalTensor<int64_t> tiedRankOrderGm;
    AscendC::GlobalTensor<int64_t> routeHashesGm;
    AscendC::GlobalTensor<int32_t> tokenGroupCountsGm;
    AscendC::GlobalTensor<int64_t> logicalExpertGm;
    AscendC::GlobalTensor<int64_t> destinationRankGm;
    HiermoeReplicaApplyTilingData tiling;
};

extern "C" __global__ __aicore__ void hiermoe_replica_apply(
    GM_ADDR routeIndices,
    GM_ADDR tokenCounts,
    GM_ADDR flatLogical,
    GM_ADDR routeRanks,
    GM_ADDR routeScores,
    GM_ADDR minimumScores,
    GM_ADDR tieCount,
    GM_ADDR tiedRankOrder,
    GM_ADDR routeHashes,
    GM_ADDR tokenGroupCounts,
    GM_ADDR logicalExpert,
    GM_ADDR destinationRank,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeReplicaApplyTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeReplicaApply op;
    op.Init(routeIndices, tokenCounts, flatLogical, routeRanks, routeScores, minimumScores, tieCount,
            tiedRankOrder, routeHashes, tokenGroupCounts, logicalExpert, destinationRank, tilingData);
    op.Process();
}
