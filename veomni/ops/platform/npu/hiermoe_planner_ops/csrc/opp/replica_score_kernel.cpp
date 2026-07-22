#include "kernel_operator.h"

struct HiermoeReplicaScoreTilingData {
    uint32_t numCandidateExperts;
    uint32_t tokenWidth;
    uint32_t epSize;
    uint32_t numLevels;
    uint32_t totalGroups;
    uint32_t outputWidth;
    uint32_t topK;
    uint32_t levelSize0;
    uint32_t levelSize1;
    uint32_t levelSize2;
    uint32_t levelGroups0;
    uint32_t levelGroups1;
    uint32_t levelGroups2;
    uint32_t levelOffset0;
    uint32_t levelOffset1;
    uint32_t levelOffset2;
};

class KernelHiermoeReplicaScore {
public:
    __aicore__ inline void Init(
        GM_ADDR routeIndices,
        GM_ADDR multiplicities,
        GM_ADDR tokenCounts,
        GM_ADDR routeRanks,
        GM_ADDR routeScores,
        GM_ADDR minimumScores,
        GM_ADDR tieCount,
        GM_ADDR tiedRankOrder,
        GM_ADDR routeHashes,
        GM_ADDR tokenGroupCounts,
        GM_ADDR candidateExperts,
        GM_ADDR deltas,
        const HiermoeReplicaScoreTilingData &tiling)
    {
        this->tiling = tiling;
        routeIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(routeIndices));
        multiplicitiesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(multiplicities));
        tokenCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenCounts));
        routeRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(routeRanks));
        routeScoresGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(routeScores));
        minimumScoresGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(minimumScores));
        tieCountGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tieCount));
        tiedRankOrderGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tiedRankOrder));
        routeHashesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(routeHashes));
        tokenGroupCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenGroupCounts));
        candidateExpertsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(candidateExperts));
        deltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(deltas));
        pipe.InitBuffer(deltaBuffer, tiling.outputWidth * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<int32_t> delta = deltaBuffer.Get<int32_t>();
        const uint32_t blockCount = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t numPairs = tiling.numCandidateExperts * tiling.epSize;
        for (uint32_t pair = AscendC::GetBlockIdx(); pair < numPairs; pair += blockCount) {
            const uint32_t candidateRow = pair / tiling.epSize;
            const uint32_t expertRow = static_cast<uint32_t>(candidateExpertsGm.GetValue(candidateRow));
            const int32_t destinationRank = static_cast<int32_t>(pair % tiling.epSize);
            AscendC::Duplicate(delta, static_cast<int32_t>(0), tiling.outputWidth);

            const uint32_t count = static_cast<uint32_t>(tokenCountsGm.GetValue(expertRow));
            const uint64_t rowOffset = static_cast<uint64_t>(expertRow) * tiling.tokenWidth;
            for (uint32_t item = 0; item < count; ++item) {
                const uint64_t packedIndex = rowOffset + item;
                const int32_t routeIndex = routeIndicesGm.GetValue(packedIndex);
                const int32_t tokenId = routeIndex / static_cast<int32_t>(tiling.topK);
                const int32_t multiplicity = multiplicitiesGm.GetValue(packedIndex);
                const int32_t currentRank = static_cast<int32_t>(routeRanksGm.GetValue(routeIndex));
                const int32_t destinationScore = routeScoresGm.GetValue(
                    static_cast<uint64_t>(routeIndex) * tiling.epSize + destinationRank);
                const int32_t minimumScore = minimumScoresGm.GetValue(routeIndex);
                int32_t candidateRank = currentRank;

                if (destinationScore < minimumScore) {
                    candidateRank = destinationRank;
                } else if (destinationScore == minimumScore) {
                    const int32_t ties = tieCountGm.GetValue(routeIndex);
                    int32_t insertion = 0;
                    const uint64_t tieOffset = static_cast<uint64_t>(routeIndex) * tiling.epSize;
                    for (int32_t index = 0; index < ties; ++index) {
                        insertion += tiedRankOrderGm.GetValue(tieOffset + index) < destinationRank ? 1 : 0;
                    }
                    const int64_t hash = routeHashesGm.GetValue(routeIndex);
                    const int32_t target = static_cast<int32_t>(hash % static_cast<int64_t>(ties + 1));
                    if (target == insertion) {
                        candidateRank = destinationRank;
                    } else {
                        int32_t existing = target - (target > insertion ? 1 : 0);
                        if (existing >= static_cast<int32_t>(tiling.epSize)) {
                            existing = static_cast<int32_t>(tiling.epSize - 1);
                        }
                        candidateRank = static_cast<int32_t>(tiedRankOrderGm.GetValue(tieOffset + existing));
                    }
                }

                if (candidateRank == currentRank) {
                    continue;
                }
                AddDelta(delta, tiling.totalGroups + static_cast<uint32_t>(currentRank), -multiplicity);
                AddDelta(delta, tiling.totalGroups + static_cast<uint32_t>(candidateRank), multiplicity);
                UpdateLevel(delta, tokenId, currentRank, candidateRank, multiplicity, tiling.levelSize0,
                            tiling.levelGroups0, tiling.levelOffset0);
                if (tiling.numLevels > 1) {
                    UpdateLevel(delta, tokenId, currentRank, candidateRank, multiplicity, tiling.levelSize1,
                                tiling.levelGroups1, tiling.levelOffset1);
                }
                if (tiling.numLevels > 2) {
                    UpdateLevel(delta, tokenId, currentRank, candidateRank, multiplicity, tiling.levelSize2,
                                tiling.levelGroups2, tiling.levelOffset2);
                }
            }
            AscendC::DataCopy(deltasGm[pair * tiling.outputWidth], delta, tiling.outputWidth);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void AddDelta(AscendC::LocalTensor<int32_t> &delta, uint32_t index, int32_t value)
    {
        delta.SetValue(index, delta.GetValue(index) + value);
    }

    __aicore__ inline void UpdateLevel(
        AscendC::LocalTensor<int32_t> &delta,
        int32_t tokenId,
        int32_t oldRank,
        int32_t newRank,
        int32_t multiplicity,
        uint32_t groupSize,
        uint32_t numGroups,
        uint32_t outputOffset)
    {
        const uint32_t oldGroup = static_cast<uint32_t>(oldRank) / groupSize;
        const uint32_t newGroup = static_cast<uint32_t>(newRank) / groupSize;
        if (oldGroup == newGroup) {
            return;
        }
        const uint64_t tokenOffset = static_cast<uint64_t>(tokenId) * tiling.totalGroups + outputOffset;
        const int32_t oldOccupancy = tokenGroupCountsGm.GetValue(tokenOffset + oldGroup);
        const int32_t newOccupancy = tokenGroupCountsGm.GetValue(tokenOffset + newGroup);
        if (oldOccupancy == multiplicity) {
            AddDelta(delta, outputOffset + oldGroup, -1);
        }
        if (newOccupancy == 0) {
            AddDelta(delta, outputOffset + newGroup, 1);
        }
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> deltaBuffer;
    AscendC::GlobalTensor<int32_t> routeIndicesGm;
    AscendC::GlobalTensor<int32_t> multiplicitiesGm;
    AscendC::GlobalTensor<int32_t> tokenCountsGm;
    AscendC::GlobalTensor<int64_t> routeRanksGm;
    AscendC::GlobalTensor<int32_t> routeScoresGm;
    AscendC::GlobalTensor<int32_t> minimumScoresGm;
    AscendC::GlobalTensor<int32_t> tieCountGm;
    AscendC::GlobalTensor<int64_t> tiedRankOrderGm;
    AscendC::GlobalTensor<int64_t> routeHashesGm;
    AscendC::GlobalTensor<int32_t> tokenGroupCountsGm;
    AscendC::GlobalTensor<int64_t> candidateExpertsGm;
    AscendC::GlobalTensor<int32_t> deltasGm;
    HiermoeReplicaScoreTilingData tiling;
};

extern "C" __global__ __aicore__ void hiermoe_replica_score(
    GM_ADDR routeIndices,
    GM_ADDR multiplicities,
    GM_ADDR tokenCounts,
    GM_ADDR routeRanks,
    GM_ADDR routeScores,
    GM_ADDR minimumScores,
    GM_ADDR tieCount,
    GM_ADDR tiedRankOrder,
    GM_ADDR routeHashes,
    GM_ADDR tokenGroupCounts,
    GM_ADDR candidateExperts,
    GM_ADDR deltas,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeReplicaScoreTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeReplicaScore op;
    op.Init(routeIndices, multiplicities, tokenCounts, routeRanks, routeScores, minimumScores,
            tieCount, tiedRankOrder, routeHashes, tokenGroupCounts, candidateExperts, deltas, tilingData);
    op.Process();
}
