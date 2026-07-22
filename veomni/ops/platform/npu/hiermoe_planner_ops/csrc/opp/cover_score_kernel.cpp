#include "kernel_operator.h"

#include "cover_score_tiling.h"

class KernelHiermoeCoverScore {
public:
    __aicore__ inline void Init(
        GM_ADDR selected,
        GM_ADDR routeIndices,
        GM_ADDR multiplicities,
        GM_ADDR tokenCounts,
        GM_ADDR routeRanks,
        GM_ADDR routeHashes,
        GM_ADDR tokenGroupCounts,
        GM_ADDR copySlots,
        GM_ADDR candidateRows,
        GM_ADDR deltas,
        const HiermoeCoverScoreTilingData &tiling)
    {
        this->tiling = tiling;
        selectedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(selected));
        routeIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(routeIndices));
        multiplicitiesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(multiplicities));
        tokenCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenCounts));
        routeRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(routeRanks));
        routeHashesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(routeHashes));
        tokenGroupCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenGroupCounts));
        copySlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(copySlots));
        candidateRowsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(candidateRows));
        deltasGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(deltas));
        pipe.InitBuffer(deltaBuffer, tiling.outputWidth * sizeof(int32_t));
        pipe.InitBuffer(lhsChoiceBuffer, 16U * sizeof(int32_t));
        pipe.InitBuffer(rhsChoiceBuffer, 16U * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<int32_t> delta = deltaBuffer.Get<int32_t>();
        AscendC::LocalTensor<int32_t> lhsChoices = lhsChoiceBuffer.Get<int32_t>();
        AscendC::LocalTensor<int32_t> rhsChoices = rhsChoiceBuffer.Get<int32_t>();
        const uint32_t blockCount = static_cast<uint32_t>(AscendC::GetBlockNum());
        for (uint32_t candidate = AscendC::GetBlockIdx(); candidate < tiling.numCandidates;
             candidate += blockCount) {
            AscendC::Duplicate(delta, static_cast<int32_t>(0), tiling.outputWidth);
            const uint64_t rowOffset = static_cast<uint64_t>(candidate) * 5U;
            const int32_t kind = static_cast<int32_t>(candidateRowsGm.GetValue(rowOffset));
            const int32_t sourceSlot = static_cast<int32_t>(candidateRowsGm.GetValue(rowOffset + 1U));
            const int32_t destinationSlot = static_cast<int32_t>(candidateRowsGm.GetValue(rowOffset + 2U));
            const int32_t lhs = static_cast<int32_t>(candidateRowsGm.GetValue(rowOffset + 3U));
            const int32_t rhs = static_cast<int32_t>(candidateRowsGm.GetValue(rowOffset + 4U));

            const int32_t lhsTies =
                PrepareChoices(lhsChoices, lhs, kind, false, sourceSlot, destinationSlot);
            int32_t rhsTies = 0;
            if (rhs >= 0) {
                rhsTies = PrepareChoices(rhsChoices, rhs, kind, true, sourceSlot, destinationSlot);
            }
            ProcessExpertTokens(delta, lhsChoices, lhsTies, rhsChoices, rhsTies, lhs, lhs, rhs, false, false);
            if (rhs >= 0) {
                ProcessExpertTokens(
                    delta, lhsChoices, lhsTies, rhsChoices, rhsTies, rhs, lhs, rhs, true, true);
            }
            AscendC::DataCopy(deltasGm[static_cast<uint64_t>(candidate) * tiling.outputWidth], delta,
                              tiling.outputWidth);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void ProcessExpertTokens(
        AscendC::LocalTensor<int32_t> &delta,
        AscendC::LocalTensor<int32_t> &lhsChoices,
        int32_t lhsTies,
        AscendC::LocalTensor<int32_t> &rhsChoices,
        int32_t rhsTies,
        int32_t iterateExpert,
        int32_t lhs,
        int32_t rhs,
        bool skipShared,
        bool iterateRhs)
    {
        const uint32_t count = static_cast<uint32_t>(tokenCountsGm.GetValue(iterateExpert));
        const uint64_t expertOffset = static_cast<uint64_t>(iterateExpert) * tiling.tokenWidth;
        for (uint32_t item = 0; item < count; ++item) {
            const uint64_t packedIndex = expertOffset + item;
            const int32_t routeIndex = routeIndicesGm.GetValue(packedIndex);
            const int32_t multiplicity = multiplicitiesGm.GetValue(packedIndex);
            const int32_t tokenId = routeIndex / static_cast<int32_t>(tiling.topK);
            if (skipShared && HasExpert(tokenId, lhs)) {
                continue;
            }
            ProcessToken(
                delta, lhsChoices, lhsTies, rhsChoices, rhsTies, tokenId, lhs, rhs,
                routeIndex, multiplicity, iterateRhs);
        }
    }

    __aicore__ inline bool HasExpert(int32_t tokenId, int32_t expert)
    {
        if (expert < 0) {
            return false;
        }
        const uint64_t tokenOffset = static_cast<uint64_t>(tokenId) * tiling.topK;
        for (uint32_t route = 0; route < tiling.topK; ++route) {
            if (selectedGm.GetValue(tokenOffset + route) == expert) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void FindRoute(int32_t tokenId, int32_t expert, int32_t &routeIndex, int32_t &multiplicity)
    {
        routeIndex = -1;
        multiplicity = 0;
        if (expert < 0) {
            return;
        }
        const uint64_t tokenOffset = static_cast<uint64_t>(tokenId) * tiling.topK;
        for (uint32_t route = 0; route < tiling.topK; ++route) {
            if (selectedGm.GetValue(tokenOffset + route) == expert) {
                if (routeIndex < 0) {
                    routeIndex = static_cast<int32_t>(tokenOffset + route);
                }
                ++multiplicity;
            }
        }
    }

    __aicore__ inline int32_t OptionSlot(
        uint32_t option,
        int32_t expert,
        int32_t kind,
        bool rhsRole,
        int32_t sourceSlot,
        int32_t destinationSlot)
    {
        if (option == tiling.copyWidth) {
            return kind == 1 && !rhsRole ? destinationSlot : static_cast<int32_t>(tiling.numSlots);
        }
        int32_t slot = static_cast<int32_t>(
            copySlotsGm.GetValue(static_cast<uint64_t>(expert) * tiling.copyWidth + option));
        if (slot >= static_cast<int32_t>(tiling.numSlots)) {
            return slot;
        }
        if (kind == 0) {
            if (!rhsRole && slot == sourceSlot) {
                slot = destinationSlot;
            } else if (rhsRole && slot == destinationSlot) {
                slot = sourceSlot;
            }
        } else if (rhsRole && slot == destinationSlot) {
            slot = static_cast<int32_t>(tiling.numSlots);
        }
        return slot;
    }

    __aicore__ inline int32_t RankDistance(int32_t rank)
    {
        if (rank == static_cast<int32_t>(tiling.sourceRank)) {
            return 0;
        }
        int32_t distance = static_cast<int32_t>(tiling.numLevels);
        if (tiling.numLevels > 2 &&
            rank / static_cast<int32_t>(tiling.levelSize2) ==
                static_cast<int32_t>(tiling.sourceRank / tiling.levelSize2)) {
            distance = 2;
        }
        if (tiling.numLevels > 1 &&
            rank / static_cast<int32_t>(tiling.levelSize1) ==
                static_cast<int32_t>(tiling.sourceRank / tiling.levelSize1)) {
            distance = 1;
        }
        return distance;
    }

    __aicore__ inline int32_t PrepareChoices(
        AscendC::LocalTensor<int32_t> &choices,
        int32_t expert,
        int32_t kind,
        bool rhsRole,
        int32_t sourceSlot,
        int32_t destinationSlot)
    {
        int32_t minimumDistance = static_cast<int32_t>(tiling.numLevels + 1U);
        int32_t ties = 0;
        for (uint32_t option = 0; option <= tiling.copyWidth; ++option) {
            const int32_t slot = OptionSlot(option, expert, kind, rhsRole, sourceSlot, destinationSlot);
            if (slot >= static_cast<int32_t>(tiling.numSlots)) {
                continue;
            }
            const int32_t distance = RankDistance(slot / static_cast<int32_t>(tiling.slotsPerRank));
            if (distance < minimumDistance) {
                minimumDistance = distance;
                ties = 1;
            } else if (distance == minimumDistance) {
                ++ties;
            }
        }
        if (ties <= 0) {
            choices.SetValue(0, 0);
            return 1;
        }
        int32_t previousSlot = -1;
        int32_t chosenSlot = -1;
        for (int32_t order = 0; order < ties; ++order) {
            chosenSlot = static_cast<int32_t>(tiling.numSlots);
            for (uint32_t option = 0; option <= tiling.copyWidth; ++option) {
                const int32_t slot = OptionSlot(option, expert, kind, rhsRole, sourceSlot, destinationSlot);
                if (slot >= static_cast<int32_t>(tiling.numSlots) || slot <= previousSlot) {
                    continue;
                }
                const int32_t distance = RankDistance(slot / static_cast<int32_t>(tiling.slotsPerRank));
                if (distance == minimumDistance && slot < chosenSlot) {
                    chosenSlot = slot;
                }
            }
            previousSlot = chosenSlot;
            choices.SetValue(order, chosenSlot / static_cast<int32_t>(tiling.slotsPerRank));
        }
        return ties;
    }

    __aicore__ inline int32_t ChoosePreparedRank(
        AscendC::LocalTensor<int32_t> &choices,
        int32_t ties,
        int64_t routeHash)
    {
        const int32_t target = static_cast<int32_t>(routeHash % static_cast<int64_t>(ties));
        return choices.GetValue(target);
    }

    __aicore__ inline void ProcessToken(
        AscendC::LocalTensor<int32_t> &delta,
        AscendC::LocalTensor<int32_t> &lhsChoices,
        int32_t lhsTies,
        AscendC::LocalTensor<int32_t> &rhsChoices,
        int32_t rhsTies,
        int32_t tokenId,
        int32_t lhs,
        int32_t rhs,
        int32_t knownRoute,
        int32_t knownMultiplicity,
        bool knownIsRhs)
    {
        int32_t lhsRoute;
        int32_t lhsMultiplicity;
        int32_t rhsRoute;
        int32_t rhsMultiplicity;
        if (knownIsRhs) {
            lhsRoute = -1;
            lhsMultiplicity = 0;
            rhsRoute = knownRoute;
            rhsMultiplicity = knownMultiplicity;
        } else {
            lhsRoute = knownRoute;
            lhsMultiplicity = knownMultiplicity;
            FindRoute(tokenId, rhs, rhsRoute, rhsMultiplicity);
        }

        int32_t lhsOld = 0;
        int32_t lhsNew = 0;
        if (lhsRoute >= 0) {
            lhsOld = static_cast<int32_t>(routeRanksGm.GetValue(lhsRoute));
            lhsNew = ChoosePreparedRank(lhsChoices, lhsTies, routeHashesGm.GetValue(lhsRoute));
            if (lhsOld == lhsNew) {
                lhsMultiplicity = 0;
            }
        }
        int32_t rhsOld = 0;
        int32_t rhsNew = 0;
        if (rhsRoute >= 0) {
            rhsOld = static_cast<int32_t>(routeRanksGm.GetValue(rhsRoute));
            rhsNew = ChoosePreparedRank(rhsChoices, rhsTies, routeHashesGm.GetValue(rhsRoute));
            if (rhsOld == rhsNew) {
                rhsMultiplicity = 0;
            }
        }
        if (lhsMultiplicity == 0 && rhsMultiplicity == 0) {
            return;
        }

        UpdateLevel(delta, tokenId, lhsOld, lhsNew, lhsMultiplicity, rhsOld, rhsNew, rhsMultiplicity,
                    tiling.levelSize0, tiling.levelOffset0);
        if (tiling.numLevels > 1) {
            UpdateLevel(delta, tokenId, lhsOld, lhsNew, lhsMultiplicity, rhsOld, rhsNew, rhsMultiplicity,
                        tiling.levelSize1, tiling.levelOffset1);
        }
        if (tiling.numLevels > 2) {
            UpdateLevel(delta, tokenId, lhsOld, lhsNew, lhsMultiplicity, rhsOld, rhsNew, rhsMultiplicity,
                        tiling.levelSize2, tiling.levelOffset2);
        }
    }

    __aicore__ inline void UpdateLevel(
        AscendC::LocalTensor<int32_t> &delta,
        int32_t tokenId,
        int32_t lhsOld,
        int32_t lhsNew,
        int32_t lhsMultiplicity,
        int32_t rhsOld,
        int32_t rhsNew,
        int32_t rhsMultiplicity,
        uint32_t groupSize,
        uint32_t outputOffset)
    {
        if (rhsMultiplicity == 0) {
            UpdateSingleLevel(delta, tokenId, lhsOld, lhsNew, lhsMultiplicity, groupSize, outputOffset);
            return;
        }
        if (lhsMultiplicity == 0) {
            UpdateSingleLevel(delta, tokenId, rhsOld, rhsNew, rhsMultiplicity, groupSize, outputOffset);
            return;
        }

        const int32_t groups[4] = {
            lhsOld / static_cast<int32_t>(groupSize),
            lhsNew / static_cast<int32_t>(groupSize),
            rhsOld / static_cast<int32_t>(groupSize),
            rhsNew / static_cast<int32_t>(groupSize)};
        const int32_t values[4] = {
            -lhsMultiplicity, lhsMultiplicity, -rhsMultiplicity, rhsMultiplicity};
        for (uint32_t position = 0; position < 4U; ++position) {
            bool first = true;
            for (uint32_t previous = 0; previous < position; ++previous) {
                first = first && groups[position] != groups[previous];
            }
            if (!first) {
                continue;
            }
            int32_t combined = 0;
            for (uint32_t other = 0; other < 4U; ++other) {
                combined += groups[other] == groups[position] ? values[other] : 0;
            }
            if (combined == 0) {
                continue;
            }
            const uint64_t occupancyIndex =
                static_cast<uint64_t>(tokenId) * tiling.totalGroups + outputOffset +
                static_cast<uint32_t>(groups[position]);
            const int32_t before = tokenGroupCountsGm.GetValue(occupancyIndex);
            const int32_t change = (before + combined > 0 ? 1 : 0) - (before > 0 ? 1 : 0);
            if (change != 0) {
                const uint32_t outputIndex = outputOffset + static_cast<uint32_t>(groups[position]);
                delta.SetValue(outputIndex, delta.GetValue(outputIndex) + change);
            }
        }
    }

    __aicore__ inline void UpdateSingleLevel(
        AscendC::LocalTensor<int32_t> &delta,
        int32_t tokenId,
        int32_t oldRank,
        int32_t newRank,
        int32_t multiplicity,
        uint32_t groupSize,
        uint32_t outputOffset)
    {
        const uint32_t oldGroup = static_cast<uint32_t>(oldRank) / groupSize;
        const uint32_t newGroup = static_cast<uint32_t>(newRank) / groupSize;
        if (multiplicity == 0 || oldGroup == newGroup) {
            return;
        }
        const uint64_t tokenOffset = static_cast<uint64_t>(tokenId) * tiling.totalGroups + outputOffset;
        const int32_t oldCount = tokenGroupCountsGm.GetValue(tokenOffset + oldGroup);
        if (oldCount > 0 && oldCount - multiplicity <= 0) {
            delta.SetValue(oldGroup + outputOffset, delta.GetValue(oldGroup + outputOffset) - 1);
        }
        const int32_t newCount = tokenGroupCountsGm.GetValue(tokenOffset + newGroup);
        if (newCount <= 0 && newCount + multiplicity > 0) {
            delta.SetValue(newGroup + outputOffset, delta.GetValue(newGroup + outputOffset) + 1);
        }
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> deltaBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lhsChoiceBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rhsChoiceBuffer;
    AscendC::GlobalTensor<int64_t> selectedGm;
    AscendC::GlobalTensor<int32_t> routeIndicesGm;
    AscendC::GlobalTensor<int32_t> multiplicitiesGm;
    AscendC::GlobalTensor<int32_t> tokenCountsGm;
    AscendC::GlobalTensor<int64_t> routeRanksGm;
    AscendC::GlobalTensor<int64_t> routeHashesGm;
    AscendC::GlobalTensor<int32_t> tokenGroupCountsGm;
    AscendC::GlobalTensor<int64_t> copySlotsGm;
    AscendC::GlobalTensor<int64_t> candidateRowsGm;
    AscendC::GlobalTensor<int32_t> deltasGm;
    HiermoeCoverScoreTilingData tiling;
};

extern "C" __global__ __aicore__ void hiermoe_cover_score(
    GM_ADDR selected,
    GM_ADDR routeIndices,
    GM_ADDR multiplicities,
    GM_ADDR tokenCounts,
    GM_ADDR routeRanks,
    GM_ADDR routeHashes,
    GM_ADDR tokenGroupCounts,
    GM_ADDR copySlots,
    GM_ADDR candidateRows,
    GM_ADDR deltas,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeCoverScoreTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeCoverScore op;
    op.Init(selected, routeIndices, multiplicities, tokenCounts, routeRanks, routeHashes,
            tokenGroupCounts, copySlots, candidateRows, deltas, tilingData);
    op.Process();
}
