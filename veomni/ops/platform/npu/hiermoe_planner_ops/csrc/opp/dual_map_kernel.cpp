#include "kernel_operator.h"

struct HiermoeDualMapTilingData {
    uint32_t numTokens;
    uint32_t tokensPerBlock;
    uint32_t topK;
    uint32_t numExperts;
    uint32_t maxCopies;
    uint32_t slotsPerRank;
    uint32_t sourceRank;
    uint32_t epSize;
    uint32_t numLevels;
    uint32_t levelSize0;
    uint32_t levelSize1;
    int64_t step;
    int64_t layerSeed;
};

class KernelHiermoeDualMap {
public:
    __aicore__ inline void Init(
        GM_ADDR selected,
        GM_ADDR copySlots,
        GM_ADDR copyCounts,
        GM_ADDR ownerRanks,
        GM_ADDR physical,
        const HiermoeDualMapTilingData &tiling)
    {
        this->tiling = tiling;
        selectedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(selected));
        copySlotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(copySlots));
        copyCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(copyCounts));
        ownerRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ownerRanks));
        physicalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(physical));
    }

    __aicore__ inline void Process()
    {
        const uint32_t firstToken = AscendC::GetBlockIdx() * tiling.tokensPerBlock;
        const uint32_t tokenLimit = firstToken + tiling.tokensPerBlock;
        const uint32_t lastToken = tokenLimit < tiling.numTokens ? tokenLimit : tiling.numTokens;
        for (uint32_t token = firstToken; token < lastToken; ++token) {
            for (uint32_t layout = 0; layout < 2; ++layout) {
                MapToken(layout, token);
            }
        }
    }

private:
    __aicore__ inline bool TokenDomainValid(uint32_t layout, uint32_t token) const
    {
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        const uint64_t ownerOffset = static_cast<uint64_t>(layout) * tiling.numExperts;
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

    __aicore__ inline bool RankAlreadyVisited(
        uint32_t layout,
        uint32_t token,
        int64_t logical,
        int64_t destinationRank,
        uint32_t groupSize)
    {
        if (groupSize == 1) {
            if (destinationRank == static_cast<int64_t>(tiling.sourceRank)) {
                return true;
            }
        } else if (
            destinationRank / static_cast<int64_t>(groupSize) ==
            static_cast<int64_t>(tiling.sourceRank) / static_cast<int64_t>(groupSize)) {
            return true;
        }
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        const uint64_t ownerOffset = static_cast<uint64_t>(layout) * tiling.numExperts;
        for (uint32_t position = 0; position < tiling.topK; ++position) {
            const int64_t other = selectedGm.GetValue(routeOffset + position);
            if (other == logical) {
                continue;
            }
            const int64_t owner = ownerRanksGm.GetValue(ownerOffset + static_cast<uint64_t>(other));
            if (groupSize == 1) {
                if (owner == destinationRank) {
                    return true;
                }
            } else if (owner / static_cast<int64_t>(groupSize) == destinationRank / static_cast<int64_t>(groupSize)) {
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
        uint32_t score = 0;
        if (tiling.numLevels > 1) {
            score = score * 2U + static_cast<uint32_t>(
                !RankAlreadyVisited(layout, token, logical, destinationRank, tiling.levelSize1));
        }
        if (tiling.numLevels > 0) {
            score = score * 2U + static_cast<uint32_t>(
                !RankAlreadyVisited(layout, token, logical, destinationRank, tiling.levelSize0));
        }
        score = score * 2U + static_cast<uint32_t>(
            !RankAlreadyVisited(layout, token, logical, destinationRank, 1));
        return score;
    }

    __aicore__ inline int64_t PositiveMod(int64_t value, int64_t modulus) const
    {
        value %= modulus;
        if (value < 0) {
            value += modulus;
        }
        return value;
    }

    __aicore__ inline int64_t RouteHash(uint32_t token, int64_t logical) const
    {
        constexpr int64_t modulus = 2147483647LL;
        int64_t value = PositiveMod(static_cast<int64_t>(token), modulus) * 1000003LL % modulus;
        value = (value + PositiveMod(logical, modulus) * 65537LL) % modulus;
        value = (value + PositiveMod(tiling.step, modulus) * 131LL) % modulus;
        value = (value + PositiveMod(tiling.layerSeed, modulus) * 17LL) % modulus;
        value = (value * 48271LL + 1LL) % modulus;
        return value % 1048573LL;
    }

    __aicore__ inline void MapToken(uint32_t layout, uint32_t token)
    {
        const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
        const uint64_t tableBase = static_cast<uint64_t>(layout) * tiling.numExperts;
        const uint64_t copyBase = tableBase * tiling.maxCopies;
        const uint64_t outputBase =
            (static_cast<uint64_t>(token) * 2U + static_cast<uint64_t>(layout)) * tiling.topK;
        if (!TokenDomainValid(layout, token)) {
            for (uint32_t position = 0; position < tiling.topK; ++position) {
                physicalGm.SetValue(outputBase + position, -1);
            }
            return;
        }
        for (uint32_t position = 0; position < tiling.topK; ++position) {
            const int64_t logical = selectedGm.GetValue(routeOffset + position);
            const int64_t rawCopies = copyCountsGm.GetValue(tableBase + logical);
            if (rawCopies <= 0 || rawCopies > static_cast<int64_t>(tiling.maxCopies)) {
                physicalGm.SetValue(outputBase + position, -1);
                continue;
            }
            const uint32_t copies = static_cast<uint32_t>(rawCopies);
            const int64_t slotLimit = static_cast<int64_t>(tiling.epSize) * tiling.slotsPerRank;
            bool validCopies = true;
            for (uint32_t copy = 0; copy < copies; ++copy) {
                const int64_t slot = copySlotsGm.GetValue(
                    copyBase + static_cast<uint64_t>(logical) * tiling.maxCopies + copy);
                validCopies = validCopies && slot >= 0 && slot < slotLimit;
            }
            if (!validCopies) {
                physicalGm.SetValue(outputBase + position, -1);
                continue;
            }
            uint32_t bestScore = 0xffffffffU;
            uint32_t tieCount = 0;
            for (uint32_t copy = 0; copy < copies; ++copy) {
                const int64_t slot = copySlotsGm.GetValue(copyBase + static_cast<uint64_t>(logical) * tiling.maxCopies + copy);
                const int64_t rank = slot / static_cast<int64_t>(tiling.slotsPerRank);
                const uint32_t score = CommunicationScore(layout, token, logical, rank);
                if (score < bestScore) {
                    bestScore = score;
                    tieCount = 1;
                } else if (score == bestScore) {
                    ++tieCount;
                }
            }
            uint32_t target = static_cast<uint32_t>(RouteHash(token, logical) % static_cast<int64_t>(tieCount));
            uint32_t seen = 0;
            int64_t chosen = -1;
            for (uint32_t copy = 0; copy < copies; ++copy) {
                const int64_t slot = copySlotsGm.GetValue(copyBase + static_cast<uint64_t>(logical) * tiling.maxCopies + copy);
                const int64_t rank = slot / static_cast<int64_t>(tiling.slotsPerRank);
                if (CommunicationScore(layout, token, logical, rank) != bestScore) {
                    continue;
                }
                if (seen == target) {
                    chosen = slot;
                    break;
                }
                ++seen;
            }
            physicalGm.SetValue(outputBase + position, chosen);
        }
    }

    AscendC::GlobalTensor<int64_t> selectedGm;
    AscendC::GlobalTensor<int64_t> copySlotsGm;
    AscendC::GlobalTensor<int64_t> copyCountsGm;
    AscendC::GlobalTensor<int64_t> ownerRanksGm;
    AscendC::GlobalTensor<int64_t> physicalGm;
    HiermoeDualMapTilingData tiling;
};

extern "C" __global__ __aicore__ void hiermoe_dual_map(
    GM_ADDR selected,
    GM_ADDR copySlots,
    GM_ADDR copyCounts,
    GM_ADDR ownerRanks,
    GM_ADDR physical,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeDualMapTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeDualMap op;
    op.Init(selected, copySlots, copyCounts, ownerRanks, physical, tilingData);
    op.Process();
}
