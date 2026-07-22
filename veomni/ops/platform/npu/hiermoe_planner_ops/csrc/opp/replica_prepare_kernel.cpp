#include "kernel_operator.h"

struct HiermoeReplicaPrepareTilingData {
    uint32_t numTokens;
    uint32_t tokenWidth;
    uint32_t topK;
    uint32_t numExperts;
};

class KernelHiermoeReplicaPrepare {
public:
    __aicore__ inline void Init(
        GM_ADDR selected,
        GM_ADDR routeIndices,
        GM_ADDR multiplicities,
        GM_ADDR tokenCounts,
        const HiermoeReplicaPrepareTilingData &tiling)
    {
        this->tiling = tiling;
        selectedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(selected));
        routeIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(routeIndices));
        multiplicitiesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(multiplicities));
        tokenCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenCounts));
        pipe.InitBuffer(routeBuffer, tiling.tokenWidth * sizeof(int32_t));
        pipe.InitBuffer(multiplicityBuffer, tiling.tokenWidth * sizeof(int32_t));
        pipe.InitBuffer(countBuffer, 8 * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockCount = static_cast<uint32_t>(AscendC::GetBlockNum());
        AscendC::LocalTensor<int32_t> routeLocal = routeBuffer.Get<int32_t>();
        AscendC::LocalTensor<int32_t> multiplicityLocal = multiplicityBuffer.Get<int32_t>();
        AscendC::LocalTensor<int32_t> countLocal = countBuffer.Get<int32_t>();
        for (uint32_t expert = AscendC::GetBlockIdx(); expert < tiling.numExperts; expert += blockCount) {
            const uint64_t outputOffset = static_cast<uint64_t>(expert) * tiling.tokenWidth;
            uint32_t count = 0;
            for (uint32_t token = 0; token < tiling.numTokens; ++token) {
                const uint64_t routeOffset = static_cast<uint64_t>(token) * tiling.topK;
                int32_t firstPosition = -1;
                int32_t multiplicity = 0;
                for (uint32_t position = 0; position < tiling.topK; ++position) {
                    if (selectedGm.GetValue(routeOffset + position) == static_cast<int64_t>(expert)) {
                        if (firstPosition < 0) {
                            firstPosition = static_cast<int32_t>(position);
                        }
                        ++multiplicity;
                    }
                }
                if (multiplicity > 0) {
                    routeLocal.SetValue(count, static_cast<int32_t>(routeOffset + static_cast<uint32_t>(firstPosition)));
                    multiplicityLocal.SetValue(count, multiplicity);
                    ++count;
                }
            }
            const uint32_t alignedCount = (count + 7U) / 8U * 8U;
            if (alignedCount > 0) {
                AscendC::DataCopy(routeIndicesGm[outputOffset], routeLocal, alignedCount);
                AscendC::DataCopy(multiplicitiesGm[outputOffset], multiplicityLocal, alignedCount);
            }
            countLocal.SetValue(0, static_cast<int32_t>(count));
            AscendC::DataCopy(tokenCountsGm[static_cast<uint64_t>(expert) * 8U], countLocal, 8U);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> routeBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> multiplicityBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> countBuffer;
    AscendC::GlobalTensor<int64_t> selectedGm;
    AscendC::GlobalTensor<int32_t> routeIndicesGm;
    AscendC::GlobalTensor<int32_t> multiplicitiesGm;
    AscendC::GlobalTensor<int32_t> tokenCountsGm;
    HiermoeReplicaPrepareTilingData tiling;
};

extern "C" __global__ __aicore__ void hiermoe_replica_prepare(
    GM_ADDR selected,
    GM_ADDR routeIndices,
    GM_ADDR multiplicities,
    GM_ADDR tokenCounts,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HiermoeReplicaPrepareTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelHiermoeReplicaPrepare op;
    op.Init(selected, routeIndices, multiplicities, tokenCounts, tilingData);
    op.Process();
}
