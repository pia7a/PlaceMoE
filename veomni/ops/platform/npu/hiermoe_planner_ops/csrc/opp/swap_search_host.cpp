#include <algorithm>

#include "register/op_def_registry.h"
#include "swap_search_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus SwapSearchTiling(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeSwapSearchTilingData>();
    const auto sampleShape = context->GetInputShape(0)->GetOriginShape();
    const auto ownerShape = context->GetInputShape(4)->GetOriginShape();
    const auto layoutShape = context->GetInputShape(3)->GetOriginShape();
    const auto *attrs = context->GetAttrs();

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    tiling->blockCount = std::min(64U, std::max(1U, platform.GetCoreNumAiv()));
    tiling->numSamples = static_cast<uint32_t>(sampleShape.GetDim(0));
    const uint32_t tokensPerBlock = (tiling->numSamples + tiling->blockCount - 1U) / tiling->blockCount;
    tiling->tokenWidth = (tokensPerBlock + 15U) / 16U * 16U;
    tiling->topK = static_cast<uint32_t>(sampleShape.GetDim(1));
    tiling->numExperts = static_cast<uint32_t>(ownerShape.GetDim(0));
    tiling->numSlots = static_cast<uint32_t>(layoutShape.GetDim(0));
    tiling->maxSwaps = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    tiling->slotsPerRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(1));
    tiling->epSize = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2));
    tiling->numLevels = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));
    tiling->levelSize0 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4));
    tiling->levelSize1 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
    tiling->levelSize2 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(6));
    tiling->payloadBytes = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(7));
    tiling->flatPayloadFactor = static_cast<float>(tiling->epSize * tiling->payloadBytes);
    tiling->interPayloadFactor0 = static_cast<float>(tiling->levelSize0 * tiling->payloadBytes);
    tiling->interPayloadFactor1 = static_cast<float>(
        tiling->levelSize0 > 0
            ? (static_cast<float>(tiling->levelSize1) / static_cast<float>(tiling->levelSize0))
                * static_cast<float>(tiling->payloadBytes)
            : 0.0F);
    tiling->intraPayloadFactor0 = static_cast<float>(
        tiling->levelSize0 > 0 ? (tiling->epSize / tiling->levelSize0) * tiling->payloadBytes : 0U);
    tiling->intraPayloadFactor1 = static_cast<float>(
        tiling->levelSize1 > 0 ? (tiling->epSize / tiling->levelSize1) * tiling->payloadBytes : 0U);
    tiling->communicationScale = *attrs->GetAttrPointer<float>(8);
    tiling->computePerAssignment = *attrs->GetAttrPointer<float>(9);
    tiling->a2aAlpha = *attrs->GetAttrPointer<float>(10);
    tiling->a2aBeta = *attrs->GetAttrPointer<float>(11);
    tiling->interAlpha0 = *attrs->GetAttrPointer<float>(12);
    tiling->interBeta0 = *attrs->GetAttrPointer<float>(13);
    tiling->interAlpha1 = *attrs->GetAttrPointer<float>(14);
    tiling->interBeta1 = *attrs->GetAttrPointer<float>(15);
    tiling->intraAlpha = *attrs->GetAttrPointer<float>(16);
    tiling->intraBeta = *attrs->GetAttrPointer<float>(17);
    tiling->chooseMinDimension = static_cast<uint32_t>(*attrs->GetAttrPointer<bool>(18));

    const uint32_t levelSizes[] = {tiling->levelSize0, tiling->levelSize1, tiling->levelSize2};
    uint32_t totalGroups = 0;
    uint32_t offset = 0;
    uint32_t *levelGroups[] = {&tiling->levelGroups0, &tiling->levelGroups1, &tiling->levelGroups2};
    uint32_t *levelOffsets[] = {&tiling->levelOffset0, &tiling->levelOffset1, &tiling->levelOffset2};
    for (uint32_t level = 0; level < 3; ++level) {
        *levelOffsets[level] = offset;
        *levelGroups[level] = level < tiling->numLevels ? tiling->epSize / levelSizes[level] : 0;
        totalGroups += *levelGroups[level];
        offset += *levelGroups[level];
    }
    tiling->totalGroups = totalGroups;
    const uint32_t expertGroupStride = (totalGroups + 31U) / 32U * 32U;
    const uint32_t solePairStride = (tiling->numExperts + 31U) / 32U * 32U;
    tiling->statsWidth = totalGroups * 32U + tiling->numExperts * 32U
        + tiling->numExperts * expertGroupStride + tiling->numExperts * 32U
        + tiling->numExperts * tiling->numLevels * solePairStride;
    tiling->statsStride = (tiling->statsWidth + 31U) / 32U * 32U;

    context->SetBlockDim(tiling->blockCount);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeSwapSearch : public OpDef {
public:
    explicit HiermoeSwapSearch(const char *name) : OpDef(name)
    {
        this->Input("sample_routes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("sample_weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("assignment_counts").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("slot_to_logical").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("owner_slots").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("updated_layout").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("updated_owners").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("actions").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("metadata").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("float_workspace").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("int_workspace").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Attr("max_swaps").AttrType(REQUIRED).Int();
        this->Attr("slots_per_rank").AttrType(REQUIRED).Int();
        this->Attr("ep_size").AttrType(REQUIRED).Int();
        this->Attr("num_levels").AttrType(REQUIRED).Int();
        this->Attr("level_size0").AttrType(REQUIRED).Int();
        this->Attr("level_size1").AttrType(REQUIRED).Int();
        this->Attr("level_size2").AttrType(REQUIRED).Int();
        this->Attr("payload_bytes").AttrType(REQUIRED).Int();
        this->Attr("communication_scale").AttrType(REQUIRED).Float();
        this->Attr("compute_per_assignment").AttrType(REQUIRED).Float();
        this->Attr("a2a_alpha").AttrType(REQUIRED).Float();
        this->Attr("a2a_beta").AttrType(REQUIRED).Float();
        this->Attr("inter_alpha0").AttrType(REQUIRED).Float();
        this->Attr("inter_beta0").AttrType(REQUIRED).Float();
        this->Attr("inter_alpha1").AttrType(REQUIRED).Float();
        this->Attr("inter_beta1").AttrType(REQUIRED).Float();
        this->Attr("intra_alpha").AttrType(REQUIRED).Float();
        this->Attr("intra_beta").AttrType(REQUIRED).Float();
        this->Attr("choose_min_dimension").AttrType(REQUIRED).Bool();
        this->AICore().SetTiling(optiling::SwapSearchTiling).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeSwapSearch);
} // namespace ops
