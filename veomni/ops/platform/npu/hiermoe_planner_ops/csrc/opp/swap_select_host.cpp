#include <algorithm>

#include "register/op_def_registry.h"
#include "swap_select_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus SwapSelectTiling(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeSwapSelectTilingData>();
    const auto sampleShape = context->GetInputShape(6)->GetOriginShape();
    const auto layoutShape = context->GetInputShape(8)->GetOriginShape();
    const auto ownerShape = context->GetInputShape(9)->GetOriginShape();
    const auto *attrs = context->GetAttrs();
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    tiling->blockCount = std::min(64U, std::max(1U, platform.GetCoreNumAiv()));
    tiling->numExperts = static_cast<uint32_t>(ownerShape.GetDim(0));
    tiling->numSlots = static_cast<uint32_t>(layoutShape.GetDim(0));
    tiling->numSamples = static_cast<uint32_t>(sampleShape.GetDim(0));
    tiling->topK = static_cast<uint32_t>(sampleShape.GetDim(1));
    const uint32_t tokensPerBlock = (tiling->numSamples + tiling->blockCount - 1U) / tiling->blockCount;
    tiling->tokenWidth = (tokensPerBlock + 15U) / 16U * 16U;
    tiling->maxSwaps = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    tiling->slotsPerRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(1));
    tiling->epSize = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2));
    tiling->localWorldSize = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));
    tiling->numLevels = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4));
    tiling->levelSize0 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
    tiling->levelSize1 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(6));
    tiling->levelSize2 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(7));
    tiling->payloadBytes = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(8));
    tiling->communicationScale = *attrs->GetAttrPointer<float>(9);
    tiling->computePerAssignment = *attrs->GetAttrPointer<float>(10);
    tiling->a2aAlpha = *attrs->GetAttrPointer<float>(11);
    tiling->a2aBeta = *attrs->GetAttrPointer<float>(12);
    tiling->interAlpha0 = *attrs->GetAttrPointer<float>(13);
    tiling->interBeta0 = *attrs->GetAttrPointer<float>(14);
    tiling->interAlpha1 = *attrs->GetAttrPointer<float>(15);
    tiling->interBeta1 = *attrs->GetAttrPointer<float>(16);
    tiling->intraAlpha = *attrs->GetAttrPointer<float>(17);
    tiling->intraBeta = *attrs->GetAttrPointer<float>(18);
    tiling->stateIntraAlpha = *attrs->GetAttrPointer<float>(19);
    tiling->stateIntraBeta = *attrs->GetAttrPointer<float>(20);
    tiling->stateInterAlpha = *attrs->GetAttrPointer<float>(21);
    tiling->stateInterBeta = *attrs->GetAttrPointer<float>(22);
    tiling->gatherIntraAlpha = *attrs->GetAttrPointer<float>(23);
    tiling->gatherIntraBeta = *attrs->GetAttrPointer<float>(24);
    tiling->gatherInterAlpha = *attrs->GetAttrPointer<float>(25);
    tiling->gatherInterBeta = *attrs->GetAttrPointer<float>(26);
    tiling->scatterIntraAlpha = *attrs->GetAttrPointer<float>(27);
    tiling->scatterIntraBeta = *attrs->GetAttrPointer<float>(28);
    tiling->scatterInterAlpha = *attrs->GetAttrPointer<float>(29);
    tiling->scatterInterBeta = *attrs->GetAttrPointer<float>(30);
    tiling->runtimeCostScale = *attrs->GetAttrPointer<float>(31);
    tiling->chooseMinDimension = static_cast<uint32_t>(*attrs->GetAttrPointer<bool>(32));

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
    context->SetBlockDim(tiling->blockCount);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeSwapSelect : public OpDef {
public:
    explicit HiermoeSwapSelect(const char *name) : OpDef(name)
    {
        this->Input("expert_token_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("expert_assignment_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("base_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("expert_group_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("sole_expert_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("sole_pair_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("sample_routes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("sample_weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("slot_to_logical").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("owner_slots").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("expert_state_bytes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("expert_gradient_bytes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("updated_layout").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("updated_owners").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("actions").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("metadata").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("float_workspace").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("int_workspace").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Attr("max_swaps").AttrType(REQUIRED).Int();
        this->Attr("slots_per_rank").AttrType(REQUIRED).Int();
        this->Attr("ep_size").AttrType(REQUIRED).Int();
        this->Attr("local_world_size").AttrType(REQUIRED).Int();
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
        this->Attr("state_intra_alpha").AttrType(REQUIRED).Float();
        this->Attr("state_intra_beta").AttrType(REQUIRED).Float();
        this->Attr("state_inter_alpha").AttrType(REQUIRED).Float();
        this->Attr("state_inter_beta").AttrType(REQUIRED).Float();
        this->Attr("gather_intra_alpha").AttrType(REQUIRED).Float();
        this->Attr("gather_intra_beta").AttrType(REQUIRED).Float();
        this->Attr("gather_inter_alpha").AttrType(REQUIRED).Float();
        this->Attr("gather_inter_beta").AttrType(REQUIRED).Float();
        this->Attr("scatter_intra_alpha").AttrType(REQUIRED).Float();
        this->Attr("scatter_intra_beta").AttrType(REQUIRED).Float();
        this->Attr("scatter_inter_alpha").AttrType(REQUIRED).Float();
        this->Attr("scatter_inter_beta").AttrType(REQUIRED).Float();
        this->Attr("runtime_cost_scale").AttrType(REQUIRED).Float();
        this->Attr("choose_min_dimension").AttrType(REQUIRED).Bool();
        this->AICore().SetTiling(optiling::SwapSelectTiling).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeSwapSelect);
} // namespace ops
