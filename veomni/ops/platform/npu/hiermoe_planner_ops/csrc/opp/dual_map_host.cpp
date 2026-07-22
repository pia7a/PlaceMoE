#include <algorithm>

#include "dual_map_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus DualMapTiling(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeDualMapTilingData>();
    const auto selectedShape = context->GetInputShape(0)->GetOriginShape();
    const auto copyShape = context->GetInputShape(1)->GetOriginShape();
    const auto *attrs = context->GetAttrs();
    tiling->numTokens = static_cast<uint32_t>(selectedShape.GetDim(0));
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t maxBlocks = std::max(1U, platform.GetCoreNumAiv());
    tiling->tokensPerBlock = (tiling->numTokens + maxBlocks * 16U - 1U) / (maxBlocks * 16U) * 16U;
    tiling->topK = static_cast<uint32_t>(selectedShape.GetDim(1));
    tiling->numExperts = static_cast<uint32_t>(copyShape.GetDim(1));
    tiling->maxCopies = static_cast<uint32_t>(copyShape.GetDim(2));
    tiling->slotsPerRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    tiling->sourceRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(1));
    tiling->epSize = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2));
    tiling->numLevels = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));
    tiling->levelSize0 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4));
    tiling->levelSize1 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
    tiling->step = *attrs->GetAttrPointer<int64_t>(6);
    tiling->layerSeed = *attrs->GetAttrPointer<int64_t>(7);
    context->SetBlockDim((tiling->numTokens + tiling->tokensPerBlock - 1U) / tiling->tokensPerBlock);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeDualMap : public OpDef {
public:
    explicit HiermoeDualMap(const char *name) : OpDef(name)
    {
        this->Input("selected").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("copy_slots").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("copy_counts").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("owner_ranks").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("physical").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Attr("slots_per_rank").AttrType(REQUIRED).Int();
        this->Attr("source_rank").AttrType(REQUIRED).Int();
        this->Attr("ep_size").AttrType(REQUIRED).Int();
        this->Attr("num_levels").AttrType(REQUIRED).Int();
        this->Attr("level_size0").AttrType(REQUIRED).Int();
        this->Attr("level_size1").AttrType(REQUIRED).Int();
        this->Attr("step").AttrType(REQUIRED).Int();
        this->Attr("layer_seed").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::DualMapTiling).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeDualMap);
} // namespace ops
