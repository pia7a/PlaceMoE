#include <algorithm>

#include "cover_score_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeCoverScoreTilingData>();
    const auto selectedShape = context->GetInputShape(0)->GetOriginShape();
    const auto routeShape = context->GetInputShape(1)->GetOriginShape();
    const auto groupShape = context->GetInputShape(6)->GetOriginShape();
    const auto copyShape = context->GetInputShape(7)->GetOriginShape();
    const auto candidateShape = context->GetInputShape(8)->GetOriginShape();
    const auto *attrs = context->GetAttrs();

    tiling->numCandidates = static_cast<uint32_t>(candidateShape.GetDim(0));
    tiling->tokenWidth = static_cast<uint32_t>(routeShape.GetDim(1));
    tiling->numTokens = static_cast<uint32_t>(selectedShape.GetDim(0));
    tiling->copyWidth = static_cast<uint32_t>(copyShape.GetDim(1));
    tiling->numSlots = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    tiling->slotsPerRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(1));
    tiling->epSize = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2));
    tiling->sourceRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));
    tiling->numLevels = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4));
    tiling->levelSize0 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
    tiling->levelSize1 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(6));
    tiling->levelSize2 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(7));
    tiling->topK = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(8));
    tiling->totalGroups = static_cast<uint32_t>(groupShape.GetDim(1));
    tiling->levelGroups0 = tiling->numLevels > 0 ? tiling->epSize / tiling->levelSize0 : 0;
    tiling->levelGroups1 = tiling->numLevels > 1 ? tiling->epSize / tiling->levelSize1 : 0;
    tiling->levelGroups2 = tiling->numLevels > 2 ? tiling->epSize / tiling->levelSize2 : 0;
    tiling->levelOffset0 = 0;
    tiling->levelOffset1 = tiling->levelGroups0;
    tiling->levelOffset2 = tiling->levelGroups0 + tiling->levelGroups1;
    tiling->outputWidth = (tiling->totalGroups + 7U) / 8U * 8U;
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(std::min(tiling->numCandidates, platform.GetCoreNumAiv()));
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeCoverScore : public OpDef {
public:
    explicit HiermoeCoverScore(const char *name) : OpDef(name)
    {
        this->Input("selected").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("route_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("multiplicities").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("token_counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("route_ranks").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("route_hashes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("token_group_counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("copy_slots").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("candidate_rows").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("deltas").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("num_slots").AttrType(REQUIRED).Int();
        this->Attr("slots_per_rank").AttrType(REQUIRED).Int();
        this->Attr("ep_size").AttrType(REQUIRED).Int();
        this->Attr("source_rank").AttrType(REQUIRED).Int();
        this->Attr("num_levels").AttrType(REQUIRED).Int();
        this->Attr("level_size0").AttrType(REQUIRED).Int();
        this->Attr("level_size1").AttrType(REQUIRED).Int();
        this->Attr("level_size2").AttrType(REQUIRED).Int();
        this->Attr("top_k").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeCoverScore);
} // namespace ops
