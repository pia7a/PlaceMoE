#include <algorithm>

#include "replica_score_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeReplicaScoreTilingData>();
    const auto routeShape = context->GetInputShape(0)->GetOriginShape();
    const auto scoreShape = context->GetInputShape(4)->GetOriginShape();
    const auto groupShape = context->GetInputShape(9)->GetOriginShape();
    const auto *attrs = context->GetAttrs();

    const auto candidateShape = context->GetInputShape(10)->GetOriginShape();
    tiling->numCandidateExperts = static_cast<uint32_t>(candidateShape.GetDim(0));
    tiling->tokenWidth = static_cast<uint32_t>(routeShape.GetDim(1));
    tiling->epSize = static_cast<uint32_t>(scoreShape.GetDim(1));
    tiling->totalGroups = static_cast<uint32_t>(groupShape.GetDim(1));
    tiling->numLevels = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    tiling->levelSize0 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(1));
    tiling->levelSize1 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2));
    tiling->levelSize2 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));
    tiling->topK = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4));
    tiling->levelGroups0 = tiling->numLevels > 0 ? tiling->epSize / tiling->levelSize0 : 0;
    tiling->levelGroups1 = tiling->numLevels > 1 ? tiling->epSize / tiling->levelSize1 : 0;
    tiling->levelGroups2 = tiling->numLevels > 2 ? tiling->epSize / tiling->levelSize2 : 0;
    tiling->levelOffset0 = 0;
    tiling->levelOffset1 = tiling->levelGroups0;
    tiling->levelOffset2 = tiling->levelGroups0 + tiling->levelGroups1;
    const uint32_t rawWidth = tiling->totalGroups + tiling->epSize;
    tiling->outputWidth = (rawWidth + 7U) / 8U * 8U;
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t numPairs = tiling->numCandidateExperts * tiling->epSize;
    context->SetBlockDim(std::min(numPairs, platform.GetCoreNumAiv()));
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeReplicaScore : public OpDef {
public:
    explicit HiermoeReplicaScore(const char *name) : OpDef(name)
    {
        this->Input("route_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("multiplicities").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("token_counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("route_ranks").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("route_scores").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("minimum_scores").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("tie_count").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("tied_rank_order").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("route_hashes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("token_group_counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("candidate_experts").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("deltas").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("num_levels").AttrType(REQUIRED).Int();
        this->Attr("level_size0").AttrType(REQUIRED).Int();
        this->Attr("level_size1").AttrType(REQUIRED).Int();
        this->Attr("level_size2").AttrType(REQUIRED).Int();
        this->Attr("top_k").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeReplicaScore);
} // namespace ops
