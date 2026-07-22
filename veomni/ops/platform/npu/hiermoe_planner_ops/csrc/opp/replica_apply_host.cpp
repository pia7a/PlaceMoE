#include "register/op_def_registry.h"
#include "replica_apply_tiling.h"

namespace optiling {
static ge::graphStatus ReplicaApplyTiling(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeReplicaApplyTilingData>();
    const auto routeShape = context->GetInputShape(0)->GetOriginShape();
    const auto scoreShape = context->GetInputShape(4)->GetOriginShape();
    const auto groupShape = context->GetInputShape(9)->GetOriginShape();
    const auto *attrs = context->GetAttrs();
    tiling->tokenWidth = static_cast<uint32_t>(routeShape.GetDim(1));
    tiling->numExperts = static_cast<uint32_t>(routeShape.GetDim(0));
    tiling->numRoutes = static_cast<uint32_t>(scoreShape.GetDim(0));
    tiling->numTokens = static_cast<uint32_t>(groupShape.GetDim(0));
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
    // Scalar GM writes use per-core caches. A single core avoids false sharing
    // when neighboring routes occupy the same cache line.
    context->SetBlockDim(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeReplicaApply : public OpDef {
public:
    explicit HiermoeReplicaApply(const char *name) : OpDef(name)
    {
        this->Input("route_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("token_counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("flat_logical").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("route_ranks").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("route_scores").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("minimum_scores").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("tie_count").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("tied_rank_order").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("route_hashes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("token_group_counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("logical_expert").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("destination_rank").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Attr("num_levels").AttrType(REQUIRED).Int();
        this->Attr("level_size0").AttrType(REQUIRED).Int();
        this->Attr("level_size1").AttrType(REQUIRED).Int();
        this->Attr("level_size2").AttrType(REQUIRED).Int();
        this->Attr("top_k").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::ReplicaApplyTiling).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeReplicaApply);
} // namespace ops
