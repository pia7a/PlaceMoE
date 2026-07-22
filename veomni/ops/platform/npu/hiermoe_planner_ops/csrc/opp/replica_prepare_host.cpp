#include <algorithm>

#include "register/op_def_registry.h"
#include "replica_prepare_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus ReplicaPrepareTiling(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeReplicaPrepareTilingData>();
    const auto selectedShape = context->GetInputShape(0)->GetOriginShape();
    const auto *attrs = context->GetAttrs();
    tiling->numTokens = static_cast<uint32_t>(selectedShape.GetDim(0));
    tiling->tokenWidth = (tiling->numTokens + 7U) / 8U * 8U;
    tiling->topK = static_cast<uint32_t>(selectedShape.GetDim(1));
    tiling->numExperts = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(std::min(tiling->numExperts, platform.GetCoreNumAiv()));
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeReplicaPrepare : public OpDef {
public:
    explicit HiermoeReplicaPrepare(const char *name) : OpDef(name)
    {
        this->Input("selected").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("route_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("multiplicities").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("token_counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("num_experts").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::ReplicaPrepareTiling).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeReplicaPrepare);
} // namespace ops
