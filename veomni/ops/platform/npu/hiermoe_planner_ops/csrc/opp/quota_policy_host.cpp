#include "quota_policy_tiling.h"
#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus QuotaPolicyTiling(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeQuotaPolicyTilingData>();
    const auto routeShape = context->GetInputShape(0)->GetOriginShape();
    const auto assignmentShape = context->GetInputShape(4)->GetOriginShape();
    const auto layoutShape = context->GetInputShape(5)->GetOriginShape();
    const auto *attrs = context->GetAttrs();
    tiling->numSamples = static_cast<uint32_t>(routeShape.GetDim(0));
    tiling->topK = static_cast<uint32_t>(routeShape.GetDim(1));
    tiling->epSize = static_cast<uint32_t>(assignmentShape.GetDim(0));
    tiling->numExperts = static_cast<uint32_t>(assignmentShape.GetDim(1));
    tiling->numSlots = static_cast<uint32_t>(layoutShape.GetDim(1));
    tiling->slotsPerRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    tiling->sourceRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(1));
    tiling->maxCopies = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));
    tiling->maskCount = tiling->maxCopies <= 8U ? 1U << tiling->maxCopies : 0U;
    tiling->samplesPerSource = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4));
    tiling->rowCapacity = static_cast<uint32_t>(
        static_cast<uint64_t>(tiling->samplesPerSource) * tiling->topK);
    tiling->rowWidth = 3U + 2U * tiling->maxCopies;
    tiling->numLevels = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
    tiling->levelSize0 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(6));
    tiling->levelSize1 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(7));
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCores = std::max(1U, platform.GetCoreNumAiv());
    tiling->blockCount = std::min(availableCores, std::max(1U, tiling->epSize));
    context->SetBlockDim(tiling->blockCount);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeQuotaPolicy : public OpDef {
public:
    explicit HiermoeQuotaPolicy(const char *name) : OpDef(name)
    {
        this->Input("sample_routes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("sample_multiplicity").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("sample_sources").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("sample_ordinals").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("assignment_counts").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("layouts").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("owner_slots").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("quota_weights").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("quota_configured").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("compact_rows").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("row_counts").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("digest").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("int_workspace").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Attr("slots_per_rank").AttrType(REQUIRED).Int();
        this->Attr("source_rank").AttrType(REQUIRED).Int();
        this->Attr("ep_size").AttrType(REQUIRED).Int();
        this->Attr("max_copies").AttrType(REQUIRED).Int();
        this->Attr("samples_per_source").AttrType(REQUIRED).Int();
        this->Attr("num_levels").AttrType(REQUIRED).Int();
        this->Attr("level_size0").AttrType(REQUIRED).Int();
        this->Attr("level_size1").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::QuotaPolicyTiling).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeQuotaPolicy);
} // namespace ops
