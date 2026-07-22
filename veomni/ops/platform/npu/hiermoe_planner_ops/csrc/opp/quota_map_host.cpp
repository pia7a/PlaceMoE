#include <algorithm>
#include <cstdint>

#include "quota_map_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static uint32_t GreatestCommonDivisor(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0U) {
        const uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

static uint32_t LeastCommonMultiple(uint32_t lhs, uint32_t rhs)
{
    return lhs / GreatestCommonDivisor(lhs, rhs) * rhs;
}

static uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

static ge::graphStatus QuotaMapTiling(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeQuotaMapTilingData>();
    const auto selectedShape = context->GetInputShape(0)->GetOriginShape();
    const auto copyShape = context->GetInputShape(1)->GetOriginShape();
    const auto quotaShape = context->GetInputShape(4)->GetOriginShape();
    const auto *attrs = context->GetAttrs();
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    tiling->numTokens = static_cast<uint32_t>(selectedShape.GetDim(0));
    tiling->topK = static_cast<uint32_t>(selectedShape.GetDim(1));
    tiling->numExperts = static_cast<uint32_t>(copyShape.GetDim(1));
    tiling->maxCopies = static_cast<uint32_t>(copyShape.GetDim(2));
    tiling->maskCount = static_cast<uint32_t>(quotaShape.GetDim(2));
    tiling->slotsPerRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    tiling->sourceRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(1));
    tiling->epSize = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2));
    tiling->numLevels = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));
    tiling->levelSize0 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4));
    tiling->levelSize1 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
    tiling->step = *attrs->GetAttrPointer<int64_t>(6);
    tiling->layerSeed = *attrs->GetAttrPointer<int64_t>(7);
    const uint32_t physicalAlignment = 8U / GreatestCommonDivisor(8U, 2U * tiling->topK);
    const uint32_t recordAlignment = 8U / GreatestCommonDivisor(8U, tiling->topK);
    const uint32_t tokenAlignment = LeastCommonMultiple(physicalAlignment, recordAlignment);
    const uint32_t availableCores = std::min(64U, std::max(1U, platform.GetCoreNumAiv()));
    const uint32_t tokenGroups = (tiling->numTokens + tokenAlignment - 1U) / tokenAlignment;
    tiling->blockCount = std::min(availableCores, std::max(1U, tokenGroups));
    const uint32_t unalignedTokens = (tiling->numTokens + tiling->blockCount - 1U) / tiling->blockCount;
    tiling->tokensPerBlock = AlignUp(std::max(1U, unalignedTokens), tokenAlignment);
    tiling->runCapacity = tiling->tokensPerBlock * tiling->topK;
    tiling->sortStride = tiling->runCapacity * tiling->blockCount;
    const uint32_t denseBucketStride = tiling->numExperts * tiling->maskCount;
    tiling->bucketStride = AlignUp(denseBucketStride, 8U);
    tiling->groupWidth = tiling->epSize;
    if (tiling->numLevels > 0U) {
        tiling->groupWidth += tiling->epSize / tiling->levelSize0;
    }
    if (tiling->numLevels > 1U) {
        tiling->groupWidth += tiling->epSize / tiling->levelSize1;
    }
    tiling->rankStride = AlignUp(tiling->epSize, 8U);
    tiling->statsStride = AlignUp(2U * (tiling->groupWidth + tiling->epSize), 8U);
    context->SetBlockDim(tiling->blockCount);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeQuotaMap : public OpDef {
public:
    explicit HiermoeQuotaMap(const char *name) : OpDef(name)
    {
        this->Input("selected").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("copy_slots").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("copy_counts").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("owner_ranks").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("quota_weights").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("quota_configured").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("token_ordinals").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("physical").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("group_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("assignment_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("int_workspace").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Attr("slots_per_rank").AttrType(REQUIRED).Int();
        this->Attr("source_rank").AttrType(REQUIRED).Int();
        this->Attr("ep_size").AttrType(REQUIRED).Int();
        this->Attr("num_levels").AttrType(REQUIRED).Int();
        this->Attr("level_size0").AttrType(REQUIRED).Int();
        this->Attr("level_size1").AttrType(REQUIRED).Int();
        this->Attr("step").AttrType(REQUIRED).Int();
        this->Attr("layer_seed").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::QuotaMapTiling).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeQuotaMap);
} // namespace ops
