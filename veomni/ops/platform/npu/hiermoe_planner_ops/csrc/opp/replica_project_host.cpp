#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "replica_project_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static uint32_t Align16(uint32_t value)
{
    return (value + 15U) / 16U * 16U;
}

static bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0ULL && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool Align16Checked(uint64_t value, uint64_t &result)
{
    uint64_t padded = 0ULL;
    if (!CheckedAdd(value, 15ULL, padded)) {
        return false;
    }
    result = padded / 16ULL * 16ULL;
    return true;
}

static ge::graphStatus ReplicaProjectTiling(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<HiermoeReplicaProjectTilingData>();
    const auto routeShape = context->GetInputShape(0)->GetOriginShape();
    const auto assignmentShape = context->GetInputShape(5)->GetOriginShape();
    const auto layoutShape = context->GetInputShape(7)->GetOriginShape();
    const auto redundantShape = context->GetInputShape(9)->GetOriginShape();
    const auto *attrs = context->GetAttrs();
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    tiling->numSamples = static_cast<uint32_t>(routeShape.GetDim(0));
    tiling->topK = static_cast<uint32_t>(routeShape.GetDim(1));
    tiling->epSize = static_cast<uint32_t>(assignmentShape.GetDim(0));
    tiling->numExperts = static_cast<uint32_t>(assignmentShape.GetDim(1));
    tiling->numSlots = static_cast<uint32_t>(layoutShape.GetDim(0));
    tiling->redundantSlotsPerRank = static_cast<uint32_t>(redundantShape.GetDim(1));
    tiling->slotsPerRank = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0));
    tiling->numLevels = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2));
    tiling->levelSize0 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));
    tiling->levelSize1 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4));
    tiling->levelSize2 = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
    tiling->step = *attrs->GetAttrPointer<int64_t>(6);
    tiling->layerSeed = *attrs->GetAttrPointer<int64_t>(7);
    const uint64_t maxRecords = static_cast<uint64_t>(tiling->numSamples) * tiling->topK;
    const uint64_t maxUint32 = std::numeric_limits<uint32_t>::max();
    const uint64_t maxUint64 = std::numeric_limits<uint64_t>::max();
    if (maxRecords > maxUint32) {
        return ge::GRAPH_FAILED;
    }
    tiling->maxRecords = static_cast<uint32_t>(maxRecords);
    uint64_t sharedRecordCapacity = 0ULL;
    uint64_t logicalPadding = 0ULL;
    if (!CheckedMul(7ULL, tiling->numExperts, logicalPadding)
        || !CheckedAdd(maxRecords, logicalPadding, sharedRecordCapacity)
        || sharedRecordCapacity > maxUint32) {
        return ge::GRAPH_FAILED;
    }
    tiling->sharedRecordCapacity = static_cast<uint32_t>(sharedRecordCapacity);
    const uint64_t epSize = tiling->epSize;
    const uint64_t distributionExperts = std::max(2U, tiling->numExperts);
    if (epSize != 0ULL && distributionExperts > maxUint64 / epSize / epSize) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t distributionSize = epSize * distributionExperts * epSize;
    if (distributionSize > maxUint32) {
        return ge::GRAPH_FAILED;
    }
    tiling->distributionSize = static_cast<uint32_t>(distributionSize);
    // A canonical sampled route contains at most one positive-multiplicity
    // record for a given (token, logical expert).  Replica edges only rescore
    // one logical expert, so their private record domain is bounded by the
    // number of sampled tokens rather than by samples * top-k.
    tiling->actionMaxRecords = tiling->numSamples;
    const uint64_t actionDistributionSize = 2ULL * epSize * epSize;
    if (actionDistributionSize > maxUint32) {
        return ge::GRAPH_FAILED;
    }
    tiling->actionDistributionSize = static_cast<uint32_t>(actionDistributionSize);

    uint64_t hashCapacity = 2U;
    const uint64_t desired = std::max<uint64_t>(2U, 2ULL * maxRecords);
    while (hashCapacity < desired) {
        hashCapacity <<= 1U;
    }
    if (hashCapacity > maxUint32) {
        return ge::GRAPH_FAILED;
    }
    tiling->hashCapacity = static_cast<uint32_t>(hashCapacity);
    uint64_t actionHashCapacity = 2U;
    const uint64_t actionDesired = std::max<uint64_t>(2U, 2ULL * tiling->actionMaxRecords);
    while (actionHashCapacity < actionDesired) {
        actionHashCapacity <<= 1U;
    }
    if (actionHashCapacity > maxUint32) {
        return ge::GRAPH_FAILED;
    }
    tiling->actionHashCapacity = static_cast<uint32_t>(actionHashCapacity);
    const uint64_t addActions = epSize * tiling->numExperts;
    const uint64_t removeActions = epSize * tiling->redundantSlotsPerRank;
    const uint64_t maxActions = std::max(addActions, removeActions);
    const uint32_t availableCores = std::min(64U, std::max(1U, platform.GetCoreNumAiv()));
    tiling->blockCount = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(1ULL, maxActions), availableCores));
    uint64_t directRawRows = 0ULL;
    uint64_t directRawPadding = 0ULL;
    uint64_t directRawRecordCapacity = 0ULL;
    if (!CheckedMul(tiling->numExperts, tiling->blockCount, directRawRows)
        || !CheckedMul(7ULL, directRawRows, directRawPadding)
        || !CheckedAdd(maxRecords, directRawPadding, directRawRecordCapacity)
        || directRawRecordCapacity > maxUint32) {
        return ge::GRAPH_FAILED;
    }
    tiling->directRawRecordCapacity = static_cast<uint32_t>(directRawRecordCapacity);

    const uint32_t levelSizes[] = {tiling->levelSize0, tiling->levelSize1, tiling->levelSize2};
    uint32_t totalGroups = 0;
    uint32_t offset = 0;
    uint32_t *levelGroups[] = {&tiling->levelGroups0, &tiling->levelGroups1, &tiling->levelGroups2};
    uint32_t *levelOffsets[] = {&tiling->levelOffset0, &tiling->levelOffset1, &tiling->levelOffset2};
    for (uint32_t level = 0; level < 3U; ++level) {
        *levelOffsets[level] = offset;
        *levelGroups[level] = level < tiling->numLevels ? tiling->epSize / levelSizes[level] : 0U;
        totalGroups += *levelGroups[level];
        offset += *levelGroups[level];
    }
    tiling->totalGroups = totalGroups;
    tiling->groupOutputStride = Align16(totalGroups);
    tiling->assignmentOutputStride = Align16(tiling->epSize);
    const uint64_t baselineIntWorkspace = 8ULL * maxRecords + 3ULL * hashCapacity
        + distributionSize + tiling->numSamples + 1ULL;
    const uint64_t actionIntWorkspace = 8ULL * tiling->actionMaxRecords + 3ULL * actionHashCapacity
        + actionDistributionSize + tiling->numSamples + 1ULL;
    const uint64_t rankStride = Align16(tiling->epSize);
    uint64_t genericBaselineFloatWorkspace = 0ULL;
    uint64_t blockRankLoads = 0ULL;
    uint64_t expertLoads = 0ULL;
    uint64_t directBaselineFloatWorkspace = 0ULL;
    if (!CheckedAdd(rankStride, maxRecords, genericBaselineFloatWorkspace)
        || !CheckedMul(tiling->blockCount, rankStride, blockRankLoads)
        || !CheckedMul(16ULL, tiling->numExperts, expertLoads)
        || !CheckedAdd(rankStride, blockRankLoads, directBaselineFloatWorkspace)
        || !CheckedAdd(directBaselineFloatWorkspace, expertLoads, directBaselineFloatWorkspace)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t baselineFloatWorkspace = std::max(
        genericBaselineFloatWorkspace,
        directBaselineFloatWorkspace);
    const uint64_t actionFloatWorkspace = static_cast<uint64_t>(Align16(tiling->epSize))
        + tiling->actionMaxRecords;
    const uint64_t baselineIntWorkspaceStride = (baselineIntWorkspace + 15ULL) / 16ULL * 16ULL;
    const uint64_t actionIntWorkspaceStride = (actionIntWorkspace + 15ULL) / 16ULL * 16ULL;
    const uint64_t baselineFloatWorkspaceStride = (baselineFloatWorkspace + 15ULL) / 16ULL * 16ULL;
    const uint64_t actionFloatWorkspaceStride = (actionFloatWorkspace + 15ULL) / 16ULL * 16ULL;
    const uint64_t directBlockLogicalStride = Align16(tiling->numExperts);
    uint64_t directBlockCountStride = 0ULL;
    const uint64_t directBlockLoadStride = Align16(tiling->numExperts);
    if (!CheckedMul(3ULL, directBlockLogicalStride, directBlockCountStride)
        || baselineIntWorkspaceStride > maxUint32 || actionIntWorkspaceStride > maxUint32
        || baselineFloatWorkspaceStride > maxUint32 || actionFloatWorkspaceStride > maxUint32
        || directBlockCountStride > maxUint32 || directBlockLoadStride > maxUint32) {
        return ge::GRAPH_FAILED;
    }
    tiling->baselineIntWorkspaceStride = static_cast<uint32_t>(baselineIntWorkspaceStride);
    tiling->actionIntWorkspaceStride = static_cast<uint32_t>(actionIntWorkspaceStride);
    tiling->baselineFloatWorkspaceStride = static_cast<uint32_t>(baselineFloatWorkspaceStride);
    tiling->actionFloatWorkspaceStride = static_cast<uint32_t>(actionFloatWorkspaceStride);
    tiling->directBlockCountStride = static_cast<uint32_t>(directBlockCountStride);
    tiling->directBlockLoadStride = static_cast<uint32_t>(directBlockLoadStride);
    const uint64_t actionWorkspaceBlocks = static_cast<uint64_t>(tiling->blockCount);
    if (actionIntWorkspaceStride > (maxUint64 - baselineIntWorkspaceStride) / actionWorkspaceBlocks
        || actionFloatWorkspaceStride > (maxUint64 - baselineFloatWorkspaceStride) / actionWorkspaceBlocks
        || directBlockCountStride > maxUint64 / actionWorkspaceBlocks
        || directBlockLoadStride > maxUint64 / actionWorkspaceBlocks) {
        return ge::GRAPH_FAILED;
    }
    tiling->actionIntWorkspaceOffset = baselineIntWorkspaceStride;
    tiling->actionFloatWorkspaceOffset = baselineFloatWorkspaceStride;
    const uint64_t actionIntWorkspaceEnd = baselineIntWorkspaceStride
        + actionWorkspaceBlocks * actionIntWorkspaceStride;
    const uint64_t actionFloatWorkspaceEnd = baselineFloatWorkspaceStride
        + actionWorkspaceBlocks * actionFloatWorkspaceStride;
    const uint64_t directBlockCountElements = actionWorkspaceBlocks * directBlockCountStride;
    const uint64_t directBlockLoadElements = actionWorkspaceBlocks * directBlockLoadStride;
    if (actionIntWorkspaceEnd > maxUint64 - directBlockCountElements
        || actionFloatWorkspaceEnd > maxUint64 - directBlockLoadElements) {
        return ge::GRAPH_FAILED;
    }
    tiling->directBlockCountOffset = actionIntWorkspaceEnd;
    tiling->directBlockLoadOffset = actionFloatWorkspaceEnd;
    tiling->sharedSummaryOffset = actionIntWorkspaceEnd + directBlockCountElements;
    uint64_t sharedCursor = tiling->numExperts;
    if (!Align16Checked(sharedCursor, sharedCursor)
        || !CheckedAdd(sharedCursor, 16ULL, sharedCursor)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tokenCoverageElements = 0ULL;
    if (!CheckedMul(tiling->numSamples, 8ULL, tokenCoverageElements)
        || !CheckedAdd(sharedCursor, tokenCoverageElements, sharedCursor)
        || !Align16Checked(sharedCursor, sharedCursor)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t logicalMetadataElements = 0ULL;
    uint64_t allDestinationMaskStride = 0ULL;
    uint64_t allDestinationMaskElements = 0ULL;
    if (!CheckedAdd(tiling->numExperts, 1ULL, logicalMetadataElements)
        || !CheckedMul(logicalMetadataElements, 8ULL, logicalMetadataElements)
        || !Align16Checked(sharedRecordCapacity, allDestinationMaskStride)
        || !CheckedMul(2ULL, allDestinationMaskStride, allDestinationMaskElements)
        || !CheckedAdd(sharedCursor, logicalMetadataElements, sharedCursor)
        || !Align16Checked(sharedCursor, sharedCursor)
        || !CheckedAdd(sharedCursor, directRawRecordCapacity, sharedCursor)
        || !Align16Checked(sharedCursor, sharedCursor)
        || !CheckedAdd(sharedCursor, directRawRecordCapacity, sharedCursor)
        || !Align16Checked(sharedCursor, sharedCursor)
        || !CheckedAdd(sharedCursor, sharedRecordCapacity, sharedCursor)
        || !Align16Checked(sharedCursor, sharedCursor)
        || !CheckedAdd(sharedCursor, sharedRecordCapacity, sharedCursor)
        || !Align16Checked(sharedCursor, sharedCursor)
        || !CheckedAdd(sharedCursor, allDestinationMaskElements, sharedCursor)) {
        return ge::GRAPH_FAILED;
    }
    tiling->sharedSummaryElements = sharedCursor;
    if (tiling->sharedSummaryOffset > maxUint64 - tiling->sharedSummaryElements) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(tiling->blockCount);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class HiermoeReplicaProject : public OpDef {
public:
    explicit HiermoeReplicaProject(const char *name) : OpDef(name)
    {
        this->Input("sample_routes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("sample_multiplicity").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("sample_weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("sample_sources").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("sample_ordinals").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("assignment_counts").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("seed_base_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("slot_to_logical").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("owner_slots").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("redundant_slots").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("candidate_experts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("base_counts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("assignment_loads").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("add_group_deltas").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("add_assignment_deltas").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("remove_group_deltas").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("remove_assignment_deltas").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("int_workspace").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("float_workspace").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("slots_per_rank").AttrType(REQUIRED).Int();
        this->Attr("ep_size").AttrType(REQUIRED).Int();
        this->Attr("num_levels").AttrType(REQUIRED).Int();
        this->Attr("level_size0").AttrType(REQUIRED).Int();
        this->Attr("level_size1").AttrType(REQUIRED).Int();
        this->Attr("level_size2").AttrType(REQUIRED).Int();
        this->Attr("step").AttrType(REQUIRED).Int();
        this->Attr("layer_seed").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::ReplicaProjectTiling).AddConfig("ascend910b");
    }
};
OP_ADD(HiermoeReplicaProject);
} // namespace ops
