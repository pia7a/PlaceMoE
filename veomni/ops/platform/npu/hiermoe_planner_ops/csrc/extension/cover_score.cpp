#include "function.h"
#include "pytorch_npu_helper.hpp"

#include <cstdint>

namespace {
void check_cover_tensor(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    const c10::Device &device)
{
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}
} // namespace

at::Tensor hiermoe_cover_score_npu(
    const at::Tensor &selected,
    const at::Tensor &route_indices,
    const at::Tensor &multiplicities,
    const at::Tensor &token_counts,
    const at::Tensor &route_ranks,
    const at::Tensor &route_hashes,
    const at::Tensor &token_group_counts,
    const at::Tensor &copy_slots,
    const at::Tensor &candidate_rows,
    int64_t num_slots,
    int64_t slots_per_rank,
    int64_t ep_size,
    int64_t source_rank,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t level_size2,
    int64_t top_k)
{
    const auto device = selected.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "cover_score requires NPU tensors");
    check_cover_tensor(selected, "selected", at::kLong, device);
    check_cover_tensor(route_indices, "route_indices", at::kInt, device);
    check_cover_tensor(multiplicities, "multiplicities", at::kInt, device);
    check_cover_tensor(token_counts, "token_counts", at::kInt, device);
    check_cover_tensor(route_ranks, "route_ranks", at::kLong, device);
    check_cover_tensor(route_hashes, "route_hashes", at::kLong, device);
    check_cover_tensor(token_group_counts, "token_group_counts", at::kInt, device);
    check_cover_tensor(copy_slots, "copy_slots", at::kLong, device);
    check_cover_tensor(candidate_rows, "candidate_rows", at::kLong, device);
    TORCH_CHECK(selected.dim() == 2, "selected must be rank 2");
    TORCH_CHECK(route_indices.dim() == 2, "route_indices must be rank 2");
    TORCH_CHECK(multiplicities.sizes() == route_indices.sizes(), "packed route tables disagree");
    TORCH_CHECK(token_counts.dim() == 1, "token_counts must be rank 1");
    TORCH_CHECK(route_indices.size(0) == token_counts.numel(), "expert tables disagree");
    TORCH_CHECK(route_ranks.numel() == selected.numel(), "route_ranks must match selected");
    TORCH_CHECK(route_hashes.numel() == selected.numel(), "route_hashes must match selected");
    TORCH_CHECK(token_group_counts.dim() == 2, "token_group_counts must be rank 2");
    TORCH_CHECK(token_group_counts.size(0) == selected.size(0), "token counts must match selected");
    TORCH_CHECK(copy_slots.dim() == 2 && copy_slots.size(0) == token_counts.numel(), "invalid copy_slots");
    TORCH_CHECK(candidate_rows.dim() == 2 && candidate_rows.size(1) == 5, "candidate_rows must be [N, 5]");
    TORCH_CHECK(num_slots > 0 && slots_per_rank > 0, "slot dimensions must be positive");
    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "cover_score supports EP sizes from one through 64");
    TORCH_CHECK(source_rank >= 0 && source_rank < ep_size, "source_rank is outside the EP group");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 3, "cover_score supports one through three levels");
    TORCH_CHECK(top_k == selected.size(1), "top_k must match selected");
    const int64_t level_sizes[] = {level_size0, level_size1, level_size2};
    int64_t total_groups = 0;
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(level_sizes[level] > 0 && ep_size % level_sizes[level] == 0, "invalid hierarchy level");
        total_groups += ep_size / level_sizes[level];
    }
    TORCH_CHECK(token_group_counts.size(1) == total_groups, "token_group_counts disagrees with hierarchy");
    const int64_t output_width = (total_groups + 7) / 8 * 8;
    auto output = at::empty({candidate_rows.size(0), output_width}, route_indices.options());
    EXEC_NPU_CMD(
        aclnnHiermoeCoverScore,
        selected,
        route_indices,
        multiplicities,
        token_counts,
        route_ranks,
        route_hashes,
        token_group_counts,
        copy_slots,
        candidate_rows,
        num_slots,
        slots_per_rank,
        ep_size,
        source_rank,
        num_levels,
        level_size0,
        level_size1,
        level_size2,
        top_k,
        output);
    return output;
}
