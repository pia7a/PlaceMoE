#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include "function.h"
#include "pytorch_npu_helper.hpp"

namespace {
void check_quota_policy_tensor(
    const at::Tensor &tensor,
    const char *name,
    const c10::Device &device)
{
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must have dtype int64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void checked_add(int64_t &total, int64_t value, const char *message)
{
    TORCH_CHECK(value >= 0 && total <= std::numeric_limits<int64_t>::max() - value, message);
    total += value;
}

int64_t checked_product(int64_t left, int64_t right, const char *message)
{
    TORCH_CHECK(
        left >= 0 && right >= 0
            && (left == 0 || right <= std::numeric_limits<int64_t>::max() / left),
        message);
    return left * right;
}

int64_t checked_align_up(int64_t value, int64_t alignment, const char *message)
{
    TORCH_CHECK(value >= 0 && alignment > 0, message);
    const int64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    TORCH_CHECK(value <= std::numeric_limits<int64_t>::max() - (alignment - remainder), message);
    return value + alignment - remainder;
}

void checked_append_aligned(int64_t &total, int64_t elements, const char *message)
{
    total = checked_align_up(total, 8, message);
    checked_add(total, elements, message);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hiermoe_quota_policy_npu(
    const at::Tensor &sample_routes,
    const at::Tensor &sample_multiplicity,
    const at::Tensor &sample_sources,
    const at::Tensor &sample_ordinals,
    const at::Tensor &assignment_counts,
    const at::Tensor &layouts,
    const at::Tensor &owner_slots,
    int64_t slots_per_rank,
    int64_t source_rank,
    int64_t ep_size,
    int64_t max_copies,
    int64_t samples_per_source,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1)
{
    const auto device = sample_routes.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "quota_policy requires NPU tensors");
    check_quota_policy_tensor(sample_routes, "sample_routes", device);
    check_quota_policy_tensor(sample_multiplicity, "sample_multiplicity", device);
    check_quota_policy_tensor(sample_sources, "sample_sources", device);
    check_quota_policy_tensor(sample_ordinals, "sample_ordinals", device);
    check_quota_policy_tensor(assignment_counts, "assignment_counts", device);
    check_quota_policy_tensor(layouts, "layouts", device);
    check_quota_policy_tensor(owner_slots, "owner_slots", device);

    constexpr int64_t max_uint32 = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(sample_routes.dim() == 2, "sample_routes must have shape [samples, top_k]");
    TORCH_CHECK(
        sample_routes.size(0) <= max_uint32 && sample_routes.size(1) > 0 && sample_routes.size(1) <= 16,
        "quota_policy supports uint32 sample counts and top-k widths from one through sixteen");
    TORCH_CHECK(sample_multiplicity.sizes() == sample_routes.sizes(), "sample_multiplicity must match sample_routes");
    const int64_t num_samples = sample_routes.size(0);
    TORCH_CHECK(
        sample_sources.dim() == 1 && sample_sources.numel() == num_samples,
        "sample_sources must contain one value per sample");
    TORCH_CHECK(
        sample_ordinals.dim() == 1 && sample_ordinals.numel() == num_samples,
        "sample_ordinals must contain one value per sample");

    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "quota_policy supports EP sizes from one through 64");
    TORCH_CHECK(source_rank >= 0 && source_rank < ep_size, "source_rank is outside the EP group");
    TORCH_CHECK(slots_per_rank > 0 && slots_per_rank <= max_uint32, "slots_per_rank is outside the tiling range");
    TORCH_CHECK(max_copies > 0 && max_copies <= 8, "quota_policy supports one through eight copies per expert");
    TORCH_CHECK(samples_per_source >= 0 && samples_per_source <= max_uint32, "samples_per_source is outside the tiling range");
    TORCH_CHECK(
        num_samples <= checked_product(ep_size, samples_per_source, "quota_policy sample capacity overflow"),
        "sample_routes exceeds the fixed global sample capacity");

    TORCH_CHECK(
        assignment_counts.dim() == 2 && assignment_counts.size(0) == ep_size,
        "assignment_counts must have shape [ep_size, experts]");
    const int64_t num_experts = assignment_counts.size(1);
    TORCH_CHECK(num_experts > 0 && num_experts <= 256, "quota_policy supports one through 256 logical experts");
    const int64_t num_slots = checked_product(ep_size, slots_per_rank, "quota_policy slot count overflow");
    TORCH_CHECK(num_slots <= max_uint32, "quota_policy slot count exceeds the tiling range");
    TORCH_CHECK(
        layouts.dim() == 2 && layouts.size(0) == 2 && layouts.size(1) == num_slots,
        "layouts must have shape [2, ep_size * slots_per_rank]");
    TORCH_CHECK(
        owner_slots.dim() == 2 && owner_slots.size(0) == 2 && owner_slots.size(1) == num_experts,
        "owner_slots must have shape [2, experts]");

    TORCH_CHECK(num_levels >= 0 && num_levels <= 2, "quota_policy supports zero through two hierarchy levels");
    const int64_t level_sizes[] = {level_size0, level_size1};
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(
            level_sizes[level] > 0 && level_sizes[level] <= max_uint32 && ep_size % level_sizes[level] == 0,
            "invalid quota_policy hierarchy level");
    }

    const int64_t mask_count = 1LL << max_copies;
    const int64_t row_capacity = checked_product(samples_per_source, sample_routes.size(1), "quota_policy row capacity overflow");
    TORCH_CHECK(row_capacity <= max_uint32, "quota_policy row capacity exceeds the tiling range");
    const int64_t row_width = 3 + 2 * max_copies;
    const int64_t route_capacity = checked_product(ep_size, row_capacity, "quota_policy route capacity overflow");
    TORCH_CHECK(route_capacity <= (1LL << 28), "quota_policy sample workspace is too large");

    const int64_t expert_copy_elements = checked_product(num_experts, max_copies, "quota_policy workspace size overflow");
    const int64_t source_expert_stride = checked_align_up(num_experts, 8, "quota_policy workspace size overflow");
    const int64_t source_expert_elements = checked_product(ep_size, source_expert_stride, "quota_policy workspace size overflow");
    const int64_t source_bucket_stride = checked_align_up(
        checked_product(num_experts, mask_count, "quota_policy workspace size overflow"),
        8,
        "quota_policy workspace size overflow");
    const int64_t bucket_elements = checked_product(ep_size, source_bucket_stride, "quota_policy workspace size overflow");
    const int64_t order_stride = checked_align_up(row_capacity, 8, "quota_policy workspace size overflow");
    const int64_t order_elements = checked_product(ep_size, order_stride, "quota_policy workspace size overflow");
    const int64_t rank_stride = checked_align_up(ep_size, 8, "quota_policy workspace size overflow");
    const int64_t source_rank_elements = checked_product(ep_size, rank_stride, "quota_policy workspace size overflow");
    const int64_t source_control_elements = checked_product(ep_size, 8, "quota_policy workspace size overflow");
    int64_t workspace_elements = 0;
    checked_append_aligned(workspace_elements, expert_copy_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, num_experts, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, num_experts, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, num_experts, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, bucket_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, bucket_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, bucket_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, source_expert_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, source_expert_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, order_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, order_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, source_rank_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, rank_stride, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, source_control_elements, "quota_policy workspace size overflow");
    checked_append_aligned(workspace_elements, source_control_elements, "quota_policy workspace size overflow");

    auto quota_weights = at::empty({2, num_experts, mask_count, max_copies}, sample_routes.options());
    auto quota_configured = at::empty({2, num_experts, mask_count}, sample_routes.options());
    auto compact_rows = at::empty({2, row_capacity, row_width}, sample_routes.options());
    auto row_counts = at::empty({2}, sample_routes.options());
    auto digest = at::empty({2, 2}, sample_routes.options());
    auto int_workspace = at::empty({workspace_elements}, sample_routes.options());

    EXEC_NPU_CMD(
        aclnnHiermoeQuotaPolicy,
        sample_routes,
        sample_multiplicity,
        sample_sources,
        sample_ordinals,
        assignment_counts,
        layouts,
        owner_slots,
        slots_per_rank,
        source_rank,
        ep_size,
        max_copies,
        samples_per_source,
        num_levels,
        level_size0,
        level_size1,
        quota_weights,
        quota_configured,
        compact_rows,
        row_counts,
        digest,
        int_workspace);
    return {quota_weights, quota_configured, compact_rows, row_counts, digest};
}
