#include "function.h"
#include "pytorch_npu_helper.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

std::tuple<at::Tensor, at::Tensor, at::Tensor> hiermoe_replica_prepare_npu(
    const at::Tensor &selected,
    int64_t num_experts)
{
    TORCH_CHECK(selected.dim() == 2, "selected must be rank 2");
    TORCH_CHECK(selected.is_contiguous(), "selected must be contiguous");
    TORCH_CHECK(selected.size(0) <= 16384, "replica_prepare supports at most 16384 tokens");
    auto options = selected.options().dtype(at::kInt);
    const int64_t token_width = (selected.size(0) + 7) / 8 * 8;
    auto route_indices = at::empty({num_experts, token_width}, options);
    auto multiplicities = at::empty({num_experts, token_width}, options);
    auto token_counts = at::empty({num_experts, 8}, options);
    EXEC_NPU_CMD(
        aclnnHiermoeReplicaPrepare,
        selected,
        num_experts,
        route_indices,
        multiplicities,
        token_counts);
    return {route_indices, multiplicities, token_counts.select(1, 0).contiguous()};
}

at::Tensor hiermoe_replica_score_npu(
    const at::Tensor &route_indices,
    const at::Tensor &multiplicities,
    const at::Tensor &token_counts,
    const at::Tensor &route_ranks,
    const at::Tensor &route_scores,
    const at::Tensor &minimum_scores,
    const at::Tensor &tie_count,
    const at::Tensor &tied_rank_order,
    const at::Tensor &route_hashes,
    const at::Tensor &token_group_counts,
    const at::Tensor &candidate_experts,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t level_size2,
    int64_t top_k)
{
    TORCH_CHECK(route_indices.dim() == 2, "route_indices must be rank 2");
    TORCH_CHECK(route_scores.dim() == 2, "route_scores must be rank 2");
    const int64_t candidates = candidate_experts.numel() * route_scores.size(1);
    const int64_t raw_width = token_group_counts.size(1) + route_scores.size(1);
    const int64_t output_width = (raw_width + 7) / 8 * 8;
    auto output = at::empty({candidates, output_width}, route_indices.options().dtype(at::kInt));
    EXEC_NPU_CMD(
        aclnnHiermoeReplicaScore,
        route_indices,
        multiplicities,
        token_counts,
        route_ranks,
        route_scores,
        minimum_scores,
        tie_count,
        tied_rank_order,
        route_hashes,
        token_group_counts,
        candidate_experts,
        num_levels,
        level_size0,
        level_size1,
        level_size2,
        top_k,
        output);
    return output;
}

namespace {
void check_apply_tensor(
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

void hiermoe_replica_apply_npu(
    const at::Tensor &route_indices,
    const at::Tensor &token_counts,
    const at::Tensor &flat_logical,
    at::Tensor &route_ranks,
    const at::Tensor &route_scores,
    at::Tensor &minimum_scores,
    at::Tensor &tie_count,
    at::Tensor &tied_rank_order,
    const at::Tensor &route_hashes,
    at::Tensor &token_group_counts,
    const at::Tensor &logical_expert,
    const at::Tensor &destination_rank,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t level_size2,
    int64_t top_k)
{
    const auto device = route_indices.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "replica_apply requires NPU tensors");
    check_apply_tensor(route_indices, "route_indices", at::kInt, device);
    check_apply_tensor(token_counts, "token_counts", at::kInt, device);
    check_apply_tensor(flat_logical, "flat_logical", at::kLong, device);
    check_apply_tensor(route_ranks, "route_ranks", at::kLong, device);
    check_apply_tensor(route_scores, "route_scores", at::kInt, device);
    check_apply_tensor(minimum_scores, "minimum_scores", at::kInt, device);
    check_apply_tensor(tie_count, "tie_count", at::kInt, device);
    check_apply_tensor(tied_rank_order, "tied_rank_order", at::kLong, device);
    check_apply_tensor(route_hashes, "route_hashes", at::kLong, device);
    check_apply_tensor(token_group_counts, "token_group_counts", at::kInt, device);
    check_apply_tensor(logical_expert, "logical_expert", at::kLong, device);
    check_apply_tensor(destination_rank, "destination_rank", at::kLong, device);
    TORCH_CHECK(route_indices.dim() == 2, "route_indices must be rank 2");
    TORCH_CHECK(token_counts.dim() == 1, "token_counts must be rank 1");
    TORCH_CHECK(flat_logical.dim() == 1, "flat_logical must be rank 1");
    TORCH_CHECK(route_ranks.dim() == 1, "route_ranks must be rank 1");
    TORCH_CHECK(route_scores.dim() == 2, "route_scores must be rank 2");
    TORCH_CHECK(minimum_scores.dim() == 1, "minimum_scores must be rank 1");
    TORCH_CHECK(tie_count.dim() == 1, "tie_count must be rank 1");
    TORCH_CHECK(tied_rank_order.dim() == 2, "tied_rank_order must be rank 2");
    TORCH_CHECK(route_hashes.dim() == 1, "route_hashes must be rank 1");
    TORCH_CHECK(token_group_counts.dim() == 2, "token_group_counts must be rank 2");
    TORCH_CHECK(logical_expert.numel() == 1, "logical_expert must contain one value");
    TORCH_CHECK(destination_rank.numel() == 1, "destination_rank must contain one value");
    const int64_t num_routes = flat_logical.numel();
    const int64_t ep_size = route_scores.size(1);
    TORCH_CHECK(top_k > 0 && num_routes % top_k == 0, "top_k must divide the route count");
    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "replica_apply supports EP sizes from 1 through 64");
    TORCH_CHECK(route_indices.size(0) == token_counts.numel(), "packed expert tables disagree");
    TORCH_CHECK(route_ranks.numel() == num_routes, "route_ranks has an invalid length");
    TORCH_CHECK(route_scores.size(0) == num_routes, "route_scores has an invalid length");
    TORCH_CHECK(minimum_scores.numel() == num_routes, "minimum_scores has an invalid length");
    TORCH_CHECK(tie_count.numel() == num_routes, "tie_count has an invalid length");
    TORCH_CHECK(tied_rank_order.sizes() == route_scores.sizes(), "tied_rank_order has an invalid shape");
    TORCH_CHECK(route_hashes.numel() == num_routes, "route_hashes has an invalid length");
    TORCH_CHECK(token_group_counts.size(0) == num_routes / top_k, "token_group_counts has an invalid length");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 3, "replica_apply supports one through three levels");
    const int64_t level_sizes[] = {level_size0, level_size1, level_size2};
    int64_t total_groups = 0;
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(level_sizes[level] > 0 && ep_size % level_sizes[level] == 0, "invalid hierarchy level size");
        total_groups += ep_size / level_sizes[level];
    }
    TORCH_CHECK(token_group_counts.size(1) == total_groups, "token_group_counts disagrees with hierarchy");
    EXEC_NPU_CMD(
        aclnnHiermoeReplicaApply,
        route_indices,
        token_counts,
        flat_logical,
        route_ranks,
        route_scores,
        minimum_scores,
        tie_count,
        tied_rank_order,
        route_hashes,
        token_group_counts,
        logical_expert,
        destination_rank,
        num_levels,
        level_size0,
        level_size1,
        level_size2,
        top_k);
}

at::Tensor hiermoe_dual_map_npu(
    const at::Tensor &selected,
    const at::Tensor &copy_slots,
    const at::Tensor &copy_counts,
    const at::Tensor &owner_ranks,
    int64_t slots_per_rank,
    int64_t source_rank,
    int64_t ep_size,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t step,
    int64_t layer_seed)
{
    const auto device = selected.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "dual_map requires NPU tensors");
    check_apply_tensor(selected, "selected", at::kLong, device);
    check_apply_tensor(copy_slots, "copy_slots", at::kLong, device);
    check_apply_tensor(copy_counts, "copy_counts", at::kLong, device);
    check_apply_tensor(owner_ranks, "owner_ranks", at::kLong, device);
    TORCH_CHECK(selected.dim() == 2, "selected must be rank 2");
    TORCH_CHECK(copy_slots.dim() == 3 && copy_slots.size(0) == 2, "copy_slots must have shape [2, E, C]");
    TORCH_CHECK(copy_counts.dim() == 2 && copy_counts.size(0) == 2, "copy_counts must have shape [2, E]");
    TORCH_CHECK(owner_ranks.sizes() == copy_counts.sizes(), "owner_ranks must match copy_counts");
    TORCH_CHECK(copy_slots.size(1) == copy_counts.size(1), "copy tables disagree on the expert count");
    TORCH_CHECK(copy_slots.size(1) > 0, "dual_map requires at least one logical expert");
    TORCH_CHECK(
        copy_slots.size(2) > 0 && copy_slots.size(2) <= 8,
        "dual_map supports one through eight copies per expert");
    TORCH_CHECK(slots_per_rank > 0, "slots_per_rank must be positive");
    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "dual_map supports EP sizes from one through 64");
    TORCH_CHECK(source_rank >= 0 && source_rank < ep_size, "source_rank is outside the EP group");
    TORCH_CHECK(num_levels >= 0 && num_levels <= 2, "dual_map supports zero through two hierarchy levels");
    const int64_t level_sizes[] = {level_size0, level_size1};
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(level_sizes[level] > 0 && ep_size % level_sizes[level] == 0, "invalid hierarchy level size");
    }
    auto output = at::empty({selected.size(0), 2, selected.size(1)}, selected.options());
    if (selected.size(0) == 0 || selected.size(1) == 0) {
        return output.permute({1, 0, 2});
    }
    EXEC_NPU_CMD(
        aclnnHiermoeDualMap,
        selected,
        copy_slots,
        copy_counts,
        owner_ranks,
        slots_per_rank,
        source_rank,
        ep_size,
        num_levels,
        level_size0,
        level_size1,
        step,
        layer_seed,
        output);
    return output.permute({1, 0, 2});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hiermoe_quota_map_npu(
    const at::Tensor &selected,
    const at::Tensor &copy_slots,
    const at::Tensor &copy_counts,
    const at::Tensor &owner_ranks,
    const at::Tensor &quota_weights,
    const at::Tensor &quota_configured,
    const at::Tensor &token_ordinals,
    int64_t slots_per_rank,
    int64_t source_rank,
    int64_t ep_size,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t step,
    int64_t layer_seed)
{
    const auto device = selected.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "quota_map requires NPU tensors");
    check_apply_tensor(selected, "selected", at::kLong, device);
    check_apply_tensor(copy_slots, "copy_slots", at::kLong, device);
    check_apply_tensor(copy_counts, "copy_counts", at::kLong, device);
    check_apply_tensor(owner_ranks, "owner_ranks", at::kLong, device);
    check_apply_tensor(quota_weights, "quota_weights", at::kLong, device);
    check_apply_tensor(quota_configured, "quota_configured", at::kLong, device);
    check_apply_tensor(token_ordinals, "token_ordinals", at::kLong, device);
    TORCH_CHECK(selected.dim() == 2, "selected must be rank 2");
    constexpr int64_t max_uint32 = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(
        selected.size(0) <= max_uint32 && selected.size(1) <= max_uint32,
        "selected dimensions exceed the quota_map tiling range");
    TORCH_CHECK(copy_slots.dim() == 3 && copy_slots.size(0) == 2, "copy_slots must have shape [2, E, C]");
    TORCH_CHECK(copy_counts.dim() == 2 && copy_counts.size(0) == 2, "copy_counts must have shape [2, E]");
    TORCH_CHECK(owner_ranks.sizes() == copy_counts.sizes(), "owner_ranks must match copy_counts");
    TORCH_CHECK(copy_slots.size(1) == copy_counts.size(1), "copy tables disagree on the expert count");
    TORCH_CHECK(copy_slots.size(1) > 0, "quota_map requires at least one logical expert");
    TORCH_CHECK(copy_slots.size(1) <= max_uint32, "the expert count exceeds the quota_map tiling range");
    const int64_t max_copies = copy_slots.size(2);
    TORCH_CHECK(max_copies > 0 && max_copies <= 8, "quota_map supports one through eight copies per expert");
    TORCH_CHECK(
        quota_weights.dim() == 4 && quota_weights.size(0) == 2
            && quota_weights.size(1) == copy_slots.size(1)
            && quota_weights.size(2) == (1LL << max_copies)
            && quota_weights.size(3) == max_copies,
        "quota_weights must have shape [2, E, 2**C, C]");
    TORCH_CHECK(
        quota_configured.dim() == 3 && quota_configured.size(0) == 2
            && quota_configured.size(1) == copy_slots.size(1)
            && quota_configured.size(2) == (1LL << max_copies),
        "quota_configured must have shape [2, E, 2**C]");
    TORCH_CHECK(
        token_ordinals.dim() == 1 && token_ordinals.numel() == selected.size(0),
        "token_ordinals must contain one value per token");
    TORCH_CHECK(slots_per_rank > 0, "slots_per_rank must be positive");
    TORCH_CHECK(slots_per_rank <= max_uint32, "slots_per_rank exceeds the quota_map tiling range");
    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "quota_map supports EP sizes from one through 64");
    TORCH_CHECK(source_rank >= 0 && source_rank < ep_size, "source_rank is outside the EP group");
    TORCH_CHECK(num_levels >= 0 && num_levels <= 2, "quota_map supports zero through two hierarchy levels");
    const int64_t level_sizes[] = {level_size0, level_size1};
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(level_sizes[level] > 0 && ep_size % level_sizes[level] == 0, "invalid hierarchy level size");
    }
    auto output = at::empty({selected.size(0), 2, selected.size(1)}, selected.options());
    int64_t group_width = ep_size;
    for (int64_t level = 0; level < num_levels; ++level) {
        group_width += ep_size / level_sizes[level];
    }
    auto group_counts = at::zeros({2, group_width}, selected.options().dtype(at::kFloat));
    auto assignment_counts = at::zeros({2, ep_size}, selected.options().dtype(at::kFloat));
    if (selected.size(0) == 0 || selected.size(1) == 0) {
        return {output.permute({1, 0, 2}), group_counts, assignment_counts};
    }
    constexpr int64_t max_elements = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(
        selected.numel() <= max_elements / 2147483647LL,
        "quota_map route count is too large for exact configured-quota projection");
    const int64_t mask_count = 1LL << max_copies;
    const auto gcd = [](int64_t lhs, int64_t rhs) {
        while (rhs != 0) {
            const int64_t remainder = lhs % rhs;
            lhs = rhs;
            rhs = remainder;
        }
        return lhs;
    };
    const auto checked_mul = [max_elements](int64_t lhs, int64_t rhs) {
        TORCH_CHECK(lhs >= 0 && rhs >= 0 && (lhs == 0 || rhs <= max_elements / lhs),
                    "quota_map workspace size overflow");
        return lhs * rhs;
    };
    const auto checked_add = [max_elements](int64_t lhs, int64_t rhs) {
        TORCH_CHECK(lhs >= 0 && rhs >= 0 && lhs <= max_elements - rhs,
                    "quota_map workspace size overflow");
        return lhs + rhs;
    };
    const auto align_up = [&checked_add](int64_t value, int64_t alignment) {
        return checked_add(value, alignment - 1) / alignment * alignment;
    };

    constexpr int64_t max_blocks = 64;
    const int64_t top_k = selected.size(1);
    const int64_t physical_alignment = 8 / gcd(8, 2 * top_k);
    const int64_t record_alignment = 8 / gcd(8, top_k);
    const int64_t token_alignment = physical_alignment / gcd(physical_alignment, record_alignment)
        * record_alignment;
    int64_t maximum_sort_stride = 0;
    for (int64_t blocks = 1; blocks <= max_blocks; ++blocks) {
        const int64_t unaligned_tokens = (selected.size(0) + blocks - 1) / blocks;
        const int64_t tokens_per_block = align_up(std::max<int64_t>(1, unaligned_tokens), token_alignment);
        const int64_t run_capacity = checked_mul(tokens_per_block, top_k);
        maximum_sort_stride = std::max(maximum_sort_stride, checked_mul(run_capacity, blocks));
    }

    const int64_t record_capacity = checked_mul(2, maximum_sort_stride);
    int64_t workspace_elements = checked_mul(5, record_capacity);
    workspace_elements = checked_add(workspace_elements, checked_mul(4, maximum_sort_stride));

    const int64_t dense_bucket_count = checked_mul(copy_slots.size(1), mask_count);
    const int64_t bucket_stride = align_up(dense_bucket_count, 8);
    workspace_elements = checked_add(workspace_elements, checked_mul(4 * max_blocks, bucket_stride));
    workspace_elements = checked_add(workspace_elements, checked_mul(4, bucket_stride));

    const int64_t rank_stride = align_up(ep_size, 8);
    workspace_elements = checked_add(workspace_elements, checked_mul(2 * max_blocks, rank_stride));
    workspace_elements = checked_add(workspace_elements, checked_mul(2, rank_stride));
    workspace_elements = checked_add(workspace_elements, checked_mul(max_blocks, 16));
    const int64_t stats_stride = align_up(checked_mul(2, checked_add(group_width, ep_size)), 8);
    workspace_elements = checked_add(workspace_elements, checked_mul(max_blocks, stats_stride));
    auto int_workspace = at::empty({workspace_elements}, selected.options());
    EXEC_NPU_CMD(
        aclnnHiermoeQuotaMap,
        selected,
        copy_slots,
        copy_counts,
        owner_ranks,
        quota_weights,
        quota_configured,
        token_ordinals,
        slots_per_rank,
        source_rank,
        ep_size,
        num_levels,
        level_size0,
        level_size1,
        step,
        layer_seed,
        output,
        group_counts,
        assignment_counts,
        int_workspace);
    return {output.permute({1, 0, 2}), group_counts, assignment_counts};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hiermoe_swap_search_npu(
    const at::Tensor &sample_routes,
    const at::Tensor &sample_weights,
    const at::Tensor &assignment_counts,
    const at::Tensor &slot_to_logical,
    const at::Tensor &owner_slots,
    int64_t max_swaps,
    int64_t slots_per_rank,
    int64_t ep_size,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t level_size2,
    int64_t payload_bytes,
    double communication_scale,
    double compute_per_assignment,
    double a2a_alpha,
    double a2a_beta,
    double inter_alpha0,
    double inter_beta0,
    double inter_alpha1,
    double inter_beta1,
    double intra_alpha,
    double intra_beta,
    bool choose_min_dimension)
{
    const auto device = sample_routes.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "swap_search requires NPU tensors");
    check_apply_tensor(sample_routes, "sample_routes", at::kLong, device);
    check_apply_tensor(sample_weights, "sample_weights", at::kFloat, device);
    check_apply_tensor(assignment_counts, "assignment_counts", at::kLong, device);
    check_apply_tensor(slot_to_logical, "slot_to_logical", at::kLong, device);
    check_apply_tensor(owner_slots, "owner_slots", at::kLong, device);
    TORCH_CHECK(sample_routes.dim() == 2, "sample_routes must be rank 2");
    TORCH_CHECK(sample_weights.dim() == 1 && sample_weights.numel() == sample_routes.size(0),
                "sample_weights must match the sample token count");
    TORCH_CHECK(assignment_counts.dim() == 2 && assignment_counts.size(0) == ep_size,
                "assignment_counts must have shape [EP, E]");
    TORCH_CHECK(owner_slots.dim() == 1 && owner_slots.numel() == assignment_counts.size(1),
                "owner_slots must contain one slot per logical expert");
    TORCH_CHECK(slot_to_logical.dim() == 1 && slot_to_logical.numel() == ep_size * slots_per_rank,
                "slot_to_logical has an invalid length");
    TORCH_CHECK(max_swaps >= 0 && max_swaps <= 16, "swap_search supports zero through sixteen swaps");
    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "swap_search supports EP sizes from 1 through 64");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 3, "swap_search supports one through three levels");
    TORCH_CHECK(sample_routes.size(1) > 0 && sample_routes.size(1) <= 16,
                "swap_search supports top-k widths from 1 through 16");
    TORCH_CHECK(payload_bytes > 0, "payload_bytes must be positive");

    const int64_t num_experts = owner_slots.numel();
    TORCH_CHECK(num_experts > 0 && num_experts <= 256,
                "swap_search supports one through 256 logical experts");
    constexpr int64_t max_blocks = 64;
    int64_t packed_tokens_capacity = 0;
    for (int64_t block_count = 1; block_count <= max_blocks; ++block_count) {
        const int64_t tokens_per_block =
            (sample_routes.size(0) + block_count - 1) / block_count;
        const int64_t token_width = (tokens_per_block + 15) / 16 * 16;
        packed_tokens_capacity = std::max(packed_tokens_capacity, block_count * token_width);
    }
    const int64_t action_rows = std::max<int64_t>(1, max_swaps);
    auto updated_layout = at::empty_like(slot_to_logical);
    auto updated_owners = at::empty_like(owner_slots);
    auto actions = at::full({action_rows, 5}, -1, owner_slots.options());
    auto metadata = at::zeros({8}, owner_slots.options().dtype(at::kInt));

    const int64_t level_sizes[] = {level_size0, level_size1, level_size2};
    int64_t total_groups = 0;
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(level_sizes[level] > 0 && ep_size % level_sizes[level] == 0,
                    "invalid hierarchy level size");
        total_groups += ep_size / level_sizes[level];
    }
    const int64_t expert_group_stride = (total_groups + 31) / 32 * 32;
    const int64_t sole_pair_stride = (num_experts + 31) / 32 * 32;
    const int64_t stats_width = total_groups * 32 + num_experts * 32
        + num_experts * expert_group_stride + num_experts * 32
        + num_experts * num_levels * sole_pair_stride;
    const int64_t stats_stride = (stats_width + 31) / 32 * 32;
    auto float_workspace = at::zeros(
        {max_blocks * stats_stride + stats_stride + ep_size + num_experts + max_blocks * 32 + 64},
        sample_weights.options());
    const int64_t expert_count_stride = (num_experts + 15) / 16 * 16;
    auto int_workspace = at::zeros(
        {expert_count_stride + max_blocks * expert_count_stride + max_blocks * 16
             + num_experts * packed_tokens_capacity + num_experts * 16 + 192,
         },
        owner_slots.options());

    EXEC_NPU_CMD(
        aclnnHiermoeSwapSearch,
        sample_routes,
        sample_weights,
        assignment_counts,
        slot_to_logical,
        owner_slots,
        max_swaps,
        slots_per_rank,
        ep_size,
        num_levels,
        level_size0,
        level_size1,
        level_size2,
        payload_bytes,
        communication_scale,
        compute_per_assignment,
        a2a_alpha,
        a2a_beta,
        inter_alpha0,
        inter_beta0,
        inter_alpha1,
        inter_beta1,
        intra_alpha,
        intra_beta,
        choose_min_dimension,
        updated_layout,
        updated_owners,
        actions,
        metadata,
        float_workspace,
        int_workspace);
    return {updated_layout, updated_owners, actions, metadata};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hiermoe_swap_select_with_stats_npu(
    const at::Tensor &expert_token_counts,
    const at::Tensor &expert_assignment_counts,
    const at::Tensor &base_counts,
    const at::Tensor &expert_group_counts,
    const at::Tensor &sole_expert_counts,
    const at::Tensor &sole_pair_counts,
    const at::Tensor &sample_routes,
    const at::Tensor &sample_weights,
    const at::Tensor &slot_to_logical,
    const at::Tensor &owner_slots,
    const at::Tensor &expert_state_bytes,
    const at::Tensor &expert_gradient_bytes,
    int64_t max_swaps,
    int64_t slots_per_rank,
    int64_t ep_size,
    int64_t local_world_size,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t level_size2,
    int64_t payload_bytes,
    double communication_scale,
    double compute_per_assignment,
    double a2a_alpha,
    double a2a_beta,
    double inter_alpha0,
    double inter_beta0,
    double inter_alpha1,
    double inter_beta1,
    double intra_alpha,
    double intra_beta,
    double state_intra_alpha,
    double state_intra_beta,
    double state_inter_alpha,
    double state_inter_beta,
    double gather_intra_alpha,
    double gather_intra_beta,
    double gather_inter_alpha,
    double gather_inter_beta,
    double scatter_intra_alpha,
    double scatter_intra_beta,
    double scatter_inter_alpha,
    double scatter_inter_beta,
    double runtime_cost_scale,
    bool choose_min_dimension)
{
    const auto device = owner_slots.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "swap_select requires NPU tensors");
    check_apply_tensor(expert_token_counts, "expert_token_counts", at::kFloat, device);
    check_apply_tensor(expert_assignment_counts, "expert_assignment_counts", at::kFloat, device);
    check_apply_tensor(base_counts, "base_counts", at::kFloat, device);
    check_apply_tensor(expert_group_counts, "expert_group_counts", at::kFloat, device);
    check_apply_tensor(sole_expert_counts, "sole_expert_counts", at::kFloat, device);
    check_apply_tensor(sole_pair_counts, "sole_pair_counts", at::kFloat, device);
    check_apply_tensor(sample_routes, "sample_routes", at::kLong, device);
    check_apply_tensor(sample_weights, "sample_weights", at::kFloat, device);
    check_apply_tensor(slot_to_logical, "slot_to_logical", at::kLong, device);
    check_apply_tensor(owner_slots, "owner_slots", at::kLong, device);
    check_apply_tensor(expert_state_bytes, "expert_state_bytes", at::kLong, device);
    check_apply_tensor(expert_gradient_bytes, "expert_gradient_bytes", at::kLong, device);
    const int64_t num_experts = owner_slots.numel();
    TORCH_CHECK(num_experts > 0 && num_experts <= 256, "swap_select supports one through 256 experts");
    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "swap_select supports EP sizes from one through 64");
    TORCH_CHECK(max_swaps >= 0 && max_swaps <= 16, "swap_select supports zero through sixteen swaps");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 3, "swap_select supports one through three levels");
    TORCH_CHECK(expert_token_counts.numel() == num_experts, "expert token counts have an invalid length");
    TORCH_CHECK(expert_assignment_counts.numel() == num_experts, "expert assignment counts have an invalid length");
    TORCH_CHECK(expert_state_bytes.numel() == num_experts, "expert state bytes have an invalid length");
    TORCH_CHECK(expert_gradient_bytes.numel() == num_experts, "expert gradient bytes have an invalid length");
    TORCH_CHECK(slot_to_logical.numel() == ep_size * slots_per_rank, "slot layout has an invalid length");
    const int64_t level_sizes[] = {level_size0, level_size1, level_size2};
    int64_t total_groups = 0;
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(level_sizes[level] > 0 && ep_size % level_sizes[level] == 0, "invalid hierarchy level");
        total_groups += ep_size / level_sizes[level];
    }
    TORCH_CHECK(base_counts.numel() == total_groups, "base counts disagree with the hierarchy");
    TORCH_CHECK(
        expert_group_counts.dim() == 2 && expert_group_counts.size(0) == num_experts
            && expert_group_counts.size(1) == total_groups,
        "expert group counts have an invalid shape");
    TORCH_CHECK(
        sole_expert_counts.dim() == 2 && sole_expert_counts.size(0) == num_experts
            && sole_expert_counts.size(1) == num_levels,
        "sole expert counts have an invalid shape");
    TORCH_CHECK(
        sole_pair_counts.dim() == 3 && sole_pair_counts.size(0) == num_experts
            && sole_pair_counts.size(1) == num_levels && sole_pair_counts.size(2) == num_experts,
        "sole pair counts have an invalid shape");
    TORCH_CHECK(
        sample_routes.dim() == 2 && sample_routes.size(1) >= 1 && sample_routes.size(1) <= 16,
        "sample_routes must have shape [M, K] with one through sixteen routes per token");
    TORCH_CHECK(
        sample_weights.dim() == 1 && sample_weights.numel() == sample_routes.size(0),
        "sample_weights must contain one value per sampled token");

    constexpr int64_t max_blocks = 64;
    constexpr int64_t max_update_blocks = max_blocks;
    constexpr int64_t local_delta_capacity = 40 * 1024;
    const auto align16 = [](int64_t value) { return (value + 15) / 16 * 16; };
    const int64_t assignment_offset = align16(total_groups);
    const int64_t state_payload_offset = align16(assignment_offset + ep_size);
    const int64_t state_rank_offset = align16(state_payload_offset + ep_size * ep_size);
    const int64_t gradient_payload_offset = align16(state_rank_offset + ep_size);
    const int64_t gradient_rank_offset = align16(gradient_payload_offset + ep_size * ep_size);
    const int64_t best_cost_offset = align16(gradient_rank_offset + ep_size);
    const int64_t current_cost_offset = best_cost_offset + max_blocks * 16;
    const int64_t expert_group_delta_stride = align16(total_groups);
    const int64_t sole_expert_delta_stride = align16(num_levels);
    const int64_t sole_pair_delta_stride = align16(num_levels * num_experts);
    const int64_t expert_group_delta_offset = align16(current_cost_offset + 16);
    const int64_t sole_expert_delta_offset = align16(
        expert_group_delta_offset + num_experts * expert_group_delta_stride);
    const int64_t sole_pair_delta_offset = align16(
        sole_expert_delta_offset + num_experts * sole_expert_delta_stride);
    const int64_t private_delta_size = align16(
        sole_pair_delta_offset + num_experts * sole_pair_delta_stride - expert_group_delta_offset);
    const int64_t private_delta_offset = align16(
        sole_pair_delta_offset + num_experts * sole_pair_delta_stride);
    const int64_t private_blocks = max_swaps > 1 ? max_update_blocks : 0;
    const int64_t best_pair_offset = align16(num_experts);
    const int64_t control_offset = best_pair_offset + max_blocks * 16;
    const int64_t membership_offset = align16(control_offset + 16);
    const int64_t membership_elements = max_swaps > 1 ? sample_routes.size(0) * 8 : 0;
    const int64_t int_workspace_size = align16(membership_offset + membership_elements);
    auto updated_layout = at::empty_like(slot_to_logical);
    auto updated_owners = at::empty_like(owner_slots);
    auto actions = at::empty({std::max<int64_t>(1, max_swaps), 5}, owner_slots.options());
    auto metadata = at::empty({8}, owner_slots.options().dtype(at::kInt));
    const int64_t float_workspace_size = private_delta_offset + private_blocks * private_delta_size;
    auto float_workspace = max_swaps > 1 && private_delta_size > local_delta_capacity
        ? at::zeros({float_workspace_size}, expert_token_counts.options())
        : at::empty({float_workspace_size}, expert_token_counts.options());
    auto int_workspace = at::empty({int_workspace_size}, owner_slots.options());
    EXEC_NPU_CMD(
        aclnnHiermoeSwapSelect,
        expert_token_counts,
        expert_assignment_counts,
        base_counts,
        expert_group_counts,
        sole_expert_counts,
        sole_pair_counts,
        sample_routes,
        sample_weights,
        slot_to_logical,
        owner_slots,
        expert_state_bytes,
        expert_gradient_bytes,
        max_swaps,
        slots_per_rank,
        ep_size,
        local_world_size,
        num_levels,
        level_size0,
        level_size1,
        level_size2,
        payload_bytes,
        communication_scale,
        compute_per_assignment,
        a2a_alpha,
        a2a_beta,
        inter_alpha0,
        inter_beta0,
        inter_alpha1,
        inter_beta1,
        intra_alpha,
        intra_beta,
        state_intra_alpha,
        state_intra_beta,
        state_inter_alpha,
        state_inter_beta,
        gather_intra_alpha,
        gather_intra_beta,
        gather_inter_alpha,
        gather_inter_beta,
        scatter_intra_alpha,
        scatter_intra_beta,
        scatter_inter_alpha,
        scatter_inter_beta,
        runtime_cost_scale,
        choose_min_dimension,
        updated_layout,
        updated_owners,
        actions,
        metadata,
        float_workspace,
        int_workspace);
    return {
        updated_layout,
        updated_owners,
        actions,
        metadata,
        float_workspace.narrow(0, 0, total_groups)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hiermoe_swap_select_npu(
    const at::Tensor &expert_token_counts,
    const at::Tensor &expert_assignment_counts,
    const at::Tensor &base_counts,
    const at::Tensor &expert_group_counts,
    const at::Tensor &sole_expert_counts,
    const at::Tensor &sole_pair_counts,
    const at::Tensor &sample_routes,
    const at::Tensor &sample_weights,
    const at::Tensor &slot_to_logical,
    const at::Tensor &owner_slots,
    const at::Tensor &expert_state_bytes,
    const at::Tensor &expert_gradient_bytes,
    int64_t max_swaps,
    int64_t slots_per_rank,
    int64_t ep_size,
    int64_t local_world_size,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t level_size2,
    int64_t payload_bytes,
    double communication_scale,
    double compute_per_assignment,
    double a2a_alpha,
    double a2a_beta,
    double inter_alpha0,
    double inter_beta0,
    double inter_alpha1,
    double inter_beta1,
    double intra_alpha,
    double intra_beta,
    double state_intra_alpha,
    double state_intra_beta,
    double state_inter_alpha,
    double state_inter_beta,
    double gather_intra_alpha,
    double gather_intra_beta,
    double gather_inter_alpha,
    double gather_inter_beta,
    double scatter_intra_alpha,
    double scatter_intra_beta,
    double scatter_inter_alpha,
    double scatter_inter_beta,
    double runtime_cost_scale,
    bool choose_min_dimension)
{
    auto result = hiermoe_swap_select_with_stats_npu(
        expert_token_counts,
        expert_assignment_counts,
        base_counts,
        expert_group_counts,
        sole_expert_counts,
        sole_pair_counts,
        sample_routes,
        sample_weights,
        slot_to_logical,
        owner_slots,
        expert_state_bytes,
        expert_gradient_bytes,
        max_swaps,
        slots_per_rank,
        ep_size,
        local_world_size,
        num_levels,
        level_size0,
        level_size1,
        level_size2,
        payload_bytes,
        communication_scale,
        compute_per_assignment,
        a2a_alpha,
        a2a_beta,
        inter_alpha0,
        inter_beta0,
        inter_alpha1,
        inter_beta1,
        intra_alpha,
        intra_beta,
        state_intra_alpha,
        state_intra_beta,
        state_inter_alpha,
        state_inter_beta,
        gather_intra_alpha,
        gather_intra_beta,
        gather_inter_alpha,
        gather_inter_beta,
        scatter_intra_alpha,
        scatter_intra_beta,
        scatter_inter_alpha,
        scatter_inter_beta,
        runtime_cost_scale,
        choose_min_dimension);
    return {
        std::get<0>(result),
        std::get<1>(result),
        std::get<2>(result),
        std::get<3>(result)};
}
