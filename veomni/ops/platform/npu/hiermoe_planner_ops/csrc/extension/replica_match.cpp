#include <algorithm>

#include "function.h"
#include "pytorch_npu_helper.hpp"

namespace {
void check_replica_match_tensor(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hiermoe_replica_match_npu(
    const at::Tensor &base_counts,
    const at::Tensor &assignment_loads,
    const at::Tensor &add_group_deltas,
    const at::Tensor &add_assignment_deltas,
    const at::Tensor &remove_group_deltas,
    const at::Tensor &remove_assignment_deltas,
    const at::Tensor &slot_to_logical,
    const at::Tensor &owner_slots,
    const at::Tensor &redundant_slots,
    const at::Tensor &candidate_experts,
    const at::Tensor &expert_state_bytes,
    const at::Tensor &expert_gradient_bytes,
    int64_t max_actions,
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
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "replica_match requires NPU tensors");
    check_replica_match_tensor(base_counts, "base_counts", at::kFloat, device);
    check_replica_match_tensor(assignment_loads, "assignment_loads", at::kFloat, device);
    check_replica_match_tensor(add_group_deltas, "add_group_deltas", at::kFloat, device);
    check_replica_match_tensor(add_assignment_deltas, "add_assignment_deltas", at::kFloat, device);
    check_replica_match_tensor(remove_group_deltas, "remove_group_deltas", at::kFloat, device);
    check_replica_match_tensor(remove_assignment_deltas, "remove_assignment_deltas", at::kFloat, device);
    check_replica_match_tensor(slot_to_logical, "slot_to_logical", at::kLong, device);
    check_replica_match_tensor(owner_slots, "owner_slots", at::kLong, device);
    check_replica_match_tensor(redundant_slots, "redundant_slots", at::kLong, device);
    check_replica_match_tensor(candidate_experts, "candidate_experts", at::kInt, device);
    check_replica_match_tensor(expert_state_bytes, "expert_state_bytes", at::kLong, device);
    check_replica_match_tensor(expert_gradient_bytes, "expert_gradient_bytes", at::kLong, device);

    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "replica_match supports EP sizes from one through 64");
    TORCH_CHECK(slots_per_rank > 0, "slots_per_rank must be positive");
    TORCH_CHECK(local_world_size > 0 && local_world_size <= ep_size, "local_world_size is outside the EP group");
    TORCH_CHECK(ep_size % local_world_size == 0, "local_world_size must divide ep_size");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 3, "replica_match supports one through three levels");
    TORCH_CHECK(payload_bytes >= 0, "payload_bytes must be nonnegative");
    const int64_t num_experts = owner_slots.numel();
    TORCH_CHECK(num_experts > 0 && num_experts <= 256, "replica_match supports one through 256 experts");
    TORCH_CHECK(slot_to_logical.dim() == 1
                    && slot_to_logical.numel() == ep_size * slots_per_rank,
                "slot_to_logical must be a flat [ep_size * slots_per_rank] tensor");
    TORCH_CHECK(redundant_slots.dim() == 2 && redundant_slots.size(0) == ep_size,
                "redundant_slots must have shape [ep_size, slots]");
    const int64_t redundant_slots_per_rank = redundant_slots.size(1);
    TORCH_CHECK(redundant_slots_per_rank > 0 && redundant_slots_per_rank <= 8,
                "replica_match supports one through eight redundant slots per rank");
    TORCH_CHECK(max_actions >= 0 && max_actions <= ep_size * redundant_slots_per_rank,
                "max_actions exceeds the redundant slot capacity");
    TORCH_CHECK(owner_slots.dim() == 1, "owner_slots must be a flat [experts] tensor");
    TORCH_CHECK(candidate_experts.dim() == 1 && candidate_experts.numel() == num_experts,
                "candidate_experts must be a flat [experts] tensor");
    TORCH_CHECK(expert_state_bytes.dim() == 1 && expert_state_bytes.numel() == num_experts,
                "expert_state_bytes must be a flat [experts] tensor");
    TORCH_CHECK(expert_gradient_bytes.dim() == 1 && expert_gradient_bytes.numel() == num_experts,
                "expert_gradient_bytes must be a flat [experts] tensor");

    const int64_t level_sizes[] = {level_size0, level_size1, level_size2};
    int64_t total_groups = 0;
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(level_sizes[level] > 0 && ep_size % level_sizes[level] == 0,
                    "invalid hierarchy level size");
        total_groups += ep_size / level_sizes[level];
    }
    TORCH_CHECK(level_sizes[num_levels - 1] == 1,
                "the final replica_match hierarchy level must contain one rank per group");
    TORCH_CHECK(base_counts.dim() == 1 && base_counts.numel() == total_groups,
                "base_counts disagrees with the hierarchy");
    TORCH_CHECK(assignment_loads.dim() == 1 && assignment_loads.numel() == ep_size,
                "assignment_loads must contain one value per rank");
    TORCH_CHECK(add_group_deltas.dim() == 3
                    && add_group_deltas.size(0) == num_experts
                    && add_group_deltas.size(1) == ep_size
                    && add_group_deltas.size(2) == total_groups,
                "add_group_deltas must have shape [experts, ep_size, groups]");
    TORCH_CHECK(add_assignment_deltas.dim() == 3
                    && add_assignment_deltas.size(0) == num_experts
                    && add_assignment_deltas.size(1) == ep_size
                    && add_assignment_deltas.size(2) == ep_size,
                "add_assignment_deltas must have shape [experts, ep_size, ep_size]");
    TORCH_CHECK(remove_group_deltas.dim() == 3
                    && remove_group_deltas.size(0) == ep_size
                    && remove_group_deltas.size(1) == redundant_slots_per_rank
                    && remove_group_deltas.size(2) == total_groups,
                "remove_group_deltas must have shape [ep_size, slots, groups]");
    TORCH_CHECK(remove_assignment_deltas.dim() == 3
                    && remove_assignment_deltas.size(0) == ep_size
                    && remove_assignment_deltas.size(1) == redundant_slots_per_rank
                    && remove_assignment_deltas.size(2) == ep_size,
                "remove_assignment_deltas must have shape [ep_size, slots, ep_size]");

    const int64_t action_capacity = std::max<int64_t>(1, ep_size * redundant_slots_per_rank);
    const int64_t num_columns = num_experts + 2 * redundant_slots_per_rank;
    const auto align16 = [](int64_t value) { return (value + 15) / 16 * 16; };
    const int64_t gradient_rank_offset = align16(ep_size * ep_size);
    const int64_t baseline_cost_offset = align16(gradient_rank_offset + ep_size);
    auto updated_layout = at::empty_like(slot_to_logical);
    auto actions = at::full({action_capacity, 5}, -1, owner_slots.options());
    auto action_gains = at::zeros({action_capacity}, base_counts.options());
    // Keep every rank on distinct 64-byte cache lines while AIVs score ranks
    // concurrently.  Return metadata-only views with the compact public shape.
    const int64_t selected_stride = align16(redundant_slots_per_rank);
    const int64_t gain_stride = align16(redundant_slots_per_rank * num_columns);
    auto selected_storage = at::full({ep_size, selected_stride}, -1, candidate_experts.options());
    auto selected_columns = selected_storage.narrow(1, 0, redundant_slots_per_rank);
    auto gain_storage = at::empty({ep_size, gain_stride}, base_counts.options());
    auto gain_matrix = gain_storage.as_strided(
        {ep_size, redundant_slots_per_rank, num_columns},
        {gain_stride, num_columns, 1});
    auto metadata = at::zeros({16}, candidate_experts.options());
    auto float_workspace = at::zeros({baseline_cost_offset + 16}, base_counts.options());
    EXEC_NPU_CMD(
        aclnnHiermoeReplicaMatch,
        base_counts,
        assignment_loads,
        add_group_deltas,
        add_assignment_deltas,
        remove_group_deltas,
        remove_assignment_deltas,
        slot_to_logical,
        owner_slots,
        redundant_slots,
        candidate_experts,
        expert_state_bytes,
        expert_gradient_bytes,
        max_actions,
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
        actions,
        action_gains,
        selected_storage,
        gain_storage,
        metadata,
        float_workspace);
    return {updated_layout, actions, action_gains, selected_columns, gain_matrix, metadata};
}
