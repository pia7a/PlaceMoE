#include <algorithm>
#include <limits>

#include "function.h"
#include "pytorch_npu_helper.hpp"

namespace {
void check_replica_project_tensor(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    const c10::Device &device)
{
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t align16(int64_t value)
{
    return (value + 15) / 16 * 16;
}

int64_t checked_add(int64_t lhs, int64_t rhs, const char *message)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0 && lhs <= std::numeric_limits<int64_t>::max() - rhs, message);
    return lhs + rhs;
}

int64_t checked_mul(int64_t lhs, int64_t rhs, const char *message)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0 && (lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs), message);
    return lhs * rhs;
}

int64_t checked_align16(int64_t value, const char *message)
{
    return checked_add(value, 15, message) / 16 * 16;
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hiermoe_replica_project_npu(
    const at::Tensor &sample_routes,
    const at::Tensor &sample_multiplicity,
    const at::Tensor &sample_weights,
    const at::Tensor &sample_sources,
    const at::Tensor &sample_ordinals,
    const at::Tensor &assignment_counts,
    const at::Tensor &seed_base_counts,
    const at::Tensor &slot_to_logical,
    const at::Tensor &owner_slots,
    const at::Tensor &redundant_slots,
    const at::Tensor &candidate_experts,
    int64_t slots_per_rank,
    int64_t ep_size,
    int64_t num_levels,
    int64_t level_size0,
    int64_t level_size1,
    int64_t level_size2,
    int64_t step,
    int64_t layer_seed)
{
    const auto device = sample_routes.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "replica_project requires NPU tensors");
    check_replica_project_tensor(sample_routes, "sample_routes", at::kLong, device);
    check_replica_project_tensor(sample_multiplicity, "sample_multiplicity", at::kLong, device);
    check_replica_project_tensor(sample_weights, "sample_weights", at::kFloat, device);
    check_replica_project_tensor(sample_sources, "sample_sources", at::kLong, device);
    check_replica_project_tensor(sample_ordinals, "sample_ordinals", at::kLong, device);
    check_replica_project_tensor(assignment_counts, "assignment_counts", at::kLong, device);
    check_replica_project_tensor(seed_base_counts, "seed_base_counts", at::kFloat, device);
    check_replica_project_tensor(slot_to_logical, "slot_to_logical", at::kLong, device);
    check_replica_project_tensor(owner_slots, "owner_slots", at::kLong, device);
    check_replica_project_tensor(redundant_slots, "redundant_slots", at::kLong, device);
    check_replica_project_tensor(candidate_experts, "candidate_experts", at::kInt, device);

    TORCH_CHECK(sample_routes.dim() == 2, "sample_routes must have shape [samples, top_k]");
    TORCH_CHECK(sample_routes.size(1) > 0 && sample_routes.size(1) <= 16,
                "replica_project supports top-k widths from one through sixteen");
    TORCH_CHECK(sample_multiplicity.sizes() == sample_routes.sizes(),
                "sample_multiplicity must match sample_routes");
    const int64_t num_samples = sample_routes.size(0);
    TORCH_CHECK(sample_weights.dim() == 1 && sample_weights.numel() == num_samples,
                "sample_weights must contain one value per sample");
    TORCH_CHECK(sample_sources.dim() == 1 && sample_sources.numel() == num_samples,
                "sample_sources must contain one value per sample");
    TORCH_CHECK(sample_ordinals.dim() == 1 && sample_ordinals.numel() == num_samples,
                "sample_ordinals must contain one value per sample");
    TORCH_CHECK(ep_size > 0 && ep_size <= 64, "replica_project supports EP sizes from one through 64");
    constexpr int64_t max_uint32 = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(slots_per_rank > 0 && slots_per_rank <= max_uint32,
                "slots_per_rank must fit in the uint32 tiling field");
    const int64_t num_slots = checked_mul(ep_size, slots_per_rank, "replica_project slot count overflow");
    TORCH_CHECK(num_slots <= max_uint32, "replica_project slot count must fit in the uint32 tiling field");
    TORCH_CHECK(assignment_counts.dim() == 2 && assignment_counts.size(0) == ep_size,
                "assignment_counts must have shape [ep_size, experts]");
    const int64_t num_experts = assignment_counts.size(1);
    TORCH_CHECK(num_experts > 0 && num_experts <= 256,
                "replica_project supports one through 256 logical experts");
    TORCH_CHECK(owner_slots.dim() == 1 && owner_slots.numel() == num_experts,
                "owner_slots must contain one slot per logical expert");
    TORCH_CHECK(slot_to_logical.dim() == 1 && slot_to_logical.numel() == num_slots,
                "slot_to_logical has an invalid length");
    TORCH_CHECK(redundant_slots.dim() == 2 && redundant_slots.size(0) == ep_size,
                "redundant_slots must have shape [ep_size, slots]");
    const int64_t redundant_slots_per_rank = redundant_slots.size(1);
    TORCH_CHECK(redundant_slots_per_rank > 0 && redundant_slots_per_rank <= 8,
                "replica_project supports one through eight redundant slots per rank");
    TORCH_CHECK(candidate_experts.dim() == 1 && candidate_experts.numel() == num_experts,
                "candidate_experts must contain one mask value per logical expert");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 3,
                "replica_project supports one through three hierarchy levels");
    const int64_t level_sizes[] = {level_size0, level_size1, level_size2};
    int64_t total_groups = 0;
    for (int64_t level = 0; level < num_levels; ++level) {
        TORCH_CHECK(level_sizes[level] > 0 && ep_size % level_sizes[level] == 0,
                    "invalid replica_project hierarchy level");
        total_groups += ep_size / level_sizes[level];
    }
    TORCH_CHECK(level_sizes[num_levels - 1] == 1,
                "the final replica_project hierarchy level must contain one rank per group");
    TORCH_CHECK(total_groups <= 192, "replica_project hierarchy exceeds the fixed group capacity");
    TORCH_CHECK(seed_base_counts.dim() == 1 && seed_base_counts.numel() == total_groups,
                "seed_base_counts must contain one value per hierarchy group");

    const int64_t max_records = sample_routes.numel();
    TORCH_CHECK(max_records <= (1LL << 28), "replica_project sample workspace is too large");
    int64_t hash_capacity = 2;
    const int64_t desired = std::max<int64_t>(2, 2 * max_records);
    while (hash_capacity < desired) {
        hash_capacity <<= 1;
    }
    const int64_t max_elements = std::numeric_limits<int64_t>::max();
    const int64_t distribution_experts = std::max<int64_t>(2, num_experts);
    TORCH_CHECK(ep_size <= max_elements / distribution_experts / ep_size,
                "replica_project distribution workspace size overflow");
    const int64_t distribution_size = ep_size * distribution_experts * ep_size;
    TORCH_CHECK(max_records <= (max_elements - 3 * hash_capacity - distribution_size - num_samples - 1) / 8,
                "replica_project integer workspace size overflow");
    const int64_t baseline_int_workspace_size = 8 * max_records + 3 * hash_capacity
        + distribution_size + num_samples + 1;
    // Each independently scored edge touches one logical expert and therefore
    // at most one canonical record per sampled token.  Keep the full-route
    // scratch once for the baseline and give action cores the smaller domain.
    const int64_t action_max_records = num_samples;
    int64_t action_hash_capacity = 2;
    const int64_t action_desired = std::max<int64_t>(2, 2 * action_max_records);
    while (action_hash_capacity < action_desired) {
        action_hash_capacity <<= 1;
    }
    TORCH_CHECK(ep_size <= max_elements / ep_size / 2,
                "replica_project action distribution workspace size overflow");
    const int64_t action_distribution_size = 2 * ep_size * ep_size;
    TORCH_CHECK(action_max_records
                    <= (max_elements - 3 * action_hash_capacity - action_distribution_size - num_samples - 1) / 8,
                "replica_project action integer workspace size overflow");
    const int64_t action_int_workspace_size = 8 * action_max_records + 3 * action_hash_capacity
        + action_distribution_size + num_samples + 1;
    const int64_t baseline_int_workspace_stride = align16(baseline_int_workspace_size);
    const int64_t action_int_workspace_stride = align16(action_int_workspace_size);
    const int64_t action_rows = std::max(
        num_experts * ep_size,
        ep_size * redundant_slots_per_rank);
    const int64_t action_workspace_blocks = std::min<int64_t>(64, std::max<int64_t>(1, action_rows));
    const int64_t rank_stride = align16(ep_size);
    const int64_t generic_baseline_float_workspace = checked_add(
        rank_stride,
        max_records,
        "replica_project generic baseline float workspace size overflow");
    int64_t direct_baseline_float_workspace = checked_add(
        rank_stride,
        checked_mul(
            action_workspace_blocks,
            rank_stride,
            "replica_project block rank-load workspace size overflow"),
        "replica_project direct baseline float workspace size overflow");
    direct_baseline_float_workspace = checked_add(
        direct_baseline_float_workspace,
        checked_mul(16, num_experts, "replica_project expert-load workspace size overflow"),
        "replica_project direct baseline float workspace size overflow");
    const int64_t baseline_float_workspace_stride = checked_align16(
        std::max(generic_baseline_float_workspace, direct_baseline_float_workspace),
        "replica_project baseline float workspace alignment overflow");
    const int64_t action_float_workspace_stride = align16(align16(ep_size) + action_max_records);
    const int64_t direct_block_count_stride = checked_mul(
        3,
        align16(num_experts),
        "replica_project direct block metadata stride overflow");
    const int64_t direct_block_load_stride = align16(num_experts);
    TORCH_CHECK(action_int_workspace_stride <= (max_elements - baseline_int_workspace_stride)
                        / action_workspace_blocks,
                "replica_project action integer workspace block expansion overflow");
    TORCH_CHECK(action_float_workspace_stride <= (max_elements - baseline_float_workspace_stride)
                        / action_workspace_blocks,
                "replica_project action float workspace block expansion overflow");
    const int64_t action_int_workspace_elements = checked_mul(
        action_workspace_blocks,
        action_int_workspace_stride,
        "replica_project action integer workspace block expansion overflow");
    const int64_t direct_block_count_elements = checked_mul(
        action_workspace_blocks,
        direct_block_count_stride,
        "replica_project direct block-count workspace size overflow");
    const int64_t private_int_workspace_size = checked_add(
        checked_add(
            baseline_int_workspace_stride,
            action_int_workspace_elements,
            "replica_project private integer workspace size overflow"),
        direct_block_count_elements,
        "replica_project private integer workspace size overflow");
    const int64_t action_float_workspace_elements = checked_mul(
        action_workspace_blocks,
        action_float_workspace_stride,
        "replica_project action float workspace block expansion overflow");
    const int64_t direct_block_load_elements = checked_mul(
        action_workspace_blocks,
        direct_block_load_stride,
        "replica_project direct block-load workspace size overflow");
    const int64_t private_float_workspace_size = checked_add(
        checked_add(
            baseline_float_workspace_stride,
            action_float_workspace_elements,
            "replica_project private float workspace size overflow"),
        direct_block_load_elements,
        "replica_project private float workspace size overflow");
    const int64_t shared_record_capacity = checked_add(
        max_records,
        checked_mul(7, num_experts, "replica_project shared record padding overflow"),
        "replica_project shared record capacity overflow");
    const int64_t direct_raw_record_capacity = checked_add(
        max_records,
        checked_mul(
            7,
            checked_mul(
                num_experts,
                action_workspace_blocks,
                "replica_project direct raw row count overflow"),
            "replica_project direct raw padding overflow"),
        "replica_project direct raw record capacity overflow");
    const int64_t all_destination_mask_stride = checked_align16(
        shared_record_capacity,
        "replica_project all-destination mask stride alignment overflow");
    int64_t shared_summary_elements = checked_align16(
        num_experts,
        "replica_project copy-mask summary alignment overflow");
    shared_summary_elements = checked_add(
        shared_summary_elements,
        16,
        "replica_project mode summary size overflow");
    shared_summary_elements = checked_add(
        shared_summary_elements,
        checked_mul(num_samples, 8, "replica_project token coverage size overflow"),
        "replica_project token coverage summary size overflow");
    shared_summary_elements = checked_align16(
        shared_summary_elements,
        "replica_project logical metadata alignment overflow");
    shared_summary_elements = checked_add(
        shared_summary_elements,
        checked_mul(
            checked_add(num_experts, 1, "replica_project logical metadata row count overflow"),
            8,
            "replica_project logical metadata size overflow"),
        "replica_project logical metadata summary size overflow");
    shared_summary_elements = checked_align16(
        shared_summary_elements,
        "replica_project direct raw route-index alignment overflow");
    shared_summary_elements = checked_add(
        shared_summary_elements,
        direct_raw_record_capacity,
        "replica_project direct raw route-index size overflow");
    shared_summary_elements = checked_align16(
        shared_summary_elements,
        "replica_project direct raw route-hash alignment overflow");
    shared_summary_elements = checked_add(
        shared_summary_elements,
        direct_raw_record_capacity,
        "replica_project direct raw route-hash size overflow");
    shared_summary_elements = checked_align16(
        shared_summary_elements,
        "replica_project route-index summary alignment overflow");
    shared_summary_elements = checked_add(
        shared_summary_elements,
        shared_record_capacity,
        "replica_project route-index summary size overflow");
    shared_summary_elements = checked_align16(
        shared_summary_elements,
        "replica_project route-hash summary alignment overflow");
    shared_summary_elements = checked_add(
        shared_summary_elements,
        shared_record_capacity,
        "replica_project route-hash summary size overflow");
    shared_summary_elements = checked_align16(
        shared_summary_elements,
        "replica_project all-destination mask alignment overflow");
    shared_summary_elements = checked_add(
        shared_summary_elements,
        checked_mul(
            2,
            all_destination_mask_stride,
            "replica_project all-destination mask size overflow"),
        "replica_project all-destination mask summary size overflow");
    const int64_t shared_summary_stride = checked_align16(
        shared_summary_elements,
        "replica_project shared summary alignment overflow");
    TORCH_CHECK(private_int_workspace_size <= max_elements - shared_summary_stride,
                "replica_project total integer workspace size overflow");
    const int64_t int_workspace_size = private_int_workspace_size + shared_summary_stride;
    const int64_t float_workspace_size = private_float_workspace_size;

    auto base_counts = at::empty({total_groups}, sample_weights.options());
    auto assignment_loads = at::empty({ep_size}, sample_weights.options());
    // Give every independently scored action its own 64-byte-aligned rows.
    // The public tensors remain metadata-only compact views; the custom op
    // receives the padded storages because it writes raw GM offsets.
    const int64_t group_stride = align16(total_groups);
    const int64_t assignment_stride = align16(ep_size);
    const int64_t add_actions = num_experts * ep_size;
    const int64_t remove_actions = ep_size * redundant_slots_per_rank;
    auto add_group_storage = at::empty({add_actions, group_stride}, sample_weights.options());
    auto add_group_deltas = add_group_storage.as_strided(
        {num_experts, ep_size, total_groups},
        {ep_size * group_stride, group_stride, 1});
    auto add_assignment_storage = at::empty({add_actions, assignment_stride}, sample_weights.options());
    auto add_assignment_deltas = add_assignment_storage.as_strided(
        {num_experts, ep_size, ep_size},
        {ep_size * assignment_stride, assignment_stride, 1});
    auto remove_group_storage = at::empty({remove_actions, group_stride}, sample_weights.options());
    auto remove_group_deltas = remove_group_storage.as_strided(
        {ep_size, redundant_slots_per_rank, total_groups},
        {redundant_slots_per_rank * group_stride, group_stride, 1});
    auto remove_assignment_storage = at::empty({remove_actions, assignment_stride}, sample_weights.options());
    auto remove_assignment_deltas = remove_assignment_storage.as_strided(
        {ep_size, redundant_slots_per_rank, ep_size},
        {redundant_slots_per_rank * assignment_stride, assignment_stride, 1});
    auto int_workspace = at::empty({int_workspace_size}, sample_routes.options());
    auto float_workspace = at::empty({float_workspace_size}, sample_weights.options());
    EXEC_NPU_CMD(
        aclnnHiermoeReplicaProject,
        sample_routes,
        sample_multiplicity,
        sample_weights,
        sample_sources,
        sample_ordinals,
        assignment_counts,
        seed_base_counts,
        slot_to_logical,
        owner_slots,
        redundant_slots,
        candidate_experts,
        slots_per_rank,
        ep_size,
        num_levels,
        level_size0,
        level_size1,
        level_size2,
        step,
        layer_seed,
        base_counts,
        assignment_loads,
        add_group_storage,
        add_assignment_storage,
        remove_group_storage,
        remove_assignment_storage,
        int_workspace,
        float_workspace);
    return {
        base_counts,
        assignment_loads,
        add_group_deltas,
        add_assignment_deltas,
        remove_group_deltas,
        remove_assignment_deltas};
}
