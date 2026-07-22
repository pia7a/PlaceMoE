#include <torch/extension.h>

#include "function.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("replica_prepare", &hiermoe_replica_prepare_npu, "Exact HierMoE packed replica routes");
    module.def("replica_score", &hiermoe_replica_score_npu, "Exact HierMoE replica candidate scoring");
    module.def("dual_map", &hiermoe_dual_map_npu, "CoRe-MoE dual-layout route mapping");
    module.def("quota_map", &hiermoe_quota_map_npu, "CoRe-MoE quota-aware dual-layout route mapping");
    module.def("swap_search", &hiermoe_swap_search_npu, "CoRe-MoE device-resident swap search");
    module.def("swap_select", &hiermoe_swap_select_npu, "CoRe-MoE aggregate-stat swap selection");
    module.def(
        "swap_select_with_stats",
        &hiermoe_swap_select_with_stats_npu,
        "CoRe-MoE aggregate-stat swap selection with final sampled counts");
    module.def("replica_match", &hiermoe_replica_match_npu, "CoRe-MoE one-shot replica slot matching");
    module.def("replica_project", &hiermoe_replica_project_npu, "CoRe-MoE fixed-summary replica projection");
    module.def("quota_policy", &hiermoe_quota_policy_npu, "CoRe-MoE exact quota policy construction");
}

TORCH_LIBRARY_FRAGMENT(veomni, module)
{
    module.def(
        "hiermoe_replica_apply(Tensor route_indices, Tensor token_counts, Tensor flat_logical, "
        "Tensor(a!) route_ranks, Tensor route_scores, Tensor(b!) minimum_scores, Tensor(c!) tie_count, "
        "Tensor(d!) tied_rank_order, Tensor route_hashes, Tensor(e!) token_group_counts, "
        "Tensor logical_expert, Tensor destination_rank, int num_levels, int level_size0, "
        "int level_size1, int level_size2, int top_k) -> ()");
}

TORCH_LIBRARY_IMPL(veomni, PrivateUse1, module)
{
    module.impl("hiermoe_replica_apply", &hiermoe_replica_apply_npu);
}
