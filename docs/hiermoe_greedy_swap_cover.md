# HierMoE greedy swap/cover planner

## Scope

The hiermoe_greedy_cover_p1 selector is a layer-mode, communication-only
placement planner. It starts from the current physical layout, evaluates owner
swaps and replica covers in the same round, accepts at most one strictly
improving steady-state action, and returns the exact token-to-physical-slot
mapping used to price that action.

If the starting layout is R2, rejecting every non-improving action makes the
steady-state communication model cost no worse than R2 under the same route
snapshot and cost model.

## Token mapping policy

For every logical token-expert route:

1. List every physical copy of that logical expert.
2. Compare copies by hierarchical distance from the token's source rank.
3. Choose a nearest copy.
4. If several copies are equally near, use a stable hash of token ordinal,
   logical expert, step, and layer to distribute routes among tied copies.

All duplicate occurrences of the same expert in one token use the same hash
and therefore the same physical copy. Different logical experts are mapped
independently. Consequently, a swap or cover can change routes only for the
one or two logical experts touched by that action.

The expert manager uses the same policy for runtime fallback mapping. The
planner-provided mapping is cached for the just-planned layer invocation, so
the action is executed with exactly the mapping that was scored.

## Exact marginal scoring

The baseline mapping is computed once. For every token and hierarchy level,
the planner stores an occupancy count for each destination group. A group
contributes one token copy to communication exactly when its occupancy is
positive.

An action is represented by kind, source slot, destination slot, lhs expert,
and rhs expert:

- A swap moves the owner copy of lhs and rhs.
- A cover adds a copy of lhs at the destination and evicts the destination
  replica rhs.

For only the affected expert routes, the planner computes old and new
destination groups and atomically applies all occupancy changes for a token:

    delta = 1[new occupancy > 0] - 1[old occupancy > 0]

This is the exact form of:

    cover gain = add gain - eviction loss + interaction

The interaction term is not estimated separately. It is captured by applying
the additions and removals together before testing the zero/nonzero occupancy
boundary. Swap scoring uses the same mechanism with two moved experts.

Every rank evaluates the same candidate rows for its local tokens. Candidate
group-count deltas are globally summed with reduce-scatter, so each rank
evaluates the exact global cost of one candidate shard. Compact cost,
peak-rank, and selected-dimension triples are then all-gathered. The baseline
row uses a separate all-reduce. Environments without a suitable process group
retain the exact full all-reduce fallback. The hierarchy model selects the
globally cheapest action and accepts it only on strict improvement.

## Empty-slot initialization

Empty slots are handled before steady-state replacement:

- only empty-cover candidates are generated;
- compatible covers are selected to fill available capacity in one
  initialization pass;
- the batch selector updates per-expert copy counts after every accepted
  cover, so the resulting layout cannot exceed
  train.hiermoe.greedy_max_copies_per_expert;
- owner slots are never evicted;
- initialization steps are reported separately from steady-state swaps and
  occupied covers.

This one-pass fill is intentionally an initialization policy, not a claim of
sequential greedy optimality. Exact sequential filling would require
re-scoring after every inserted replica and is substantially more expensive.
Following steady-state rounds restore the strict-improvement invariant.

## Efficient implementation

The NPU path uses a fused AscendC candidate scorer:

- replica_prepare builds sparse expert-to-token route tables once;
- one AI-core work stream evaluates each candidate's affected routes;
- nearest-copy choices are prepared once per candidate rather than once per
  token;
- the common single-expert move has a specialized occupancy-update path;
- distributed candidate counts use reduce-scatter instead of replicating the
  full reduced count matrix on every rank;
- the output contains compact hierarchy-count deltas, not full token maps.

After choosing a winner, the planner clones the baseline route map and updates
only routes whose logical expert is lhs or rhs. This incremental apply is
exactly checked against a complete remap in CPU and NPU tests.

For tokens T, top-k K, hierarchy levels L, candidate actions A, copy limit C,
and unique-token counts U(e), the work is:

- baseline mapping and occupancy: O(T * K * (C + L));
- fused scoring:
  O(sum_a (U(lhs_a) + U(rhs_a)) * (K + L) + A * C^2);
- one small baseline all-reduce, one sharded candidate reduce-scatter over
  A * sum_level(number of groups at level) counts, and one all-gather of
  3 * A scalar metrics;
- winner apply: O(T * K + T * C).

Candidate actions run in parallel across AI cores. Current fused limits are
16,384 local tokens, EP size 64, three hierarchy levels, and eight copies per
expert. Inputs outside those limits use the exact PyTorch fallback.

The production default is train.hiermoe.greedy_max_copies_per_expert=4.
The same cap is used by empty-slot initialization, steady-state candidate
generation, and runtime token remapping. Values from 1 through 8 are accepted
for workload-specific profiling.

## Reproducible planner benchmark

Run the saved-route benchmark on one NPU:

    python scripts/profile/benchmark_hiermoe_greedy_planner.py \
      --route-dir /workspace/output/hiermoe_p4_route_replay_20260720/routes \
      --layer 0 --rank 0

Under torchrun, each EP rank automatically loads its matching route file and
the benchmark includes the real sharded candidate collectives.
