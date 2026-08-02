# PlaceMoE planner A2A root-cause analysis (2026-08-02)

## Scope

This investigation compares the submitted paper abstraction, the canonical
modular planner, and the paper-era `structured_degree2` / `recursive_classifier`
implementations. All diagnostic comparisons use the same optimize routes
(steps 0--2) and held-out routes (step 3); no training timing is inferred from
the route evaluator.

## Result

The held-out evaluator is not the source of the regression. On identical route
captures it reproduces the paper-era plans and ranks their A2A improvements in
the same direction as runtime. The regression came from narrowing the proposal
set while extracting the paper abstraction into the modular optimizer.

The paper's calibrated pairwise objective remains a valid proposal heuristic,
but pairwise co-selection counts cannot exactly represent a top-k token's
destination-group union. Consequently, exact-route selection can recover the
best plan only if the proposal set contains layouts and mappings with the
required group structure.

## Paper abstraction versus executed proposals

| Planner step | Paper abstraction | Paper-era implementation | Extracted modular implementation before repair |
| --- | --- | --- | --- |
| Partition score | Calibrated communication reuse plus peak compute time | Scale-normalized affinity/load surrogate, followed by exact-route selection | Kept only the calibrated refinement |
| Initial mapping after a new layout | Demand-ordered mapping over copies in the new `L` | Rebuilt a fresh mapping | Carried the preceding mapping after round 0 |
| Mapping proposals | Calibrated communication--compute update | Fixed normalized tradeoffs at 8, 32, and 128 | Kept only the calibrated update, whose effective normalized compute weight is much smaller |
| Alternation state | Retain the lowest-cost pair generated across rounds | Projected the global exact-cost incumbent | Projected only the current round's incumbent |
| Four-node, two-copy placement | Abstract hierarchical affinity partitioning | Also enumerated degree-2 node-overlap libraries and group-coherent LUTs | Omitted the structured proposal family |
| Final selection | Replay complete profiled routes with the calibrated cost | Exact complete-route replay | Exact complete-route replay |

The last row is why held-out route replay was useful diagnostically: it ranks
the historical plans correctly. The regression occurs in the preceding rows,
before exact selection sees a complete candidate.

## EP32: structured overlap

Qwen3-VL / ShareGPT4V, EP32, one extra copy per expert:

| Planner on the same fresh routes | Held-out A2A (ms) | Held-out compute (ms) | Held-out joint cost (ms) | Mean destination nodes |
| --- | ---: | ---: | ---: | ---: |
| Modular calibrated planner | 6009.692 | 1193.961 | 7203.653 | 2.0694 |
| Structured node placement + modular mapping | 5237.601 | 849.547 | 6087.147 | 2.3483 |
| Structured node placement + group-coherent mapping | 4686.501 | 869.506 | 5556.008 | 1.6719 |
| Repaired canonical CLI | 4686.501 | 869.506 | 5556.008 | 1.6719 |

The structured node library accounts for 58.4% of the A2A gap and all of the
compute improvement. The group-coherent mapping accounts for the remaining
41.6% of the A2A gap, trading a small amount of compute balance for much lower
communication.

The reason is structural. In the applicable four-node, two-copy case,
`structured_degree2` enumerates balanced overlapping node libraries and maps a
whole source-node/expert-class group to a shared destination node. The generic
coordinate update moves one `(source rank, expert)` entry at a time and can be
trapped before those jointly beneficial moves are formed.

## EP64: normalized proposals and alternation state

Qwen3-VL / Tulu-3, EP64, all 48 MoE layers:

| Planner on the same fresh routes | Held-out A2A (ms) | Held-out compute (ms) | Held-out joint cost (ms) |
| --- | ---: | ---: | ---: |
| Modular calibrated planner before repair | 9898.466 | 1343.670 | 11242.136 |
| Paper-era `recursive_classifier` | 8766.573 | 982.015 | 9748.589 |
| Repaired canonical CLI | 8766.573 | 982.015 | 9748.589 |

The repaired CLI reduces held-out A2A, compute, and joint cost by 11.44%,
26.92%, and 13.29%, respectively, relative to the extracted planner before
repair. An automated comparison finds zero differences across all 48 winning
restart IDs and all optimize/held-out communication, compute, total, locality,
and peak-load metrics.

Qwen3-VL / Tulu-3, EP64, layer 7 provides an exact staged comparison:

| Stage | Optimize cost (ms) | Held-out cost (ms) |
| --- | ---: | ---: |
| Pre-fix modular planner | 694.847 | 230.101 |
| Normalized placement, but only carried mapping | 661.698 | 228.754 |
| Fresh mapping under each new layout | 617.240 | 205.555 |
| Normalized mapping proposals + monotonic incumbent | 596.599 | 198.367 |
| Paper-era `recursive_classifier` | 596.599 | 198.367 |

The final modular result matches the paper-era result to displayed precision.
After isolating the calibrated and normalized proposal state machines, the
canonical CLI again selects `placemoe_normalized_p2_c0` with exactly the same
layer-7 optimize and held-out costs.
A direct stage comparison additionally establishes that:

- uniform copy demand and affinity are bitwise identical;
- for the winning restart, the first two normalized node/rank placements and
  fresh mappings are bitwise identical;
- normalized mapping proposals at weights 8, 32, and 128 are bitwise identical;
- the final divergence was caused by projecting the current round's mapping
  instead of the exact-cost best-so-far incumbent into the next round.

Across the first 12 EP64 layers, the old normalized placement plus fresh
mapping reduces held-out joint cost from 2600.698 to 2281.874 ms; the old
mapping refinement further reduces it to 2259.930 ms. Thus, 93.6% of that
staged gap is recovered before mapping refinement.

One final compatibility detail is numerically observable. The first modular
port vectorized the normalized pair exchange and maintained group affinities
incrementally. Although its formula is mathematically equivalent to the
paper-era scalar loop, floating-point accumulation and tie-breaking changed a
few node assignments in 9 of 48 layers. The compatibility branch therefore
retains the historical scalar pair order and recomputes affinity-to-group sums
after each exchange. A staged layer-0 comparison then matches all 3 rounds
bitwise, including placement, initial mapping, normalized mapping proposals,
and projected copy statistics.

## Topology-general community proposal

The four-node structured result exposed a useful optimization granularity, not
a four-node requirement. The new default proposal applies the same principle
without a degree-2 template:

1. use an affinity partition to define balanced expert communities;
2. place successive copies of each community through balanced node
   permutations, then map abstract nodes to physical nodes by source locality;
3. jointly choose the destination node of each source-node/community block
   using exact token community-mask unions and calibrated projected compute;
4. run generic rank placement and fine-grained mapping refinement; and
5. retain the complete pair only when exact route replay selects it.

Source rows and source-node combinations use bounded beams. Partial replica
budgets are supported when an allocation preserves whole communities;
allocations that split a community automatically use generic placement. The
proposal therefore does not depend on four nodes, two copies, or a hard-coded
overlap graph.

On EP32 Qwen3-VL / ShareGPT4V, the community proposal wins all 48 layers:

| Proposal set | Held-out A2A (ms) | Held-out compute (ms) | Held-out joint cost (ms) |
| --- | ---: | ---: | ---: |
| Generic calibrated/normalized proposals | 5534.331 | 814.831 | 6349.162 |
| Legacy `structured_degree2` | 4686.501 | 869.506 | 5556.008 |
| Topology-general community proposal | 4686.499 | 869.506 | 5556.006 |

The community and legacy layouts are identical in 47 of 48 layers. Their
compute cost and destination-rank metrics match in all layers; the remaining
A2A difference is 0.001945 ms in favor of the community proposal. Its planner
wall time is 814.5 s, compared with 1035.9 s for the legacy structured-enabled
search and 701.0 s for generic-only search.

On EP64 Qwen3-VL / Tulu-3, the community proposal wins 29 of 48 layers while
the normalized generic proposal wins the other 19:

| Proposal set | Held-out A2A (ms) | Held-out compute (ms) | Held-out joint cost (ms) |
| --- | ---: | ---: | ---: |
| Generic calibrated/normalized proposals | 8766.573 | 982.015 | 9748.589 |
| Generic plus community proposals | 8328.225 | 1054.594 | 9382.819 |

The community proposals reduce A2A by 5.00% and joint cost by 3.75%, while
accepting 7.39% more compute cost. This is the intended calibrated
communication--compute tradeoff rather than an A2A-only result. On held-out
routes, 27 layers improve, 19 retain the generic winner exactly, and 2 show
small per-layer regressions; the aggregate held-out result improves.

## Repair

The canonical planner now keeps the paper path and broadens candidate
generation without changing final selection:

1. evaluate calibrated and scale-normalized placement as independent proposal
   branches for every configured restart;
2. make generic placement restarts independent of deduplicated logical
   replica-allocation candidates;
3. include topology-general community placement and block-mapping proposals
   by default, with bounded search and generic fallback;
4. in the calibrated paper branch, retain both the mapping carried by movable
   copies and a fresh demand-ordered mapping under every new layout;
5. in the normalized compatibility branch, start from a fresh mapping and
   evaluate fixed normalized communication--compute tradeoffs using the
   historical deterministic pair-refinement order; and
6. propagate each branch's exact-cost best-so-far mapping into its next
   alternation round; and
7. retain `structured_degree2` only as an explicit legacy diagnostic.

Every complete candidate is still selected using exact replay of the profiled
token routes. The historical token-KMeans hyperedge planner remains opt-in and
is not part of the canonical path.

## Verification

- EP32 full-layer replay selects the topology-general community proposal in
  all 48 layers and matches the paper-era structured result to 0.001945 ms.
- EP64 full-layer replay has zero per-layer metric or winner-restart
  differences from the paper-era recursive report when community proposals
  are disabled; enabling them reduces aggregate held-out joint cost by 3.75%.
- Focused assertions cover normalized partition retention, normalized mapping
  communication--compute tradeoffs, 4- and 8-node community placement,
  partial replica budgets, bounded source-row search, and copy separation.
- All changed Python files pass bytecode compilation and `git diff --check`.

The validation image does not include `pytest`, so the focused assertions were
executed directly in that image rather than through the pytest runner.

## Evidence files

The generated diagnostic layouts and reports are intentionally ignored build
artifacts. The principal report names are:

- `diagnostic_ep32_share16_structured_20260802_report.json`
- `diagnostic_ep32_share16_structured_node_only_20260802_report.json`
- `diagnostic_ep32_share16_repaired_full_20260802_report.json`
- `diagnostic_ep64_tulu16_legacy_recursive_20260802_report.json`
- `diagnostic_ep64_tulu16_legacy_recursive_nomap_l12_20260802_report.json`
- `diagnostic_ep64_tulu16_repaired_incumbent_l7_20260802_report.json`
- `diagnostic_ep64_tulu16_repaired_branches_l7_20260802_report.json`
- `diagnostic_ep64_tulu16_repaired_compat_full_20260802_report.json`
- `diagnostic_ep32_community_block_full_20260802_report.json`
- `diagnostic_ep64_community_block_full_20260802_report.json`
