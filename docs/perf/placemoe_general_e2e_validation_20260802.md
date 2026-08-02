# PlaceMoE General Planner E2E Validation (2026-08-02)

## Scope

This minimal validation uses the paper's EP32 Qwen3-VL/ShareGPT4V case:
32 NPUs, 48 MoE layers, approximately 16K input tokens per rank, 20 training
steps, and steady-state statistics from steps 11--20. Existing VeOmni and
Replication measurements are reused; only the current topology-general
PlaceMoE planner and runtime are rerun.

The source worktree diff used for the run has SHA-256
`22ece22fa378a6dd00f4c3f956a136b291ab18112e639acdabf03e62ecb7408d`.

## Fresh planning

The planner reuses the existing steps 0--3 route capture and runs with 48
layer processes, 4 candidate threads per layer, and 1 numerical-library thread
per worker. The default topology-general affinity-community proposals are
enabled; legacy structured and hyperedge proposals are disabled.

- Planner wall time: 168.188 s.
- Mean independent layer time: 147.935 s.
- Sum of independent layer times: 7100.897 s.
- Held-out joint cost: 5556.006 ms.
- Mirrored-Replication held-out cost: 7271.245 ms.
- Held-out improvement: 1.309x.
- All 48 selected strategies are `community_block_*`.
- The generated artifact is E2E eligible.

Artifact hashes:

- Layout: `1917ec1bc777ebbc992fdcb89580e6e050027517cdcd636a7760877e37412f56`.
- Report: `012952c0a4122581267441abc183b3e4e2af4121ee2c9b8329beef0ca0406379`.

The layout hash was verified on all 4 nodes before execution and is recorded
again in the E2E summary.

## E2E result

| Method | E2E (ms) | A2A (ms) | Expert compute (ms) |
|---|---:|---:|---:|
| VeOmni, reused canonical run | 31017.165 | 24630.138 | 1162.795 |
| Replication, reused canonical run | 16281.943 | 5704.716 | 1020.293 |
| General PlaceMoE, fresh run | 13983.421 | 4127.170 | 1101.101 |

The fresh PlaceMoE run completed successfully on all 4 nodes, observed all 32
MoE ranks, loaded the expected layout hash, and produced 10 steady-state
samples. Its E2E standard deviation is 45.704 ms (CV 0.33%).

Relative to the reused canonical runs, general PlaceMoE achieves:

- 2.218x E2E and 5.968x A2A speedup over VeOmni.
- 1.164x E2E and 1.382x A2A speedup over Replication.
- 1.130x E2E and 1.403x A2A speedup over the pre-repair canonical generic
  PlaceMoE run.

The paper's EP32 ShareGPT4V PlaceMoE result is 13784.331 ms E2E and 3598.649
ms A2A. The fresh general implementation reaches 13983.421 ms E2E, within
1.4% of the paper result, while its A2A remains 14.7% higher. Using the paper's
VeOmni measurement, the fresh result corresponds to 2.295x E2E speedup versus
the reported 2.329x.

## Conclusion

The current topology-general algorithm is effective in the actual runtime.
Its held-out advantage transfers to both A2A and end-to-end training, and it
nearly reproduces the paper's EP32 E2E result without legacy structured
candidates. The remaining gap is concentrated in A2A rather than overall E2E;
it does not invalidate the implementation but is the next optimization target
if exact component-level reproduction is required.
