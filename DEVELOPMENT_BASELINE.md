# HierMoE greedy swap/cover development baseline

Created on 2026-07-22 from:

`/home/tzq/npu_profile_outputs/hiermoe_exact_p1_optional_20260721/src`

The copy excludes generated or historical artifacts:

- Python, pytest, and Ruff caches
- `profile/runs/`
- `pretrain_runs/`
- `veomni/ops/platform/npu/hiermoe_planner_ops/build/`

The existing HierMoE planner extension binary is retained locally for compatibility,
but shared-library binaries are not tracked by Git. Planner and A2A benchmarks should
read existing profile inputs from their original directories instead of copying them
into this source tree.

The source snapshot did not include the `.agents/` instruction directory. It was
copied from `/home/tzq/VeOmni-0.1.11/.agents/` so the repository-local development
and profiling instructions referenced by `AGENTS.md` remain available.

Key source fingerprints before development:

| File | SHA-256 |
| --- | --- |
| `veomni/distributed/moe/hiermoe/expert_swap.py` | `7a34f7766cc38e87f8aaacea7c62d6c440dcfef22003c3c0ed28c805cb5afa74` |
| `veomni/distributed/moe/hiermoe/planner.py` | `e66f8ff39a0a66e92173098ef7d752f862fb662aadfedbf757e851b9ba894a07` |
| `veomni/distributed/moe/hiermoe/core_planner.py` | `c110181563ddb1131f488d4e9c3fecb138bc28daff989a641a9aecdf814162a9` |
| `veomni/distributed/moe/hiermoe/all_to_all.py` | `5c72943d8c189edc22bb7e2e38904fa0e1c9a9e10ca6ddd925a63df628264539` |
| `tests/distributed/test_hiermoe.py` | `8c26581d0bed74ba9f85eb43196af2b4962d308d0b563d21a991a682abfb717e` |

The host default Python is 3.9.9 and has no `pytest`; it is not a valid VeOmni
test environment. Correctness and NPU timing must be run in the existing Python
3.11/Ascend training environment.
