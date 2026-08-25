# PlaceMoE

PlaceMoE is an optional MoE runtime for Ascend NPU training. It uses profiled
token routes to jointly choose the number and physical layout of expert copies
and the source-aware mapping from logical expert requests to those copies. The
runtime performs hierarchical token-deduplicated communication and overlaps
replica-gradient synchronization with backward computation.

PlaceMoE does not modify the router output or the logical model. It only changes
where each logical expert is materialized and which physical copy serves a
request. The current production validation covers `fused_npu`; unsupported
kernel paths fail explicitly rather than silently disabling gradient overlap.

## Model compatibility

A model that already runs with VeOmni EP normally needs no PlaceMoE-specific
registration. PlaceMoE detects the two stacked expert representations used by
VeOmni:

- fused `gate_up_proj` plus `down_proj`; and
- separate `gate_proj`, `up_proj`, and `down_proj`.

For another representation, implement `placemoe.model_adapter.MoEModelAdapter`
and call `register_moe_model_adapter`. The adapter exposes the expert-indexed
parameters and their fused-kernel weight view; the planner and trainer remain
model independent.

## Configuration

Add the following block to a working VeOmni EP configuration. This example uses
2 nodes with 8 ranks per node. Artifact paths may be shared storage or identical
local paths on every node.

```yaml
train:
  accelerator:
    ep_size: 16
    dp_shard_size: 16
    fsdp_config:
      fsdp_mode: fsdp2
      offload: false
  hiermoe:
    hierarchy_group_sizes: [8, 16]
    redundant_slot_increment_per_device: 1
    placemoe:
      enabled: true
      base_directory: /shared/placemoe/qwen3vl_ep16
      runtime_perf_model: runtime_perf_model.json
      calibration:
        artifact: planner_calibration.json
        require_scope: true
        expected_scope:
          model_id: Qwen3-VL-30B-A3B-Instruct
          ep_size: 16
          ranks_per_node: 8
          hierarchy_group_sizes: [8, 16]
      hot_update:
        enabled: true
        layout_interval_steps: 100
        mapping_interval_steps: 100
        last_update_step: 1000
        work_root: hot_update
        failure_policy: raise
      resources:
        workers: 48
        candidate_workers: 4
        worker_threads: 1
```

`layout_interval_steps` and `mapping_interval_steps` are independent. A value of
0 disables that update. A mapping-only update replaces the dispatch lookup table
without moving expert state. A layout update also migrates expert parameters and
optimizer states, so it is normally less frequent. When both are due, one full
layout-and-mapping update is performed. `last_update_step: 0` disables all
periodic updates while retaining an optional `initial_artifact`.

`failure_policy: raise` is recommended for validation and production rollout.
Use `continue` only when continuing with the current valid layout is explicitly
preferred over failing the training job.

## Calibration and launch

PlaceMoE has two reusable calibration artifacts:

1. `runtime_perf_model` measures the cluster topology and EP configuration. It
   can be reused until the hardware, topology, communication stack, dtype, or EP
   configuration changes.
2. `calibration.artifact` fits model and execution costs from a short default-
   layout training run. It can be reused while the model and execution
   configuration remain unchanged.

Run `prepare` concurrently on every node. It validates cached artifacts, reuses
matching files, and creates only missing or invalid stages. The short model
calibration uses 2 warm-up steps, 1 fitting step, and 2 held-out validation steps
by default.

```bash
# Node 0
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
uv run placemoe prepare \
  --config configs/qwen3vl_ep16_placemoe.yaml \
  --entrypoint tasks/train_vlm.py

# Node 1: use the same command with NODE_RANK=1.
```

Use `--force-runtime` or `--force-model` to rebuild one stage. Before training,
run the deployment checks on every node:

```bash
uv run placemoe doctor --config configs/qwen3vl_ep16_placemoe.yaml
```

Then launch the normal VeOmni entrypoint on every node:

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
scripts/placemoe/launch_npu.sh \
  tasks/train_vlm.py configs/qwen3vl_ep16_placemoe.yaml
```

Change only `NODE_RANK` on the other node. The launcher runs `placemoe doctor`
before `torchrun`; set `PLACEMOE_SKIP_PREFLIGHT=1` only if the same configuration
was already checked in the current environment.

## Runtime lifecycle

At each configured interval, rank 0 writes a complete token-route snapshot and
launches the CPU planner asynchronously. Training continues with the current
valid pair. At a step boundary, the runtime broadcasts the candidate, validates
it on every rank, migrates state if the layout changed, and atomically installs
the new layout and mapping. Checkpoints include the physical layout and mapping
so that resume either restores the same state or reports an incompatible
topology explicitly.

The canonical planner is `scripts/profile/plan_placemoe.py`. It uses the same
topology-general algorithm for different EP sizes; compatibility wrappers are
not part of the training interface.

## Troubleshooting

- A process waiting after `torchrun` usually means that not all nodes used the
  same `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, or configuration file.
- Calibration scope failures mean the artifact belongs to another model or
  topology. Run `prepare`; do not edit the scope manually.
- `performance_model_schema` warnings indicate an old runtime artifact without
  measured state-migration or gradient-synchronization costs. Re-run runtime
  calibration before performance evaluation.
- PlaceMoE requires one complete EP group, `ep_size == NNODES *
  NPROC_PER_NODE`, and `ep_fsdp_size == 1`. FSDP CPU offload is unsupported
  while physical expert migration is enabled.

## Further reading

- [Container quick start](../usage/placemoe_container_quickstart.md) and its
  [Chinese version](../usage/placemoe_container_quickstart_zh.md)
- [Pre-training and configuration reference](../usage/placemoe_pretraining.md)
  and its [Chinese version](../usage/placemoe_pretraining_zh.md)
- [Validated Ascend image distribution](../usage/placemoe_image_distribution.md)
  and its [Chinese version](../usage/placemoe_image_distribution_zh.md)
- [Integration with a model-modified VeOmni fork](../usage/placemoe_veomni_bridge.md)
  and its [Chinese version](../usage/placemoe_veomni_bridge_zh.md)
