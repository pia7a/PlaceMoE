# Integrating PlaceMoE with a model-modified VeOmni fork

PlaceMoE separates model integration from MoE runtime integration. The VeOmni
host invokes an MoE provider through the versioned
`veomni.moe_runtime_bridges` entry point, while PlaceMoE discovers expert
modules through its public model-adapter API.

## Recommended workflow

For a fork whose changes are limited to model registration, generated model
code, checkpoint conversion, or a parallel plan:

1. use a released PlaceMoE commit as the branch base;
2. cherry-pick or rebase the model-only VeOmni commits onto that base;
3. retain the model's normal VeOmni registration and EP parallel plan;
4. enable PlaceMoE in the training YAML; and
5. run `placemoe doctor` before distributed training.

This avoids editing the user's model forward or training loop. It also keeps
the PlaceMoE host hooks reviewable as one versioned integration boundary.

An arbitrary upstream VeOmni checkout cannot support PlaceMoE through a
zero-touch import because VeOmni does not currently expose every required
dispatch, step-boundary, checkpoint, and backward-overlap hook. Do not use
`sitecustomize`, runtime monkey-patching, or an implicit fallback to emulate
these hooks.

## Automatic model support

PlaceMoE automatically supports the 2 expert representations used by VeOmni's
standard fused MoE interface:

- `gate_up_proj` and `down_proj`; or
- `gate_proj`, `up_proj`, and `down_proj`.

Each tensor must be stacked by local expert slot along its leading dimension,
and the expert module must expose `num_experts`. This contract covers the
validated Qwen3-VL and DeepSeek-V3 paths and applies equally to a new model that
uses the same VeOmni interface.

A different representation implements `MoEModelAdapter` and registers it once
from the model package:

```python
from placemoe import register_moe_model_adapter

register_moe_model_adapter(MyModelAdapter())
```

The adapter exposes expert parameters and normalized fused-kernel weights; it
does not contain placement, mapping, communication, or planner logic.

## Strict startup contract

PlaceMoE stops with an actionable error when:

- the host bridge API version is incompatible;
- no model adapter matches an expert module;
- expert slot dimensions are inconsistent;
- replica-gradient hooks cannot be registered; or
- the backward path does not execute every required hook.

Replica-gradient overlap never silently falls back to blocking
synchronization. This makes an incomplete model or host integration visible
before it can change performance or training semantics.

## Validation

Run on every node:

```bash
placemoe doctor --config configs/my_train.yaml
```

The doctor checks the bridge provider and API version together with the
software stack, topology-dependent calibration, paths, replica capacity, and
hot-update schedule. Then run a short 2-node job and compare the generated
layout and mapping artifacts with a known route fixture before measuring
end-to-end performance.
