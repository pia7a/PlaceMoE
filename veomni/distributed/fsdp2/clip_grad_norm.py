import math
import os
import time
from typing import List

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from ...utils.device import get_device_type
from ...utils.logging import get_logger
from ..parallel_state import get_parallel_state


logger = get_logger(__name__)


_HIERMOE_DIAG_PHASES = os.environ.get("VEOMNI_HIERMOE_DIAG_PHASES", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _clip_diag_phase(phase: str) -> None:
    if not _HIERMOE_DIAG_PHASES:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(
        f"HIERMOE_CLIP_DIAG rank={rank} phase={phase} monotonic={time.monotonic():.6f}",
        flush=True,
    )


def clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    if hasattr(model, "_extra_parallel_param_groups"):
        return extra_parallel_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    if getattr(model, "_fsdp_cpu_offload_enabled", False):
        return _cpu_offload_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
        foreach=foreach,
    )
    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()
    return grad_norm


@torch.no_grad()
def _cpu_offload_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    ps = get_parallel_state()
    params = [p for p in model.parameters() if p.grad is not None]
    total_norm_or_pth_sum = _fsdp2_reduce_group(
        params=params,
        norm_type=norm_type,
        reduce_groups=[("fsdp", ps.fsdp_group)],
    )
    total_norm = _finalize_total_norm(total_norm_or_pth_sum, norm_type)
    _raise_if_nonfinite(total_norm, norm_type, error_if_nonfinite)

    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach=foreach)

    return total_norm


@torch.no_grad()
def extra_parallel_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    ExtraParallel-aware gradient clipping for composable FSDP2:

    - Compute local norms for non-ExtraParallel and ExtraParallel parameter groups separately.
    - For finite p: sum p-th powers across the appropriate groups, then take 1/p.
      • non-ExtraParallel: all-reduce over FSDP group.
      • ExtraParallel: all-reduce over Para-FSDP (e.g. ep_fsdp, emb_fsdp) group, then over Para (e.g. ep, emb) group.
    - For inf-norm: take elementwise MAX with the same reduction groups (MAX).
    - Use a single global clip coefficient for both groups.
    """
    ps = get_parallel_state()
    from ..moe.hiermoe import get_hiermoe_redundant_grad_norm_masks

    grad_row_masks = get_hiermoe_redundant_grad_norm_masks()
    fsdp_group = ps.fsdp_group
    extra_parallel_group = {
        para: ps.extra_parallel_group(para) if ps.extra_parallel_enabled(para) else None
        for para in ps.extra_parallel_names
    }
    # For Para (e.g. ep, emb) params sharded by FSDP2 along hidden dimension
    extra_parallel_fsdp_group = {
        para: ps.extra_parallel_fsdp_device_mesh[para][f"{para}_fsdp"].get_group()
        if ps.extra_parallel_enabled(para) and ps.extra_parallel_fsdp_device_mesh[para] is not None
        else None
        for para in ps.extra_parallel_names
    }

    # Build param groups for ExtraParallel params and non-ExtraParallel params (filter out params without grads)
    extra_parallel_params = {
        para: [p for p in model._extra_parallel_param_groups.get(para, []) if p.grad is not None]
        for para in ps.extra_parallel_names
    }
    non_extra_parallel_params: List[torch.nn.Parameter] = [
        p for p in model._extra_parallel_param_groups.get("non_extra_parallel", []) if p.grad is not None
    ]

    # Compute and reduce non-ExtraParallel
    non_extra_parallel_total = _fsdp2_reduce_group(
        params=non_extra_parallel_params,
        norm_type=norm_type,
        reduce_groups=[("fsdp", fsdp_group)],
        grad_row_masks=grad_row_masks,
    )
    logger.debug_rank0(f"non_extra_parallel total grad norm: {non_extra_parallel_total}")

    for para in ps.extra_parallel_names:
        logger.debug_rank0(
            f"{para}_params reduces groups: {extra_parallel_fsdp_group[para]=}, {extra_parallel_group[para]=}"
        )

    # Compute and reduce ExtraParallel: first across para_fsdp (e.g. ep_fsdp, emb_fsdp), then across para (e.g. ep, emb)
    extra_parallel_total = {
        para: torch.tensor(0.0, device=torch.device(get_device_type()), dtype=torch.float32)
        for para in ps.extra_parallel_names
    }
    for para in ps.extra_parallel_names:
        if len(extra_parallel_params[para]) > 0:
            para_total = _fsdp2_reduce_group(
                params=extra_parallel_params[para],
                norm_type=norm_type,
                reduce_groups=[
                    (f"{para}_fsdp", extra_parallel_fsdp_group[para]),
                    (f"{para}", extra_parallel_group[para]),
                ],
                grad_row_masks=grad_row_masks,
            )
            extra_parallel_total[para] = para_total
            logger.debug_rank0(f"{para} total grad norm: {para_total}")

    total_norm = _combine_reduced_norm_totals(non_extra_parallel_total, extra_parallel_total, norm_type)

    _raise_if_nonfinite(total_norm, norm_type, error_if_nonfinite)

    # Apply the same clip coefficient to both groups
    for para in ps.extra_parallel_names:
        torch.nn.utils.clip_grads_with_norm_(extra_parallel_params[para], max_norm, total_norm, foreach=foreach)
    torch.nn.utils.clip_grads_with_norm_(non_extra_parallel_params, max_norm, total_norm, foreach=foreach)

    return total_norm


def _combine_reduced_norm_totals(
    non_extra_parallel_total: torch.Tensor,
    extra_parallel_totals: dict[str, torch.Tensor],
    norm_type: float,
) -> torch.Tensor:
    if math.isinf(norm_type):
        return torch.stack((non_extra_parallel_total, *extra_parallel_totals.values())).amax()
    return _finalize_total_norm(non_extra_parallel_total + sum(extra_parallel_totals.values()), norm_type)


def _finalize_total_norm(total_norm_or_pth_sum: torch.Tensor, norm_type: float) -> torch.Tensor:
    if math.isinf(norm_type):
        return total_norm_or_pth_sum
    return total_norm_or_pth_sum ** (1.0 / float(norm_type))


def _raise_if_nonfinite(total_norm: torch.Tensor, norm_type: float, error_if_nonfinite: bool) -> None:
    if not error_if_nonfinite:
        return
    if bool((~torch.isfinite(total_norm)).item()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, "
            "so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )


# compute local sum of param gard norm
def _local_grad_for_norm(
    param: torch.nn.Parameter,
    grad_row_masks: dict[int, torch.Tensor] | None,
) -> torch.Tensor | None:
    grad = param.grad
    if grad is None:
        return None
    grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
    mask = None if grad_row_masks is None else grad_row_masks.get(id(param))
    if mask is not None:
        if grad_local.ndim == 0 or int(mask.numel()) != int(grad_local.shape[0]):
            raise RuntimeError(
                "HierMoE gradient-norm mask does not match the local expert-slot dimension: "
                f"mask={tuple(mask.shape)}, grad={tuple(grad_local.shape)}."
            )
        # Sparse redundant layouts may place no canonical owner for a layer on
        # this EP rank. Do not feed an empty masked NPU tensor to foreach-norm:
        # some accelerator backends do not make progress on that input. This
        # rank contributes the additive identity to the subsequent all-reduce.
        if not bool(mask.any()):
            return None
        # Keep the local tensor shape static. Boolean indexing produces a
        # rank-dependent first dimension for partial-capacity layouts, which
        # does not make progress in the NPU foreach-norm path. Zeroing
        # non-owner rows is mathematically equivalent for every p-norm while
        # preserving the original tensor shape on every rank.
        if not bool(mask.all()):
            mask_device = mask.to(device=grad_local.device, dtype=grad_local.dtype)
            mask_shape = (int(mask_device.numel()),) + (1,) * (grad_local.ndim - 1)
            grad_local = grad_local * mask_device.reshape(mask_shape)
    return grad_local.detach().to(torch.float32)


def _local_pth_sum(
    params: List[torch.nn.Parameter],
    p: float,
    grad_row_masks: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    reduce_device = torch.device(get_device_type())
    res = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
    grads_local: list[torch.Tensor] = []
    partial_mask_count = 0
    with torch.no_grad():
        for param in params:
            grad = param.grad
            if grad is None:
                continue
            grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
            mask = None if grad_row_masks is None else grad_row_masks.get(id(param))
            if mask is None or bool(mask.all()):
                grads_local.append(grad_local.detach().to(torch.float32))
                continue
            if grad_local.ndim == 0 or int(mask.numel()) != int(grad_local.shape[0]):
                raise RuntimeError(
                    "HierMoE gradient-norm mask does not match the local expert-slot dimension: "
                    f"mask={tuple(mask.shape)}, grad={tuple(grad_local.shape)}."
                )
            if not bool(mask.any()):
                continue

            # A partial owner mask is specific to sparse redundant layouts. Do
            # not place these tensors in accelerator foreach/vector-norm
            # kernels: both combinations fail to make progress for the large
            # expert tensor shape on some NPU runtimes. Use elementary
            # elementwise power plus row reduction instead. This computes the
            # exact p-th-power contribution while keeping the tensor shape
            # identical on every rank.
            if partial_mask_count == 0:
                _clip_diag_phase("partial_mask_pth_start")
            partial_mask_count += 1
            grad_rows = grad_local.detach().to(torch.float32).reshape(int(grad_local.shape[0]), -1)
            if p == 2.0:
                row_pth_sums = torch.sum(torch.square(grad_rows), dim=1)
            else:
                row_pth_sums = torch.sum(torch.pow(torch.abs(grad_rows), p), dim=1)
            mask_device = mask.to(device=row_pth_sums.device, dtype=row_pth_sums.dtype)
            res = res + torch.sum(row_pth_sums * mask_device).to(reduce_device)

    if partial_mask_count:
        _clip_diag_phase("partial_mask_pth_done")
    if not grads_local:
        return res
    with torch.no_grad():
        _clip_diag_phase("foreach_pth_start")
        grouped_grads_local = _group_tensors_by_device_and_dtype([grads_local])
        for (device, _), ([device_grads_local], _) in grouped_grads_local.items():
            if _has_foreach_support(device_grads_local, device) or _device_has_foreach_support(device):
                out = torch._foreach_pow_(torch._foreach_norm(device_grads_local, p), p)
                res += torch.sum(torch.stack(out)).to(reduce_device)
            else:
                for grad_local in device_grads_local:
                    gn = torch.norm(grad_local, p=p)
                    res = res + (gn**p).to(reduce_device)
        _clip_diag_phase("foreach_pth_done")
    return res


def _local_max(
    params: List[torch.nn.Parameter],
    grad_row_masks: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    reduce_device = torch.device(get_device_type())
    mx = None
    for param in params:
        grad_local = _local_grad_for_norm(param, grad_row_masks)
        if grad_local is None or grad_local.numel() == 0:
            continue
        if mx is None:
            mx = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
        gn = torch.max(torch.abs(grad_local))
        mx = torch.maximum(mx, gn.to(reduce_device))
    if mx is None:
        mx = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
    return mx


def _fsdp2_reduce_group(
    params: List[torch.nn.Parameter],
    norm_type: float,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
    grad_row_masks: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute local group statistic and reduce over provided groups.

    For finite p, returns the globally-reduced sum of p-th powers (not the final norm).
    For inf, returns the globally-reduced max.
    """
    if math.isinf(norm_type):
        val = _local_max(params, grad_row_masks)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        _clip_diag_phase("local_pth_sum_start")
        val = _local_pth_sum(params, p, grad_row_masks)
        _clip_diag_phase("local_pth_sum_done")
        logger.debug_rank0(f"local total grad norm: {val}. ProcessGroups to sum {reduce_groups}")
        for name, group in reduce_groups:
            if group is not None:
                _clip_diag_phase(f"all_reduce_{name}_start")
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
                _clip_diag_phase(f"all_reduce_{name}_done")
                logger.debug_rank0(f"After Sum of group {name} total grad norm is {val}")
        return val
