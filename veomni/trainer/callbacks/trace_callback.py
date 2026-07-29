# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from tqdm import trange

from ...distributed.parallel_state import get_parallel_state
from ...utils import helper
from ...utils.dist_utils import all_reduce
from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


class MoERouterMonitorCallback(Callback):
    """Monitors MoE expert load distribution and logs heatmaps to wandb.

    Activation is gated only by ``moe_load_balance_monitor_interval > 0``; the
    monitor itself does not require wandb. Logging to wandb is gated by
    ``wandb.enable`` and ``global_rank == 0``.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.monitor = None
        self.monitor_jsonl_path: Path | None = None
        self.timing_jsonl_path: Path | None = None
        self.validator_jsonl_path: Path | None = None

        args: "VeOmniArguments" = self.trainer.args
        monitor_dir = os.environ.get("VERL_MOE_MONITOR_DIR")
        if monitor_dir and args.train.global_rank == 0:
            self.monitor_jsonl_path = Path(monitor_dir) / "moe_monitor_rank0.jsonl"

        timing_dir = os.environ.get("VERL_MOE_TIMING_DIR")
        if timing_dir:
            self.timing_jsonl_path = Path(timing_dir) / f"moe_timing_rank{args.train.global_rank}.jsonl"

        validator_dir = (
            os.environ.get("VEOMNI_MOE_VALIDATOR_DIR") if _env_flag("VEOMNI_MOE_VALIDATOR_ENABLE") else None
        )
        if validator_dir is None and _env_flag("VEOMNI_MOE_VALIDATOR_ENABLE"):
            profile_dir = os.environ.get("VERL_MOE_PROFILE_DIR")
            if profile_dir:
                validator_dir = str(Path(profile_dir) / "moe_validator")
        if validator_dir:
            self.validator_jsonl_path = Path(validator_dir) / f"moe_validator_rank{args.train.global_rank}.jsonl"

        if args.train.moe_load_balance_monitor_interval <= 0:
            logger.info_rank0("MoE router monitor disabled (moe_load_balance_monitor_interval=0).")
            return

        config = self.trainer.model_config
        num_experts = self._resolve_num_experts(config)
        if num_experts is None:
            logger.warning_rank0(
                "moe_load_balance_monitor_interval > 0 but model config has no 'num_experts' "
                "or 'text_config.num_experts'. "
                "MoE router monitor not activated."
            )
            return

        from ...utils.moe_monitor import MoERouterMonitor, set_active_monitor

        # Process groups are read lazily in on_train_begin once the device
        # mesh is guaranteed to be initialized.
        self.monitor = MoERouterMonitor(num_experts=num_experts)
        set_active_monitor(self.monitor)
        ps = get_parallel_state()
        logger.info_rank0(
            f"MoE router monitor created: num_experts={num_experts}, "
            f"interval={args.train.moe_load_balance_monitor_interval}, "
            f"ep_size={ps.ep_size if ps.ep_enabled else 1}"
        )

    @staticmethod
    def _resolve_num_experts(config: Any) -> int | None:
        for candidate in (config, getattr(config, "text_config", None)):
            if candidate is None:
                continue
            value = getattr(candidate, "num_experts", None)
            if value is not None:
                return int(value)
        return None

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        if self.monitor is None:
            return
        from ...utils.moe_monitor import attach_moe_router_monitor

        # fsdp_group is the dp_sp mesh dim — exactly the set of ranks that
        # hold distinct token slices. EP is intentionally not in this group;
        # see MoERouterMonitor.__init__ docstring.
        ps = get_parallel_state()
        self.monitor.dp_group = ps.fsdp_group
        self.monitor.ep_group = ps.ep_group if ps.ep_enabled else None

        attached = attach_moe_router_monitor(self.trainer.model, self.monitor)
        if attached == 0:
            logger.warning_rank0(
                "MoE router monitor: no recognized router modules found in the model. "
                "Disabling monitor. To add support for a new router class, register an "
                "extractor in veomni/utils/moe_monitor.py (see ROUTER_EXTRACTORS)."
            )
            from ...utils.moe_monitor import set_active_monitor

            set_active_monitor(None)
            self.monitor = None
        else:
            logger.info_rank0(f"MoE router monitor: attached to {attached} router module(s).")

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        interval = max(1, int(args.train.moe_load_balance_monitor_interval))
        if state.global_step % interval != 0:
            return

        # compute_metrics runs an all-reduce across EP/DP groups, so every rank
        # must call it — but only rank 0 logs.
        metrics: dict[str, Any] = {}
        if self.monitor is not None:
            metrics = self.monitor.compute_metrics(current_step=state.global_step)
            self._write_monitor_payload(state)
        self._write_timing_payload(state)
        self._write_validator_payload(state)

        if self.monitor is None or not metrics or args.train.global_rank != 0 or not args.train.wandb.enable:
            return

        import wandb

        wandb_metrics = {}
        for k, v in metrics.items():
            if k.endswith("expert_load_heatmap"):
                start, end = self.monitor._last_step_range
                wandb_metrics[k] = wandb.Image(v, caption=f"Steps {start}-{end}")
            else:
                wandb_metrics[k] = v
        wandb.log(wandb_metrics, step=state.global_step)

        start, end = self.monitor._last_step_range
        logger.info_rank0(
            f"Step {state.global_step}: uploaded MoE load balance heatmap "
            f"(steps {start}-{end}), "
            f"max_vio max={metrics['moe/max_vio/max']:.4f} avg={metrics['moe/max_vio/avg']:.4f}, "
            f"min_vio max={metrics['moe/min_vio/max']:.4f} avg={metrics['moe/min_vio/avg']:.4f}, "
            f"avg_vio max={metrics['moe/avg_vio/max']:.4f} avg={metrics['moe/avg_vio/avg']:.4f}."
        )

    def _write_monitor_payload(self, state: TrainerState) -> None:
        if self.monitor_jsonl_path is None or self.monitor is None:
            return

        payload = self.monitor.get_last_structured_payload()
        if not payload:
            return

        args: "VeOmniArguments" = self.trainer.args
        ps = get_parallel_state()
        payload.update(
            {
                "step": int(state.global_step),
                "rank": int(args.train.global_rank),
                "attached_routers": int(getattr(self.monitor, "_attached_count", 0)),
                "expert_parallel_size": int(ps.ep_size if ps.ep_enabled else 1),
                "fsdp_size": int(ps.fsdp_size),
            }
        )
        self.monitor_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.monitor_jsonl_path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_timing_payload(self, state: TrainerState) -> None:
        if self.timing_jsonl_path is None:
            return

        payload: dict[str, Any] = {}
        try:
            from ...distributed.moe.timing import flush_moe_timing_spans

            span_payload = flush_moe_timing_spans()
            if span_payload:
                payload.update(
                    {
                        "step": int(state.global_step),
                        "num_records": 0,
                        **span_payload,
                        "note": "Accelerator-event span timings cover EP MoE forward/backward components.",
                    }
                )
        except Exception as exc:
            logger.warning_rank0(f"MoE span timing payload flush unavailable: {exc}")

        try:
            from ...ops.kernels.moe.group_gemm import flush_moe_timing_payload

            kernel_payload = flush_moe_timing_payload(current_step=state.global_step)
            if kernel_payload:
                payload.update(kernel_payload)
        except Exception as exc:
            if not payload:
                logger.warning_rank0(f"MoE timing payload flush unavailable: {exc}")
                self.timing_jsonl_path = None
                return

        if not payload:
            return

        args: "VeOmniArguments" = self.trainer.args
        payload.update({"step": int(state.global_step), "rank": int(args.train.global_rank)})
        self.timing_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.timing_jsonl_path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_validator_payload(self, state: TrainerState) -> None:
        if self.validator_jsonl_path is None:
            return

        try:
            from ...distributed.moe.validation import flush_moe_validation_records

            payload = flush_moe_validation_records(current_step=state.global_step)
        except Exception as exc:
            logger.warning_rank0(f"MoE validation payload flush unavailable: {exc}")
            self.validator_jsonl_path = None
            return

        if not payload:
            return

        args: "VeOmniArguments" = self.trainer.args
        payload.update({"step": int(state.global_step), "rank": int(args.train.global_rank)})
        self.validator_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.validator_jsonl_path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.monitor is not None:
            from ...utils.moe_monitor import set_active_monitor

            set_active_monitor(None)
            self.monitor = None
            logger.info_rank0("MoE router monitor disabled.")


class WandbTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0 and args.train.wandb.enable:
            from dataclasses import asdict

            import wandb

            wandb.init(
                project=args.train.wandb.project,
                name=args.train.wandb.name,
                id=args.train.wandb.id,
                resume="allow" if args.train.wandb.id else None,
                config={**asdict(args.model), **asdict(args.data), **asdict(args.train)},
            )

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args

        if args.train.global_rank == 0 and args.train.wandb.enable:
            import wandb

            wandb.log(self.trainer.step_env_metrics, step=state.global_step)


class ProfileTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            self.profiler = helper.create_profiler(
                start_step=args.train.profile.start_step,
                end_step=args.train.profile.end_step,
                trace_dir=args.train.profile.trace_dir,
                record_shapes=args.train.profile.record_shapes,
                profile_memory=args.train.profile.profile_memory,
                with_stack=args.train.profile.with_stack,
                with_modules=args.train.profile.with_modules,
                global_rank=args.train.global_rank,
            )
            self.profiler.start()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            if state.global_step <= args.train.profile.end_step:
                self.profiler.step()

            if state.global_step == args.train.profile.end_step:
                self.profiler.stop()


class EnvironMeterCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)

        args: "VeOmniArguments" = self.trainer.args
        profile_dir = os.environ.get("VERL_MOE_PROFILE_DIR")
        self.env_metrics_jsonl_path = (
            Path(profile_dir) / "env_metrics" / "env_metrics_rank0.jsonl"
            if profile_dir and args.train.global_rank == 0
            else None
        )
        self.trainer.environ_meter = helper.EnvironMeter(
            config=trainer.model_config,
            global_batch_size=args.train.global_batch_size,
            empty_cache_steps=args.train.empty_cache_steps,
            enable_multisource=args.data.enable_multisource,
            dataloader=trainer.train_dataloader,
            data_path=args.data.train_path,
            gc_steps=args.train.gc_steps,
        )

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        for micro_batch in micro_batches:
            self.trainer.environ_meter.add(micro_batch)
        self.start_time = time.time()

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        delta_time = time.time() - self.start_time
        step_env_metrics = self.trainer.environ_meter.step(delta_time, global_step=state.global_step)

        step_train_metrics = {
            "total_loss": loss,
        }
        step_train_metrics.update(loss_dict)
        step_train_metrics["grad_norm"] = grad_norm

        # gather training_step_info from all ranks
        step_train_metrics = {
            f"training/{k}": all_reduce(v, group=get_parallel_state().fsdp_group)
            for k, v in step_train_metrics.items()
        }

        if self.trainer.lr_scheduler is not None:
            lr = max(self.trainer.lr_scheduler.get_last_lr())
            step_train_metrics["training/lr"] = lr

        step_env_metrics.update(step_train_metrics)
        hiermoe_config = self.trainer.args.train.hiermoe
        step_env_metrics["hiermoe/enable"] = int(bool(hiermoe_config.enable))
        step_env_metrics["hiermoe/expert_swap_interval"] = int(hiermoe_config.expert_swap_interval)
        step_env_metrics["hiermoe/expert_swap_max_pairs_per_layer"] = int(
            hiermoe_config.expert_swap_max_pairs_per_layer
        )
        if hiermoe_config.enable and state.global_step % int(hiermoe_config.log_interval) == 0:
            try:
                from ...distributed.moe.hiermoe import flush_hiermoe_metrics

                step_env_metrics.update(flush_hiermoe_metrics())
            except Exception as exc:
                logger.warning_rank0(f"Failed to flush HierMoE metrics: {exc}")

        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = step_env_metrics
        self._write_env_metrics(state, step_env_metrics)

    def _write_env_metrics(self, state: TrainerState, metrics: Dict[str, Any]) -> None:
        if self.env_metrics_jsonl_path is None:
            return

        payload = {"step": int(state.global_step), **metrics}
        self.env_metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.env_metrics_jsonl_path.open("a", encoding="utf-8") as writer:
            writer.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=lambda value: value.item() if hasattr(value, "item") else str(value),
                )
                + "\n"
            )


class TqdmCallback(Callback):
    def on_epoch_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        self.data_loader_tqdm = trange(
            args.train_steps,
            desc=f"Epoch {state.epoch + 1}/{args.train.num_train_epochs}",
            total=args.train_steps,
            initial=self.trainer.start_step,
            disable=args.train.local_rank != 0,
        )

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        self.data_loader_tqdm.close()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        postfix = ", ".join(f"{k.split('/', 1)[-1]}: {v:.2f}" for k, v in self.trainer.step_train_metrics.items())
        self.data_loader_tqdm.set_postfix_str(postfix)
        self.data_loader_tqdm.update()
