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

"""
Base Trainer class for distributed training.

This module provides the BaseTrainer class which serves as the foundation
for all trainer implementations. Subclasses can override specific methods
to customize training behavior.

Features:
    - Callback system for extensible training hooks
    - Distributed training support
    - Gradient accumulation
    - Checkpointing
"""

import json
import os
import time
import warnings
from abc import ABC
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.checkpoint import set_checkpoint_debug_enabled
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.modeling_outputs import ModelOutput

from ..arguments import VeOmniArguments, save_args
from ..checkpoint import CheckpointerBase
from ..data import (
    DistributedDataloader,
    build_dataloader,
    build_dataset,
)
from ..data.chat_template import ChatTemplate
from ..data.data_collator import DataCollator, MainCollator
from ..data.data_transform import build_data_transform
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.moe.runtime_bridge import configure_moe_runtime_bridge, get_moe_runtime_bridge
from ..distributed.offloading import build_activation_offloading_context
from ..distributed.parallel_state import get_parallel_state, init_parallel_state
from ..distributed.torch_parallelize import build_parallelize_model
from ..models import build_foundation_model, build_tokenizer
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..optim import build_lr_scheduler, build_optimizer
from ..utils import helper, logging
from ..utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from ..utils.full_timing_profiler import get_full_timing_profiler
from ..utils.loss_utils import count_loss_token, mean_global_loss
from ..utils.model_utils import pretty_print_trainable_parameters
from ..utils.physical_moe_load import bind_physical_moe_layer_keys, physical_moe_load_enabled
from .callbacks import (
    CheckpointerCallback,
    EnvironMeterCallback,
    EvaluateCallback,
    HFLoraCkptCallback,
    HuggingfaceCkptCallback,
    MoERouterMonitorCallback,
    ProfileTraceCallback,
    TqdmCallback,
    TrainerState,
    WandbTraceCallback,
)


logger = logging.get_logger(__name__)


_HIERMOE_DIAG_PHASES = os.environ.get("VEOMNI_HIERMOE_DIAG_PHASES", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _hiermoe_diag_phase(phase: str) -> None:
    if not _HIERMOE_DIAG_PHASES:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(
        f"HIERMOE_DIAG_PHASE rank={rank} phase={phase} monotonic={time.monotonic():.6f}",
        flush=True,
    )


def _scalar_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _resolve_total_train_steps(planned_steps: int) -> int:
    """Apply an optional run-wide step cap across epoch boundaries."""

    raw = os.environ.get("VEOMNI_TOTAL_MAX_STEPS", "").strip()
    if not raw:
        return planned_steps
    try:
        total_max_steps = int(raw)
    except ValueError as exc:
        raise ValueError(f"VEOMNI_TOTAL_MAX_STEPS must be an integer, got {raw!r}.") from exc
    if total_max_steps <= 0:
        raise ValueError(f"VEOMNI_TOTAL_MAX_STEPS must be positive, got {total_max_steps}.")
    return min(planned_steps, total_max_steps)


def _hiermoe_gradient_bytes_per_element(mixed_precision) -> int:
    dtype_name = (
        mixed_precision.reduce_dtype or mixed_precision.param_dtype or "float32"
        if mixed_precision.enable
        else "bfloat16"
    )
    dtype = getattr(torch, dtype_name or "float32")
    return torch.empty((), dtype=dtype).element_size()


def _collect_muon_kwargs(optimizer_cfg) -> Dict[str, Any]:
    """Pull Muon-specific hyperparameters out of ``OptimizerConfig``."""
    return {
        "lr": optimizer_cfg.muon_lr,
        "momentum": optimizer_cfg.muon_momentum,
        "nesterov": optimizer_cfg.muon_nesterov,
        "weight_decay": optimizer_cfg.muon_weight_decay,
        "ns_steps": optimizer_cfg.muon_ns_steps,
        "ns_coefficients": tuple(optimizer_cfg.muon_ns_coefficients),
        "eps": optimizer_cfg.muon_eps,
        "adjust_lr_fn": optimizer_cfg.muon_adjust_lr_fn,
    }


class BaseTrainer(Stateful, ABC):
    """
    Base trainer class for distributed model training.

    This class provides the core training infrastructure including:
    - Distributed initialization and parallelism setup
    - Model, optimizer, and scheduler initialization
    - Training step execution with gradient accumulation
    - Checkpointing and fault tolerance
    - Metrics logging

    Subclasses can override the following methods to customize behavior:
    - `post_init()`: Add custom initialization after setup
    - `forward_backward_step()`: Customize forward/backward logic
    - `train_step()`: Customize training step execution
    - `train()`: Train the model

    Callback Hooks:
        The trainer calls callback methods at various stages:
        - evaluate_callback: evaluation callback
        - trace_callback: tracing callback (meter, wandb, tqdm, profile)
        - checkpoint_callback: checkpointing callback
    """

    # Core configs
    args: VeOmniArguments
    device: torch.device

    # Data
    data_transform: Callable
    train_dataset: Dataset
    collate_fn: DataCollator
    train_dataloader: DistributedDataloader

    # Model
    model: PreTrainedModel = None
    model_config: PretrainedConfig = PretrainedConfig()
    tokenizer: PreTrainedTokenizerBase = None
    processor: ProcessorMixin = None
    chat_template: ChatTemplate = None
    model_assets: List[Any] = []

    # Training components
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None

    # Training context
    model_fwd_context: Any
    model_bwd_context: Any

    # Runtime metrics, controlled by trace_callback
    environ_meter: helper.EnvironMeter  # see in trace_callback.EnvironMeterCallback
    step_env_metrics: Dict[str, Any]  # mfu, flops, tokens, etc
    step_train_metrics: Dict[str, Any]  # loss, grad_norm, lr, etc

    # Checkpointer
    checkpointer: CheckpointerBase  # see in checkpoint_callback.CheckpointerCallback

    # Callback system
    state: TrainerState

    # Training states
    train_steps: int = 0  # total training steps
    total_train_steps: int = 0  # run-wide training steps across epochs
    start_epoch: int = 0  # start epoch
    start_step: int = 0  # start step

    def __init__(self, args: VeOmniArguments):
        """
        Initialize the trainer.

        Args:
            args: Global Arguments
                Should have attributes: model, data, train
                model: ModelArguments
                data: DataArguments
                train: TrainingArguments
        """

        self.args: VeOmniArguments = args
        self._setup()
        # build model
        self._build_model()
        # freeze module and print trainable parameters
        self._freeze_model_module()
        # build model assets (config, tokenizer, processor, chat_template)
        self._build_model_assets()
        # build dataset and dataloader
        self._build_data_transform()
        self._build_dataset()
        self._build_collate_fn()
        self._build_dataloader()

        # Parallelize model
        self._build_parallelized_model()
        # Build optimizer and lr scheduler
        self._build_optimizer()
        self._build_lr_scheduler()
        # Build training context
        self._build_training_context()
        # Initialize callbacks
        self._init_callbacks()

    def _setup(self):
        # log args
        logger.info_rank0(json.dumps(asdict(self.args), indent=2))

        # init distributed environment
        device_str = f"{get_device_type()}:{self.args.train.local_rank}"
        get_torch_device().set_device(device_str)
        self.device = torch.device(device_str)

        # Initialize distributed process group
        if not dist.is_initialized():
            dist.init_process_group(backend=get_dist_comm_backend())

        logger.info(f"Process rank: {self.args.train.global_rank}, world size: {self.args.train.world_size}")

        # Initialize parallel state
        init_parallel_state(
            dp_size=self.args.train.accelerator.dp_size,
            dp_replicate_size=self.args.train.accelerator.dp_replicate_size,
            dp_shard_size=self.args.train.accelerator.dp_shard_size,
            tp_size=self.args.train.accelerator.tp_size,
            pp_size=self.args.train.accelerator.pp_size,
            cp_size=self.args.train.accelerator.cp_size,
            ulysses_size=self.args.train.accelerator.ulysses_size,
            extra_parallel_sizes=self.args.train.accelerator.extra_parallel_sizes,
            extra_parallel_placement_innermost=self.args.train.accelerator.extra_parallel_placement_innermost,
            extra_parallel_names=self.args.train.accelerator.extra_parallel_names,
            dp_mode=self.args.train.accelerator.fsdp_config.fsdp_mode,
            async_enabled=self.args.train.accelerator.enable_async,
        )
        ps = get_parallel_state()
        mixed_precision = self.args.train.accelerator.fsdp_config.mixed_precision
        placemoe = self.args.train.hiermoe.placemoe
        if placemoe.enabled and placemoe.calibration.artifact:
            model_id = os.path.basename(
                os.path.normpath(str(self.args.model.model_path or self.args.model.config_path))
            )
            placemoe.calibration.expected_scope["model_id"] = model_id
        configure_moe_runtime_bridge(
            self.args.train.hiermoe,
            ep_group=ps.ep_group if getattr(ps, "ep_enabled", False) else None,
            ep_fsdp_size=ps.ep_fsdp_size if getattr(ps, "ep_enabled", False) else 1,
            activation_checkpointing_enabled=self.args.train.gradient_checkpointing.enable,
            fsdp_offload_enabled=self.args.train.accelerator.fsdp_config.offload,
            gradient_bytes_per_element=_hiermoe_gradient_bytes_per_element(mixed_precision),
        )

        # Set random seed
        helper.set_seed(self.args.train.seed, self.args.train.enable_full_determinism)

        # Enable high precision for bf16
        helper.enable_high_precision_for_bf16()

        # Enable third party logging
        if self.args.train.local_rank == 0:
            helper.enable_third_party_logging()

        # Save arguments
        if self.args.train.global_rank == 0:
            save_args(self.args, self.args.train.checkpoint.output_dir)

        # Gradient checkpointing debug
        set_checkpoint_debug_enabled(self.args.train.gradient_checkpointing.debug)

    def _build_model(self):
        logger.info_rank0("Build model")
        self.model = build_foundation_model(
            config_path=self.args.model.config_path,
            weights_path=self.args.model.model_path,
            torch_dtype="float32" if self.args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
            init_device=self.args.train.init_device,
            ops_implementation=self.args.model.ops_implementation,
            config_kwargs=self.args.model.model_config,
        )
        self.model_config = self.model.config

    def _setup_lora(self):
        """Wrap ``self.model`` with PEFT LoRA if ``lora_config`` is configured.

        Handles two cases:
        - Resume: ``lora_config["lora_adapter"]`` is set → use
          ``PeftModel.from_pretrained`` so the PEFT config is read from disk.
          Actual adapter *weights* are loaded later during parallelization.
        - Scratch: only ``rank``, ``alpha``, ``lora_modules`` are set →
          use ``get_peft_model``.
        """
        lora_config = self.args.model.lora_config
        if not bool(lora_config):
            return

        lora_adapter_path = lora_config.get("lora_adapter", None)
        if lora_adapter_path is not None:
            logger.info_rank0(f"Wrapping model with PeftModel from {lora_adapter_path}.")
            from peft import PeftModel

            # When init_device="meta" the base model params are meta tensors.
            # PeftModel.from_pretrained tries to copy loaded weights into the meta
            # adapter params — a no-op that PyTorch warns about.
            # Actual adapter weights are loaded later via adapter_path during parallelization.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="copying from a non-meta parameter")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_adapter_path,
                    is_trainable=lora_config.get("is_trainable", True),
                )
        else:
            logger.info_rank0("Initialising LoRA adapter from scratch.")
            from peft import LoraConfig, get_peft_model

            peft_cfg = LoraConfig(
                r=lora_config["rank"],
                lora_alpha=lora_config["alpha"],
                target_modules=lora_config["lora_modules"],
            )
            logger.info_rank0(f"LoraConfig: {peft_cfg.to_dict()}.")
            self.model = get_peft_model(self.model, peft_cfg)

        self.model.print_trainable_parameters()

    def _freeze_model_module(self):
        self._setup_lora()
        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info("VRAM usage after building model")

    def _build_model_assets(self):
        # model assets
        self.tokenizer = build_tokenizer(self.args.model.tokenizer_path)
        self.model_assets = [self.model_config, self.tokenizer]

    def _build_data_transform(self):
        self.data_transform = build_data_transform(
            self.args.data.data_type,
            tokenizer=self.tokenizer,
            max_seq_len=self.args.data.max_seq_len,
            text_keys=self.args.data.text_keys,
        )

    def _build_dataset(self):
        args: VeOmniArguments = self.args
        # Build dataset
        self.train_dataset = build_dataset(
            dataset_name=args.data.dataset_name,
            transform=self.data_transform,
            seed=args.train.seed,
            **asdict(args.data),
        )
        dataset_length = None if not hasattr(self.train_dataset, "__len__") else len(self.train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.accelerator.dp_size
        args.compute_train_steps(dataset_length)
        self.train_steps = args.train_steps

    def _build_collate_fn(self):
        seq_classification = self.args.data.data_type == "classification"
        pad_to_length = self.args.train.pad_to_length
        self.collate_fn = MainCollator(
            pad_to_length=pad_to_length,
            seq_classification=seq_classification,
        )

    def _build_dataloader(self):
        args: VeOmniArguments = self.args
        dataloader_kwargs = asdict(args.data.dataloader)
        dataloader_type = dataloader_kwargs.pop("type")
        self.train_dataloader = build_dataloader(
            dataloader_type=dataloader_type,
            dataset=self.train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train_steps,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz=args.train.dyn_bsz,
            dyn_bsz_runtime=args.train.dyn_bsz_runtime,
            dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
            seed=args.train.seed,
            collate_fn=self.collate_fn,
            **dataloader_kwargs,
        )

    def _build_parallelized_model(self):
        args: VeOmniArguments = self.args
        kwargs = {}
        cpu_load_param_name = None
        if hasattr(self.model, "get_parallel_plan"):
            cpu_load_param_name = getattr(self.model.get_parallel_plan(), "cpu_load_param_name", None)
        kwargs["cpu_load_param_name"] = cpu_load_param_name
        if bool(args.model.lora_config):
            lora_adapter_path = args.model.lora_config.get("lora_adapter", None)
            kwargs["adapter_path"] = lora_adapter_path
            kwargs["is_peft_model"] = True

        muon_expert_zero_comm = args.train.optimizer.type == "muon" and args.train.optimizer.muon_expert_zero_comm

        # Parallelize model
        self.model = build_parallelize_model(
            self.model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=args.train.gradient_checkpointing.enable,
            basic_modules=list(
                set(getattr(self.model, "_no_split_modules", None) or []) | set(args.model.basic_modules)
            ),
            enable_reentrant=args.train.gradient_checkpointing.enable_reentrant,
            enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
            enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
            broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
            max_load_broadcast_size=args.train.accelerator.fsdp_config.max_load_broadcast_size,
            muon_expert_zero_comm=muon_expert_zero_comm,
            **kwargs,
        )
        self.model.train()
        get_moe_runtime_bridge().bind_model(self.model)
        if physical_moe_load_enabled():
            attached = bind_physical_moe_layer_keys(self.model)
            if attached == 0:
                raise RuntimeError("Physical MoE load metrics were enabled but no expert modules were found.")
            logger.info_rank0(f"Physical MoE load metrics: registered {attached} expert layer(s).")

    def _build_optimizer(self):
        args: VeOmniArguments = self.args
        # Build optimizer
        self.optimizer = build_optimizer(
            self.model,
            lr=args.train.optimizer.lr,
            weight_decay=args.train.optimizer.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer.type,
            no_decay_modules=args.train.optimizer.no_decay_modules,
            no_decay_params=args.train.optimizer.no_decay_params,
            muon_kwargs=_collect_muon_kwargs(args.train.optimizer),
        )
        get_moe_runtime_bridge().bind_optimizer(self.optimizer)

    def _build_lr_scheduler(self):
        args: VeOmniArguments = self.args
        # Build lr scheduler
        planned_steps = args.train_steps * args.train.num_train_epochs
        self.total_train_steps = _resolve_total_train_steps(planned_steps)
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=self.total_train_steps,
            lr=args.train.optimizer.lr,
            lr_min=args.train.optimizer.lr_min,
            lr_decay_style=args.train.optimizer.lr_decay_style,
            lr_decay_ratio=args.train.optimizer.lr_decay_ratio,
            lr_warmup_ratio=args.train.optimizer.lr_warmup_ratio,
            lr_start=args.train.optimizer.lr_start,
        )

    def reached_total_train_steps(self) -> bool:
        """Return whether the run-wide training-step limit has been reached."""

        return self.total_train_steps > 0 and self.state.global_step >= self.total_train_steps

    def _build_training_context(self):
        """Build training context for distributed training."""
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.args.train.accelerator.offload_config.enable_activation,
            self.args.train.gradient_checkpointing.enable,
            self.args.train.accelerator.offload_config.activation_gpu_limit,
        )

    def _init_callbacks(self):
        """Initialize callbacks."""
        self.environ_meter_callback = EnvironMeterCallback(self)
        self.tqdm_callback = TqdmCallback(self)
        self.wandb_callback = WandbTraceCallback(self)
        self.profile_callback = ProfileTraceCallback(self)
        self.checkpointer_callback = CheckpointerCallback(self)
        if self.args.model.lora_config:
            self.hf_ckpt_callback = HFLoraCkptCallback(self)
        else:
            self.hf_ckpt_callback = HuggingfaceCkptCallback(self)
        self.evaluate_callback = EvaluateCallback(self)
        self.moe_monitor_callback = MoERouterMonitorCallback(self)
        ps = get_parallel_state()
        self.full_timing_profiler = get_full_timing_profiler(
            run_kind="pretrain",
            rank=self.args.train.global_rank,
            local_rank=self.args.train.local_rank,
            fsdp_size=getattr(ps, "fsdp_size", None),
            ep_size=getattr(ps, "ep_size", 1) if getattr(ps, "ep_enabled", False) else 1,
        )
        attached = self.full_timing_profiler.attach_model_hooks(self.model)
        if attached and self.args.train.global_rank == 0:
            logger.info_rank0(f"Full timing profiler: attached to {attached} module(s).")
        self.state = TrainerState()

    def _full_profile_range(self, section: str, metadata: dict[str, Any] | None = None):
        profiler = getattr(self, "full_timing_profiler", None)
        if profiler is None:
            return nullcontext()
        return profiler.cuda_range(section, metadata=metadata)

    def on_train_begin(self):
        self.environ_meter_callback.on_train_begin(self.state)
        self.tqdm_callback.on_train_begin(self.state)
        self.wandb_callback.on_train_begin(self.state)
        self.profile_callback.on_train_begin(self.state)
        self.checkpointer_callback.on_train_begin(self.state)
        self.hf_ckpt_callback.on_train_begin(self.state)
        self.evaluate_callback.on_train_begin(self.state)
        self.moe_monitor_callback.on_train_begin(self.state)

    def on_train_end(self):
        self.environ_meter_callback.on_train_end(self.state)
        self.tqdm_callback.on_train_end(self.state)
        self.wandb_callback.on_train_end(self.state)
        self.profile_callback.on_train_end(self.state)
        self.checkpointer_callback.on_train_end(self.state)
        self.hf_ckpt_callback.on_train_end(self.state)
        self.evaluate_callback.on_train_end(self.state)
        self.moe_monitor_callback.on_train_end(self.state)

    def on_epoch_begin(self):
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.tqdm_callback.on_epoch_begin(self.state)
        self.wandb_callback.on_epoch_begin(self.state)
        self.profile_callback.on_epoch_begin(self.state)
        self.checkpointer_callback.on_epoch_begin(self.state)
        self.hf_ckpt_callback.on_epoch_begin(self.state)
        self.evaluate_callback.on_epoch_begin(self.state)

    def on_epoch_end(self):
        self.environ_meter_callback.on_epoch_end(self.state)
        self.tqdm_callback.on_epoch_end(self.state)
        self.wandb_callback.on_epoch_end(self.state)
        self.profile_callback.on_epoch_end(self.state)
        self.checkpointer_callback.on_epoch_end(self.state)
        self.hf_ckpt_callback.on_epoch_end(self.state)
        self.evaluate_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches=None):
        get_moe_runtime_bridge().set_step(max(0, self.state.global_step - 1))
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.tqdm_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.wandb_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.profile_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpointer_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.hf_ckpt_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.evaluate_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self.run_hiermoe_expert_swap()
        get_moe_runtime_bridge().log_metrics(self.state.global_step)
        self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.tqdm_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.wandb_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.profile_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.hf_ckpt_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.evaluate_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.moe_monitor_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def run_hiermoe_expert_swap(self) -> None:
        hiermoe_config = self.args.train.hiermoe
        trainer_step = int(self.state.global_step)
        if getattr(self, "_hiermoe_last_swap_step", -1) == trainer_step:
            return
        placement_step = max(0, trainer_step - 1)
        with self._full_profile_range("hiermoe_expert_swap"):
            if hiermoe_config.enable and hiermoe_config.expert_swap:
                get_moe_runtime_bridge().run_step_update(placement_step)
        self._hiermoe_last_swap_step = trainer_step

    def hiermoe_layer_swap_forward(self):
        return get_moe_runtime_bridge().training_forward()

    @staticmethod
    def hiermoe_placement_disabled():
        return get_moe_runtime_bridge().placement_disabled()

    def sync_hiermoe_redundant_gradients(self) -> None:
        hiermoe_config = self.args.train.hiermoe
        if (
            hiermoe_config.enable
            and hiermoe_config.expert_swap
            and hiermoe_config.redundant_slot_increment_per_device > 0
        ):
            get_moe_runtime_bridge().sync_redundant_gradients()

    def preforward(self, micro_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess micro batches before forward pass.

        Tensors are moved to ``self.device`` non-blockingly. Nested dicts
        (e.g. ``multimodal_metadata`` emitted by ``PackingCollator``) are
        recursed so inner tensor values land on the device too; Python ints
        / lists / etc. pass through unchanged.
        """

        def _to_device(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return v.to(self.device, non_blocking=True)
            if isinstance(v, dict):
                return {k: _to_device(vv) for k, vv in v.items()}
            return v

        micro_batch = {k: _to_device(v) for k, v in micro_batch.items()}
        if getattr(self, "LOG_SAMPLE", True):
            helper.print_example(example=micro_batch, rank=self.args.train.local_rank)
            self.LOG_SAMPLE = False
        return micro_batch

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Postprocess model outputs after forward pass."""
        loss_dict: Dict[str, torch.Tensor] = mean_global_loss(
            outputs.loss, self.micro_batch_token_len, self.micro_batches_token_len
        )
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss, loss_dict

    def forward_backward_step(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _hiermoe_diag_phase("preforward_start")
        with self._full_profile_range("preforward"):
            micro_batch = self.preforward(micro_batch)
        _hiermoe_diag_phase("preforward_done")

        _hiermoe_diag_phase("forward_start")
        with self._full_profile_range("forward"):
            with (
                self.model_fwd_context,
                set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode),
                self.hiermoe_layer_swap_forward(),
            ):
                outputs: ModelOutput = self.model(**micro_batch, use_cache=False)
        _hiermoe_diag_phase("forward_done")

        loss: torch.Tensor
        loss_dict: Dict[str, torch.Tensor]
        with self._full_profile_range("loss"):
            loss, loss_dict = self.postforward(outputs, micro_batch)
        _hiermoe_diag_phase("loss_done")

        # Backward pass
        _hiermoe_diag_phase("backward_start")
        with self._full_profile_range("backward"):
            with self.model_bwd_context, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
                loss.backward()
        _hiermoe_diag_phase("backward_done")

        del micro_batch
        return loss, loss_dict

    def model_reshard(self, micro_step: int, num_micro_steps: int):
        """Reshard model after backward pass."""
        get_moe_runtime_bridge().configure_microstep(micro_step, num_micro_steps)
        args: VeOmniArguments = self.args
        if (
            args.train.accelerator.fsdp_config.fsdp_mode == "fsdp2"
            and not args.train.accelerator.fsdp_config.reshard_after_backward
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                self.model.set_reshard_after_backward(False)
            elif micro_step == num_micro_steps - 1:
                self.model.set_reshard_after_backward(True)

    def _configure_hsdp_allreduce(self, micro_step: int, num_micro_steps: int):
        args: VeOmniArguments = self.args
        if (
            args.train.accelerator.fsdp_config.fsdp_mode == "fsdp2"
            and args.train.accelerator.dp_replicate_size > 1
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                self.model.set_requires_all_reduce(False)
            elif micro_step == num_micro_steps - 1:
                self.model.set_requires_all_reduce(True)

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        args = self.args
        self.state.global_step += 1
        self.full_timing_profiler.begin_step(
            self.state.global_step,
            metadata={"run_kind": "pretrain", "trainer": type(self).__name__},
        )

        with self._full_profile_range("data_wait"):
            micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.on_step_begin(micro_batches=micro_batches)

        # Forward and backward for each micro batch
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # token num for fixed_ce_loss in postforward
        self.micro_batches_token_len = count_loss_token(micro_batches)
        num_micro_steps = len(micro_batches)
        # forward and backward pass with gradient_accumulationsteps
        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, num_micro_steps)
            self._configure_hsdp_allreduce(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss in postforward
            self.micro_batch_token_len = count_loss_token(micro_batch)
            with self._full_profile_range("micro_batch", {"micro_step": micro_step}):
                loss, loss_dict = self.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        self.sync_hiermoe_redundant_gradients()

        # Gradient clipping
        with self._full_profile_range("grad_clip"):
            grad_norm = veomni_clip_grad_norm(self.model, args.train.optimizer.max_grad_norm)

        # Optimizer and scheduler step
        with self._full_profile_range("optimizer_step"):
            self.optimizer.step()
        with self._full_profile_range("lr_scheduler"):
            self.lr_scheduler.step()
        with self._full_profile_range("zero_grad"):
            self.optimizer.zero_grad()

        self.run_hiermoe_expert_swap()
        self.full_timing_profiler.end_step(metadata={"loss": float(total_loss), "grad_norm": _scalar_float(grad_norm)})
        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def destroy_distributed(self):
        get_moe_runtime_bridge().shutdown()
        helper.empty_cache()
        dist.barrier()
        get_moe_runtime_bridge().destroy_process_groups()
        dist.destroy_process_group()

    def train(self):
        args: VeOmniArguments = self.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Total train steps: {self.total_train_steps}. "
            f"Start epoch: {self.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.start_epoch, args.train.num_train_epochs):
            if self.reached_total_train_steps():
                break
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)
            self.state.epoch = epoch

            self.on_epoch_begin()

            # Create a batch generator
            data_iterator = iter(self.train_dataloader)

            for _ in range(self.start_step, args.train_steps):
                if self.reached_total_train_steps():
                    break
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.start_step = 0

            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()

        synchronize()

        self.destroy_distributed()
