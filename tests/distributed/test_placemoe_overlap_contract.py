# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from types import SimpleNamespace

import pytest

from veomni.distributed.moe.hiermoe.expert_swap import ExpertSwapManager


class _HookHandle:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _UnsupportedParameter:
    register_post_accumulate_grad_hook = None


class _RegisteredParameter:
    def __init__(self) -> None:
        self.handle = _HookHandle()

    def register_post_accumulate_grad_hook(self, _hook):
        return self.handle


class _FailingParameter:
    @staticmethod
    def register_post_accumulate_grad_hook(_hook):
        raise RuntimeError("unsupported parameter wrapper")


def _manager() -> ExpertSwapManager:
    manager = object.__new__(ExpertSwapManager)
    manager.gradient_overlap_enabled = True
    manager._pipeline_grad_hook_handles = []
    manager._pipeline_grad_hook_params = set()
    return manager


def test_required_overlap_rejects_parameter_without_gradient_hook():
    manager = _manager()
    layer = SimpleNamespace(key="layers.0.experts", expert_parameters=(_UnsupportedParameter(),))

    with pytest.raises(RuntimeError, match="requires replica-gradient overlap"):
        manager._register_pipeline_gradient_hooks(layer)

    assert manager._pipeline_grad_hook_handles == []
    assert manager._pipeline_grad_hook_params == set()


def test_required_overlap_rolls_back_partial_hook_registration():
    manager = _manager()
    registered = _RegisteredParameter()
    failing = _FailingParameter()
    layer = SimpleNamespace(key="layers.0.experts", expert_parameters=(registered, failing))

    with pytest.raises(RuntimeError, match="Blocking synchronization is not selected automatically"):
        manager._register_pipeline_gradient_hooks(layer)

    assert registered.handle.removed is True
    assert manager._pipeline_grad_hook_handles == []
    assert manager._pipeline_grad_hook_params == set()


def test_required_overlap_rejects_a_backward_path_that_bypasses_registered_hooks(monkeypatch):
    manager = _manager()
    manager._pipeline_layer_order = ("layers.0.experts",)
    manager.layers = {
        "layers.0.experts": SimpleNamespace(expert_parameters=(object(), object(), object())),
    }
    manager._pipeline_grad_ready = defaultdict(set, {"layers.0.experts": {0, 1}})
    manager._pipeline_grad_submit_lock = SimpleNamespace(
        __enter__=lambda _self: None,
        __exit__=lambda _self, *_args: None,
    )
    monkeypatch.setattr(
        manager,
        "_replica_grad_schedule_for_layer",
        lambda _layer: SimpleNamespace(groups=(object(),)),
    )

    with pytest.raises(RuntimeError, match="did not observe all registered expert gradients"):
        manager._finish_pipeline_gradient_sync()
