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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from veomni.distributed.moe import runtime_bridge


class _FakeBridge:
    api_version = runtime_bridge.MOE_RUNTIME_BRIDGE_API_VERSION

    def __init__(self, name: str = "placemoe") -> None:
        self.name = name
        self.calls = []

    def configure(self, config, **kwargs):
        self.calls.append(("configure", config, kwargs))
        return "configured-state"

    def maybe_expand_expert_slots(self, model, ep_size):
        self.calls.append(("expand", model, ep_size))

    def bind_model(self, model):
        self.calls.append(("model", model))

    def bind_optimizer(self, optimizer):
        self.calls.append(("optimizer", optimizer))

    def set_step(self, step):
        self.calls.append(("step", step))

    def training_forward(self):
        return nullcontext()

    def placement_disabled(self):
        return nullcontext()

    def configure_microstep(self, micro_step, num_micro_steps):
        self.calls.append(("microstep", micro_step, num_micro_steps))

    def sync_redundant_gradients(self):
        self.calls.append(("sync",))

    def run_step_update(self, step):
        self.calls.append(("update", step))
        return "updated"

    def log_metrics(self, step):
        self.calls.append(("metrics", step))

    def shutdown(self):
        self.calls.append(("shutdown",))

    def destroy_process_groups(self):
        self.calls.append(("destroy",))


@pytest.fixture(autouse=True)
def _reset_bridge():
    runtime_bridge.reset_moe_runtime_bridge_for_test()
    yield
    runtime_bridge.reset_moe_runtime_bridge_for_test()


def test_canonical_config_selects_and_publishes_placemoe_bridge(monkeypatch):
    bridge = _FakeBridge()
    monkeypatch.setattr(
        runtime_bridge, "_load_entry_point_bridge", lambda name: bridge if name == "placemoe" else None
    )
    config = SimpleNamespace(placemoe=SimpleNamespace(enabled=True))

    state = runtime_bridge.configure_moe_runtime_bridge(
        config,
        ep_group="group",
        ep_fsdp_size=1,
        activation_checkpointing_enabled=False,
        fsdp_offload_enabled=False,
        gradient_bytes_per_element=4,
    )

    assert state == "configured-state"
    assert runtime_bridge.get_moe_runtime_bridge() is bridge
    assert bridge.calls == [
        (
            "configure",
            config,
            {
                "ep_group": "group",
                "ep_fsdp_size": 1,
                "activation_checkpointing_enabled": False,
                "fsdp_offload_enabled": False,
                "gradient_bytes_per_element": 4,
            },
        )
    ]


def test_bridge_api_version_mismatch_fails_before_configuration(monkeypatch):
    bridge = _FakeBridge()
    bridge.api_version += 1
    monkeypatch.setattr(runtime_bridge, "_load_entry_point_bridge", lambda _name: bridge)
    config = SimpleNamespace(placemoe=SimpleNamespace(enabled=True))

    with pytest.raises(RuntimeError, match="API version"):
        runtime_bridge.configure_moe_runtime_bridge(config)

    assert bridge.calls == []


def test_unconfigured_bridge_access_fails_explicitly():
    with pytest.raises(RuntimeError, match="has not been configured"):
        runtime_bridge.get_moe_runtime_bridge()
