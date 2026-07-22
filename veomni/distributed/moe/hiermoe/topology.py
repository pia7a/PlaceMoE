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

import os
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Hierarchy:
    """EP-local hierarchy description for HierMoE communication modeling."""

    ep_size: int
    group_sizes: tuple[int, ...]
    source: str
    local_world_size: int = 1

    @property
    def selected_dim(self) -> int:
        return max(1, len(self.group_sizes))


def infer_hierarchy(
    ep_size: int,
    topology: str = "auto",
    hierarchy_group_sizes: Sequence[int] | None = None,
) -> Hierarchy:
    if ep_size < 1:
        raise ValueError(f"ep_size must be positive, got {ep_size}.")

    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    explicit_sizes = tuple(int(size) for size in (hierarchy_group_sizes or ()))
    if explicit_sizes:
        for size in explicit_sizes:
            if size <= 0:
                raise ValueError(f"Hierarchy group size must be positive, got {size}.")
        if explicit_sizes[-1] != ep_size and ep_size % explicit_sizes[-1] != 0:
            raise ValueError(f"The last hierarchy group size ({explicit_sizes[-1]}) must divide ep_size ({ep_size}).")
        return Hierarchy(
            ep_size=ep_size,
            group_sizes=explicit_sizes,
            source="config",
            local_world_size=min(local_world_size, ep_size),
        )

    if topology != "auto":
        raise ValueError(f"Unsupported HierMoE topology {topology!r}; supported value is 'auto'.")

    if local_world_size <= 1 or ep_size <= 1:
        return Hierarchy(
            ep_size=ep_size,
            group_sizes=(ep_size,),
            source="auto-single",
            local_world_size=min(local_world_size, ep_size),
        )

    intra_size = min(local_world_size, ep_size)
    if ep_size % intra_size != 0:
        return Hierarchy(
            ep_size=ep_size,
            group_sizes=(ep_size,),
            source="auto-flat",
            local_world_size=min(local_world_size, ep_size),
        )

    num_nodes = ep_size // intra_size
    if num_nodes >= 8 and ep_size % (2 * intra_size) == 0:
        return Hierarchy(
            ep_size=ep_size,
            group_sizes=(intra_size, 2 * intra_size, ep_size),
            source="auto-3d",
            local_world_size=intra_size,
        )

    return Hierarchy(
        ep_size=ep_size,
        group_sizes=(intra_size, ep_size),
        source="auto-2d",
        local_world_size=intra_size,
    )
