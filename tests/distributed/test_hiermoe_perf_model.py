# Copyright 2026 Bytedance Ltd. and/or its affiliates

import torch

from veomni.distributed.moe.hiermoe import perf_model as perf_model_module
from veomni.distributed.moe.hiermoe.perf_model import _fit_peer_link


def test_startup_peer_probe_passes_a_p2p_op_list(monkeypatch):
    observed = []

    class Request:
        @staticmethod
        def wait():
            return None

    def batch_isend_irecv(ops):
        assert isinstance(ops, list)
        observed.append(ops)
        return [Request() for _ in ops]

    def median_probe(operation, **_kwargs):
        operation()
        return 1.0

    monkeypatch.setattr(perf_model_module.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(perf_model_module.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(perf_model_module.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(perf_model_module.dist, "get_process_group_ranks", lambda _group: [0, 1])
    monkeypatch.setattr(perf_model_module.dist, "P2POp", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr(perf_model_module.dist, "batch_isend_irecv", batch_isend_irecv)
    monkeypatch.setattr(perf_model_module, "_median_probe", median_probe)

    fitted = _fit_peer_link(
        object(),
        device=torch.device("cpu"),
        local_world_size=2,
        intra=True,
        payload_sizes=(16, 32),
        warmup=0,
        repeats=1,
    )

    assert fitted is not None
    assert len(observed) == 2
    assert all(len(ops) == 2 for ops in observed)
