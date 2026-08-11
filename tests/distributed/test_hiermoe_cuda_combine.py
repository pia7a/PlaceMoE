# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import pytest
import torch

from veomni.distributed.moe.hiermoe.all_to_all import _index_add_dim0_cast_output


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("source_rows", "output_rows", "hidden_size"),
    [(0, 7, 16), (19, 7, 33), (257, 31, 2048)],
)
def test_cuda_segment_sum_matches_fp32_reference_and_backward(
    dtype: torch.dtype,
    source_rows: int,
    output_rows: int,
    hidden_size: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260805)
    source = torch.randn(
        (source_rows, hidden_size),
        dtype=dtype,
        device="cuda",
        generator=generator,
        requires_grad=True,
    )
    index = torch.randint(
        output_rows,
        (source_rows,),
        dtype=torch.long,
        device="cuda",
        generator=generator,
    )

    actual = _index_add_dim0_cast_output(source, index, output_rows)
    expected_fp32 = torch.zeros((output_rows, hidden_size), dtype=torch.float32, device="cuda")
    expected_fp32.index_add_(0, index, source.detach().float())
    expected = expected_fp32.to(dtype)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    grad_output = torch.randn(actual.shape, dtype=dtype, device="cuda", generator=generator)
    actual.backward(grad_output)
    torch.testing.assert_close(source.grad, grad_output.index_select(0, index), atol=0, rtol=0)


def test_cuda_segment_sum_normalizes_cpu_index_for_backward() -> None:
    source = torch.randn((13, 17), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    index = torch.arange(13, dtype=torch.long).remainder(5)

    output = _index_add_dim0_cast_output(source, index, 5)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    expected_grad = grad_output.index_select(0, index.to(device="cuda"))
    torch.testing.assert_close(source.grad, expected_grad, atol=0, rtol=0)


def test_cuda_segment_sum_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEOMNI_HIERMOE_CUDA_SEGMENT_SUM", "0")
    source = torch.randn((23, 17), dtype=torch.bfloat16, device="cuda")
    index = torch.arange(23, device="cuda").remainder(5)

    actual = _index_add_dim0_cast_output(source, index, 5)
    expected = torch.zeros((5, 17), dtype=torch.float32, device="cuda")
    expected.index_add_(0, index, source.float())

    torch.testing.assert_close(actual, expected.to(source.dtype), atol=2e-2, rtol=2e-2)


def test_cuda_segment_sum_rejects_out_of_range_index() -> None:
    source = torch.randn((2, 8), dtype=torch.bfloat16, device="cuda")
    index = torch.tensor([0, 2], dtype=torch.long, device="cuda")

    with pytest.raises(ValueError, match="outside"):
        _index_add_dim0_cast_output(source, index, 2)
