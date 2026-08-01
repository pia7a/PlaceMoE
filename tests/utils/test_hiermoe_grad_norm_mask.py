import torch

from veomni.distributed.fsdp2.clip_grad_norm import _local_grad_for_norm, _local_pth_sum


def test_fully_masked_expert_gradient_is_skipped() -> None:
    param = torch.nn.Parameter(torch.ones((3, 2), dtype=torch.float32))
    param.grad = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    result = _local_grad_for_norm(
        param,
        {id(param): torch.zeros((3,), dtype=torch.bool)},
    )

    assert result is None


def test_partially_masked_expert_gradient_zeros_non_owner_rows() -> None:
    param = torch.nn.Parameter(torch.ones((3, 2), dtype=torch.float32))
    param.grad = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    result = _local_grad_for_norm(
        param,
        {id(param): torch.tensor([True, False, True])},
    )

    torch.testing.assert_close(
        result,
        torch.tensor([[0.0, 1.0], [0.0, 0.0], [4.0, 5.0]], dtype=torch.float32),
    )


def test_fully_selected_expert_gradient_keeps_all_rows() -> None:
    param = torch.nn.Parameter(torch.ones((3, 2), dtype=torch.float32))
    param.grad = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    result = _local_grad_for_norm(
        param,
        {id(param): torch.ones((3,), dtype=torch.bool)},
    )

    torch.testing.assert_close(result, param.grad)


def test_partial_owner_pth_sum_matches_masked_reference() -> None:
    param = torch.nn.Parameter(torch.ones((3, 2), dtype=torch.float32))
    param.grad = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    mask = torch.tensor([True, False, True])

    result = _local_pth_sum([param], 2.0, {id(param): mask})
    reference = torch.sum(torch.square(param.grad[mask])).to(result.device)

    torch.testing.assert_close(result, reference)


if __name__ == "__main__":
    test_fully_masked_expert_gradient_is_skipped()
    test_partially_masked_expert_gradient_zeros_non_owner_rows()
    test_fully_selected_expert_gradient_keeps_all_rows()
    test_partial_owner_pth_sum_matches_masked_reference()
