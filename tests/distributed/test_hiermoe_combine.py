import torch

from veomni.distributed.moe.hiermoe.all_to_all import _aggregate_weighted_outputs


def test_combine_aggregation_handles_different_source_token_counts() -> None:
    weighted_outputs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    source_token_indices = torch.tensor([7, 7, 1, 8, 8])

    combined, output_splits = _aggregate_weighted_outputs(
        weighted_outputs,
        source_token_indices,
        split_sizes=[2, 3],
    )

    assert output_splits == [1, 2]
    torch.testing.assert_close(combined, torch.tensor([[3.0], [3.0], [9.0]]))
