import torch

from veomni.optim.lr_scheduler import build_lr_scheduler


def test_zero_learning_rate_uses_constant_zero_schedule() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.0)
    scheduler = build_lr_scheduler(
        optimizer,
        train_steps=20,
        lr=0.0,
        lr_decay_style="cosine",
        lr_warmup_ratio=0.1,
        lr_min=1e-7,
    )

    for _ in range(20):
        optimizer.step()
        scheduler.step()
        assert scheduler.get_last_lr() == [0.0]
