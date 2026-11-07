
import torch

from src.optimizers import SGDMomentum, AdamWCustom


def test_sgd_matches_pytorch():

    initial = torch.tensor([1.0, -2.0, 3.0])

    custom_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())

    custom = SGDMomentum(
        [custom_param],
        lr=0.01,
        momentum=0.9,
        weight_decay=0.01,
    )

    reference = torch.optim.SGD(
        [reference_param],
        lr=0.01,
        momentum=0.9,
        weight_decay=0.01,
    )

    gradients = [
        torch.tensor([0.5, -0.2, 0.1]),
        torch.tensor([0.3, 0.4, -0.5]),
        torch.tensor([-0.1, 0.2, 0.6]),
        torch.tensor([0.7, -0.3, 0.2]),
    ]

    for grad in gradients:
        custom.zero_grad()
        reference.zero_grad()

        custom_param.grad = grad.clone()
        reference_param.grad = grad.clone()

        custom.step()
        reference.step()

        assert torch.allclose(
            custom_param,
            reference_param,
            atol=1e-7,
            rtol=1e-6,
        )


def test_adamw_matches_pytorch():

    initial = torch.tensor([1.0, -2.0, 3.0])

    custom_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())

    custom = AdamWCustom(
        [custom_param],
        lr=0.001,
        weight_decay=0.01,
    )

    reference = torch.optim.AdamW(
        [reference_param],
        lr=0.001,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )

    gradients = [
        torch.tensor([0.5, -0.2, 0.1]),
        torch.tensor([0.3, 0.4, -0.5]),
        torch.tensor([-0.1, 0.2, 0.6]),
        torch.tensor([0.7, -0.3, 0.2]),
    ]

    for grad in gradients:
        custom.zero_grad()
        reference.zero_grad()

        custom_param.grad = grad.clone()
        reference_param.grad = grad.clone()

        custom.step()
        reference.step()

        assert torch.allclose(
            custom_param,
            reference_param,
            atol=1e-7,
            rtol=1e-6,
        )
