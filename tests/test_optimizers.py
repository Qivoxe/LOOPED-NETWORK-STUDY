
import torch
import pytest

from src.optimizers import SGDMomentum, AdamWCustom


@pytest.mark.parametrize(
    "optimizer_class,kwargs",
    [
        (SGDMomentum, {"lr": 0.01, "momentum": 0.9}),
        (AdamWCustom, {"lr": 0.001}),
    ],
)
def test_optimizer_updates_parameters(optimizer_class, kwargs):

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    optimizer = optimizer_class([parameter], **kwargs)

    initial = parameter.detach().clone()

    loss = parameter.square().sum()
    loss.backward()

    optimizer.step()

    assert not torch.equal(initial, parameter)


def test_sgd_momentum_state():

    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer = SGDMomentum(
        [parameter],
        lr=0.1,
        momentum=0.9,
    )

    parameter.grad = torch.tensor([2.0])
    optimizer.step()

    state = optimizer.state[parameter]

    assert "momentum_buffer" in state
    assert torch.allclose(
        state["momentum_buffer"],
        torch.tensor([2.0])
    )


def test_adamw_state():

    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer = AdamWCustom(
        [parameter],
        lr=0.001,
    )

    parameter.grad = torch.tensor([0.5])
    optimizer.step()

    state = optimizer.state[parameter]

    assert state["step"] == 1
    assert "exp_avg" in state
    assert "exp_avg_sq" in state


def test_invalid_learning_rate():

    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError):
        SGDMomentum([parameter], lr=-0.1)

    with pytest.raises(ValueError):
        AdamWCustom([parameter], lr=-0.1)
