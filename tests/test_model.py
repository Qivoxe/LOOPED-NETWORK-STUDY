import pytest
import torch

from src.models.looped_network import LoopedNetwork


def test_output_shape():
    model = LoopedNetwork(
        input_dim=2,
        hidden_dim=64,
        output_dim=2,
        num_steps=4
    )

    x = torch.randn(32, 2)
    output = model(x)

    assert output.shape == (32, 2)


def test_shared_parameters():
    model = LoopedNetwork(num_steps=4)

    parameter_ids = [
        id(parameter)
        for parameter in model.shared_block.parameters()
    ]

    # The shared block must contain unique parameters.
    assert len(parameter_ids) == len(set(parameter_ids))

    # Its parameter count must not grow with recurrent depth.
    shallow = LoopedNetwork(num_steps=1)
    deep = LoopedNetwork(num_steps=8)

    shallow_count = sum(
        p.numel() for p in shallow.shared_block.parameters()
    )

    deep_count = sum(
        p.numel() for p in deep.shared_block.parameters()
    )

    assert shallow_count == deep_count


def test_backward_pass():
    model = LoopedNetwork(num_steps=4)

    x = torch.randn(16, 2)
    output = model(x)

    loss = output.square().mean()
    loss.backward()

    for parameter in model.shared_block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_invalid_depth():
    with pytest.raises(ValueError):
        LoopedNetwork(num_steps=0)