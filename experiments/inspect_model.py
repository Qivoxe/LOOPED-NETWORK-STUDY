import torch

from src.models.looped_network import LoopedNetwork


def inspect_model(num_steps):

    model = LoopedNetwork(
        input_dim=2,
        hidden_dim=64,
        output_dim=2,
        num_steps=num_steps
    )

    x = torch.randn(8, 2)

    output = model(x)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    shared_parameters = sum(
        p.numel()
        for p in model.shared_block.parameters()
    )

    print(f"\nRecurrent steps: {num_steps}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Shared block parameters: {shared_parameters:,}")


if __name__ == "__main__":

    for steps in [1, 2, 4, 8]:
        inspect_model(steps)