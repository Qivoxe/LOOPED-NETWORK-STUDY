
import torch
import torch.nn as nn


class SharedBlock(nn.Module):
    """A computational block reused across recurrent steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.network(self.norm(x))


class LoopedNetwork(nn.Module):
    """Residual neural network with shared recurrent parameters."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_steps: int = 4,
    ):
        super().__init__()

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        self.num_steps = num_steps

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # One instance reused at every recurrent step.
        self.shared_block = SharedBlock(hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)

        for _ in range(self.num_steps):
            h = self.shared_block(h)

        return self.output_projection(h)
