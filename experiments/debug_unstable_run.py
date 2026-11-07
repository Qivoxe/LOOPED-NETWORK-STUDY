
import torch
import yaml
from pathlib import Path

from src.models.looped_network import LoopedNetwork
from src.optimizers import SGDMomentum
from src.training.reproducibility import set_seed
from experiments.run_experiments import (
    prepare_dataset,
    create_loaders,
)


ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs/learning_rate_experiment.yaml") as f:
    config = yaml.safe_load(f)

SEED = 2026
DEPTH = 8
LR = 0.03

set_seed(SEED)

dataset = prepare_dataset(config)

model_config = config["model"].copy()
model_config["num_steps"] = DEPTH

model = LoopedNetwork(**model_config)

optimizer = SGDMomentum(
    model.parameters(),
    lr=LR,
    momentum=0.9,
    weight_decay=0.0,
)

train_loader, _ = create_loaders(
    *dataset,
    config["training"]["batch_size"],
    SEED,
)

criterion = torch.nn.CrossEntropyLoss()

print("Reproducing unstable configuration:")
print(f"SGD | LR={LR} | Depth={DEPTH} | Seed={SEED}")

for epoch in range(1, 4):
    model.train()
    epoch_losses = []
    failed = False

    for batch_idx, (inputs, targets) in enumerate(train_loader, 1):

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if not torch.isfinite(loss):
            print(f"NON-FINITE LOSS at epoch {epoch}, batch {batch_idx}")
            failed = True
            break

        loss.backward()

        gradients = [
            p.grad.detach()
            for p in model.parameters()
            if p.grad is not None
        ]

        grad_finite = all(
            torch.isfinite(g).all().item() for g in gradients
        )

        grad_norm = torch.sqrt(
            sum(g.norm(2).square() for g in gradients)
        ).item()

        max_grad = max(g.abs().max().item() for g in gradients)

        if not grad_finite:
            print(f"NON-FINITE GRADIENT at epoch {epoch}, batch {batch_idx}")
            failed = True
            break

        optimizer.step()

        params_finite = all(
            torch.isfinite(p).all().item()
            for p in model.parameters()
        )

        max_param = max(
            p.detach().abs().max().item()
            for p in model.parameters()
        )

        if not params_finite:
            print(
                f"NON-FINITE PARAMETERS after epoch {epoch}, "
                f"batch {batch_idx}"
            )
            print(f"Loss before update: {loss.item():.6f}")
            print(f"Gradient norm: {grad_norm:.6f}")
            print(f"Maximum gradient: {max_grad:.6f}")
            failed = True
            break

        epoch_losses.append(loss.item())

        if batch_idx == 1 or batch_idx % 5 == 0:
            print(
                f"Epoch {epoch}, batch {batch_idx} | "
                f"loss={loss.item():.5f} | "
                f"grad_norm={grad_norm:.4f} | "
                f"max_grad={max_grad:.4f} | "
                f"max_param={max_param:.4f}"
            )

    if failed:
        print("\nDiagnostic stopped at first numerical failure.")
        break

    print(
        f"Epoch {epoch} completed | "
        f"mean batch loss={sum(epoch_losses)/len(epoch_losses):.5f}"
    )

print("\nDiagnostic finished.")
