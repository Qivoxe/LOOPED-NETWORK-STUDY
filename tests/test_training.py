
import torch

from torch.utils.data import DataLoader, TensorDataset

from src.models.looped_network import LoopedNetwork
from src.optimizers import AdamWCustom
from src.training.trainer import Trainer


def test_training_pipeline():

    X = torch.randn(128, 2)
    y = (X[:, 0] + X[:, 1] > 0).long()

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    model = LoopedNetwork(num_steps=2)

    optimizer = AdamWCustom(
        model.parameters(),
        lr=0.001,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
    )

    history = trainer.fit(
        loader,
        loader,
        epochs=2,
    )

    assert len(history) == 2

    assert "train_loss" in history[0]
    assert "validation_loss" in history[0]
    assert "gradient_norm" in history[0]


def test_evaluation_does_not_update_parameters():

    model = LoopedNetwork(num_steps=2)

    optimizer = AdamWCustom(model.parameters())

    trainer = Trainer(model, optimizer, device="cpu")

    X = torch.randn(32, 2)
    y = torch.randint(0, 2, (32,))

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=16,
    )

    before = [
        p.detach().clone()
        for p in model.parameters()
    ]

    trainer.evaluate(loader)

    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new)
