
import time
import torch

from src.training.metrics import (
    calculate_accuracy,
    calculate_gradient_norm,
)


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion=None,
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.criterion = criterion or torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        gradient_norms = []

        start_time = time.perf_counter()

        for inputs, targets in dataloader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            loss.backward()

            gradient_norms.append(
                self._gradient_norm()
            )

            self.optimizer.step()

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == targets
            ).sum().item()

            total_samples += batch_size

        elapsed = time.perf_counter() - start_time

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "gradient_norm": sum(gradient_norms) / len(gradient_norms),
            "epoch_time": elapsed,
        }

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, targets in dataloader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == targets
            ).sum().item()

            total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def _gradient_norm(self):

        total = torch.tensor(0.0, device=self.device)

        for parameter in self.model.parameters():

            if parameter.grad is not None:
                total += parameter.grad.detach().norm(2).square()

        return total.sqrt().item()

    def fit(self, train_loader, validation_loader, epochs):

        history = []

        for epoch in range(epochs):

            train_metrics = self.train_epoch(train_loader)

            validation_metrics = self.evaluate(
                validation_loader
            )

            record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "gradient_norm": train_metrics["gradient_norm"],
                "epoch_time": train_metrics["epoch_time"],
            }

            history.append(record)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {record['train_loss']:.4f} | "
                f"Val Loss: {record['validation_loss']:.4f} | "
                f"Val Accuracy: {record['validation_accuracy']:.4f}"
            )

        return history
