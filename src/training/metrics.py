
import torch


def calculate_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean().item()


def calculate_gradient_norm(model):
    squared_norms = []

    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norms.append(
                parameter.grad.detach().norm(2).square()
            )

    if not squared_norms:
        return 0.0

    return torch.stack(squared_norms).sum().sqrt().item()


def calculate_metrics(loss, logits, targets):
    return {
        "loss": loss.item(),
        "accuracy": calculate_accuracy(logits, targets),
    }
