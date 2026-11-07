
import torch


class SGDMomentum(torch.optim.Optimizer):
    """
    SGD optimizer with momentum.

    Implements:
        v_t = momentum * v_(t-1) + grad_t
        param = param - lr * v_t
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError("Learning rate must be non-negative")

        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in [0, 1)")

        if weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                grad = parameter.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "SGDMomentum does not support sparse gradients"
                    )

                if weight_decay != 0:
                    grad = grad.add(parameter, alpha=weight_decay)

                state = self.state[parameter]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = grad.clone()
                else:
                    state["momentum_buffer"].mul_(momentum).add_(grad)

                parameter.add_(
                    state["momentum_buffer"],
                    alpha=-lr
                )

        return loss
