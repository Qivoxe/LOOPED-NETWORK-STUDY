
import torch


class AdamWCustom(torch.optim.Optimizer):
    """
    AdamW implementation with decoupled weight decay.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError("Learning rate must be non-negative")

        if not 0 <= betas[0] < 1:
            raise ValueError("Invalid beta1")

        if not 0 <= betas[1] < 1:
            raise ValueError("Invalid beta2")

        if eps <= 0:
            raise ValueError("Epsilon must be positive")

        if weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                grad = parameter.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamWCustom does not support sparse gradients"
                    )

                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                corrected_avg = exp_avg / bias_correction1
                corrected_avg_sq = exp_avg_sq / bias_correction2

                denominator = corrected_avg_sq.sqrt().add_(eps)

                # Decoupled weight decay.
                parameter.mul_(1 - lr * weight_decay)

                parameter.addcdiv_(
                    corrected_avg,
                    denominator,
                    value=-lr
                )

        return loss
