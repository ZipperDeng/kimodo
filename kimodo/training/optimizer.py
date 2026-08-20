"""Adam-atan2 optimizer used in the Kimodo technical report."""

import torch


class AdamAtan2(torch.optim.Optimizer):
    """Adam with a bounded ``atan2(m, sqrt(v))`` update.

    This follows the optimizer described by Everett et al.; decoupled weight
    decay is provided for practical large-model training.
    """

    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), weight_decay=0.0):
        """Create Adam-atan2.

        Unlike Adam, Adam-atan2 intentionally has no epsilon.  The ``atan2``
        operation remains well-defined when both moments are zero and preserves
        the optimizer's scale invariance.
        """
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamAtan2 does not support sparse gradients")
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group["weight_decay"]:
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                step = state["step"]
                bias_correction1 = 1 - beta1**step
                bias_correction2_sqrt = (1 - beta2**step) ** 0.5
                denominator = v.sqrt() / bias_correction2_sqrt
                update = torch.atan2(m, denominator)
                param.add_(update, alpha=-(group["lr"] / bias_correction1))
        return loss
