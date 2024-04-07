# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/lucidrains/lion-pytorch/tree/main/lion_pytorch

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["Lion"]


class Lion(Optimizer):
    """Implements Lion, EvoLved Sign Momentum optimizer.

    This implementation is based on: `Symbolic Discovery of Optimization Algorithms` (see
    https://arxiv.org/abs/2302.06675)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """Inits :class:`Lion`.

        Parameters
        ----------
        params : iterable
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr : float, optional
            Learning rate. Default is ``1e-3``.
        betas : tuple of floats, optional
            Coefficients used for computing running averages of gradient and its square. Default is ``(0.9, 0.999)``.
        eps : float, optional
            Term added to the denominator to improve numerical stability. Default is ``1e-8``.
        weight_decay : float, optional
            Weight decay (L2 penalty). Default is ``0``.
        """
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Step through the optimizer"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)

                state["step"] += 1

                # Weight update
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                step_size = exp_avg * beta1 + grad * (1 - beta1)

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * group["lr"])

                # more conservative since it's an approximated value
                p_data_fp32.add_(torch.sign(step_size), alpha=-group["lr"])

                p.data.copy_(p_data_fp32)

            return loss
