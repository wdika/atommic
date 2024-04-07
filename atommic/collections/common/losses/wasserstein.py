# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from https://github.com/dfdazac/wassdistance/blob/master/layers.py

import torch

from atommic.core.classes.loss import Loss


class SinkhornDistance(Loss):
    r"""Given two empirical measures each with :math:`P_1` locations :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2`
    locations :math:`y\in\mathbb{R}^{D_2}`, outputs an approximation of the regularized OT cost for point clouds.
    """

    def __init__(self, eps=0.1, max_iter=100, reduction="mean"):
        """Inits :class:`SinkhornDistance`.

        Parameters
        ----------
        eps : float
            Regularization coefficient. Default is ``0.1``.
        max_iter : int
            Maximum number of Sinkhorn iterations. Default is ``100``.
        reduction : string, optional
            Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied.
            Default is ``mean``.
        """
        super().__init__()
        self.eps = torch.tensor([eps])
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        r"""Forward pass of the Sinkhorn algorithm.

        Parameters
        ----------
        x : torch.Tensor
            :math:`(N, P_1, D_1)`
        y : torch.Tensor
            :math:`(N, P_2, D_2)`

        Returns
        -------
        torch.Tensor
            The Sinkhorn distance between the two point clouds. Output shape :math:`(N)` or :math:`()`, depending on
            `reduction`
        """
        self.eps = self.eps.clone().to(x.device)

        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = (
            torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / x_points)
            .squeeze()
            .to(x.device)
        )
        nu = (
            torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / y_points)
            .squeeze()
            .to(x.device)
        )

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits = actual_nits + 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost  # , pi, C

    def M(self, C, u, v):
        r"""Modified cost for logarithmic updates $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"""
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        r"""Returns the matrix of $|x_i-y_j|^p$."""
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        r"""Barycenter subroutine, used by kinetic acceleration through extrapolation."""
        return tau * u + (1 - tau) * u1
