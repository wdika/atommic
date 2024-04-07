# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
import torch.nn.functional as F

from atommic.core.classes.loss import Loss


class NoiseAwareLoss(Loss):
    """Computes the Noise Aware loss between two tensors.

    .. note::
        Extends :class:`atommic.core.classes.loss.Loss`.

    Examples
    --------
    >>> from atommic.collections.reconstruction.losses.na import NoiseAwareLoss
    >>> import torch
    >>> loss = NoiseAwareLoss(win_size=7, k1=0.01, k2=0.03)
    >>> loss(X=torch.rand(1, 1, 256, 256), Y=torch.rand(1, 1, 256, 256))
    tensor(0.0872)
    """

    def forward(
        self, target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor = None, sigma: float = 0.0
    ) -> torch.Tensor:
        """Forward pass of :class:`NoiseAwareLoss`.

        Parameters
        ----------
        target : torch.Tensor
            The target tensor.
        pred : torch.Tensor
            The predicted tensor.
        mask : torch.Tensor
            The mask tensor. If None, all pixels are considered.
        sigma : float
            The noise level.
        """
        pred = pred.to(target.dtype)
        if mask is None:
            mask = torch.ones_like(target)
        mask = mask.to(target.dtype)

        # Compute the mean squared error
        mse = F.mse_loss(target, pred, reduction="none")

        # Compute the noise variance at each pixel
        sigma = torch.median(torch.abs(target - pred)) / 0.6745

        noise_var = sigma**2 / (1 - mask + 1e-8)

        # Compute the noise aware loss
        loss = mse / (2 * noise_var) + torch.log(
            2 * noise_var * torch.sqrt(torch.tensor([2 * 3.1415926535])).to(target.device)
        )
        loss = loss.mean()

        return loss
