# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """Wrapper around PyTorch's CrossEntropyLoss to support 2D and 3D inputs."""

    def __init__(
        self,
        num_samples: int = 50,
        ignore_index: int = -100,
        reduction: str = "none",
        label_smoothing: float = 0.0,
        weight: torch.Tensor = None,
    ):
        """Inits :class:`CrossEntropyLoss`.

        Parameters
        ----------
        num_samples : int, optional
            Number of Monte Carlo samples, by default 50
        ignore_index : int, optional
            Index to ignore, by default -100
        reduction : str, optional
            Reduction method, by default "none"
        label_smoothing : float, optional
            Label smoothing, by default 0.0
        weight : torch.Tensor, optional
            Weight for each class, by default None
        """
        super().__init__()
        self.mc_samples = num_samples
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, target: torch.Tensor, _input: torch.Tensor, pred_log_var: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of :class:`CrossEntropyLoss`.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor. Shape: (batch_size, num_classes, *spatial_dims)
        _input : torch.Tensor
            Prediction tensor. Shape: (batch_size, num_classes, *spatial_dims)
        pred_log_var : torch.Tensor, optional
            Prediction log variance tensor. Shape: (batch_size, num_classes, *spatial_dims). Default is ``None``.

        Returns
        -------
        torch.Tensor
            Loss tensor. Shape: (batch_size, *spatial_dims)
        """
        # In case we do not have a batch dimension, add it
        if _input.dim() == 3:
            _input = _input.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        self.cross_entropy.weight = self.cross_entropy.weight.clone().to(_input.device)

        if self.mc_samples == 1 or pred_log_var is None:
            return self.cross_entropy(_input.float(), target).mean()

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        loss = self.cross_entropy(noisy_pred, tiled_target).view(self.mc_samples, -1, *_input.shape[-2:]).mean(0)
        return loss.mean()
