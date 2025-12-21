# coding=utf-8
__author__ = "Dimitris Karkalousos"

import warnings
from typing import List, Optional

import torch

from atommic.collections.common.parts.utils import is_none
from atommic.collections.segmentation.losses.utils import one_hot
from atommic.core.classes.loss import Loss


class CategoricalCrossEntropyLoss(Loss):
    """Wrapper around PyTorch's CrossEntropyLoss to support 2D and 3D inputs."""

    def __init__(
        self,
        include_background: bool = True,
        num_samples: int = 50,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        weight: Optional[List] = None,
        to_onehot_y: bool = False,
        num_segmentation_classes: int = None,
    ):
        """Inits :class:`CategoricalCrossEntropyLoss`.

        Parameters
        ----------
        include_background : bool
            Whether to include the computation on the first channel of the predicted output. Default is ``True``.
        num_samples : int, optional
            Number of Monte Carlo samples. Default is ``50``.
        ignore_index : int, optional
            Index to ignore. Default is ``-100``.
        reduction : Union[str, None]
            Specifies the reduction to apply:
            ``none``: no reduction will be applied.
            ``mean``: reduction with averaging over both batch and channel dimensions if input is 2D, or batch
            dimension only if input is 1D
            ``sum``: reduction with summing over both batch and channel dimensions if input is 2D, or batch dimension
            only if input is 1D
            Default is ``mean``.
        label_smoothing : float, optional
            Label smoothing. Default is ``0.0``.
        weight : list of floats, optional
            List with weights for each class. Default is ``None``.
        to_onehot_y : bool
            Whether to convert `y` into the one-hot format. Default is ``False``.
        num_segmentation_classes: int
            Total number of segmentation classes. Default is ``None``.
        """
        super().__init__()
        self.include_background = include_background
        self.mc_samples = num_samples
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.weight = None if is_none(weight) else torch.tensor(weight)
        self.to_onehot_y = to_onehot_y
        self.num_segmentation_classes = num_segmentation_classes

        self.cross_entropy = torch.nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    def forward(
        self, target: torch.Tensor, _input: torch.Tensor, pred_log_var: torch.Tensor = None  # noqa: MC0001
    ) -> torch.Tensor:
        """Forward pass of :class:`CategoricalCrossEntropyLoss`.

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
            CategoricalCrossEntropy Loss
        """
        if _input.dim() == 3:
            # if _input.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            _input = _input.unsqueeze(0) if _input.shape[-3] == self.num_segmentation_classes else _input.unsqueeze(1)
        if target.dim() == 3:
            # if target.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            target = target.unsqueeze(0) if target.shape[-3] == self.num_segmentation_classes else target.unsqueeze(1)

        self.cross_entropy.weight = (
            self.cross_entropy.weight.to(target).clone() if self.cross_entropy.weight is not None else None
        )

        n_pred_ch = _input.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y = True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background = False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                _input = _input[:, 1:]

        if self.mc_samples == 1 or pred_log_var is None:
            return self.cross_entropy(_input.float(), target)

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        loss = self.cross_entropy(noisy_pred, tiled_target).view(self.mc_samples, -1, *_input.shape[-2:])
        return loss


class BinaryCrossEntropyLoss(Loss):
    """Wrapper around PyTorch's BinaryCrossEntropyLoss to support 2D and 3D inputs."""

    def __init__(
        self,
        include_background: bool = True,
        num_samples: int = 50,
        weight: Optional[List] = None,
        reduction: str = "mean",
        to_onehot_y: bool = False,
        num_segmentation_classes: int = None,
    ):
        """Inits :class:`BinaryCrossEntropyLoss`.

        Parameters
        ----------
        include_background : bool
            Whether to include the computation on the first channel of the predicted output. Default is ``False``.
        num_samples : int, optional
            Number of Monte Carlo samples. Default is ``50``.
        weight : list of floats, optional
            List of weight for each sample. Default is ``None``.
        reduction : Union[str, None]
            Specifies the reduction to apply:
            ``none``: no reduction will be applied.
            ``mean``: reduction with averaging over both batch and channel dimensions if input is 2D, or batch
            dimension only if input is 1D
            ``sum``: reduction with summing over both batch and channel dimensions if input is 2D, or batch dimension
            only if input is 1D
            Default is ``mean``.
        to_onehot_y : bool
            Whether to convert `y` into the one-hot format. Default is ``False``.
        num_segmentation_classes: int
            Total number of segmentation classes. Default is ``None``.
        """
        super().__init__()
        self.include_background = include_background
        self.mc_samples = num_samples
        self.weight = None if is_none(weight) else torch.tensor(weight)
        if self.weight is not None:
            self.weight = self.weight.view(1, len(self.weight), 1, 1)
        self.reduction = reduction
        self.to_onehot_y = to_onehot_y
        self.num_segmentation_classes = num_segmentation_classes

        self.binary_cross_entropy = torch.nn.BCEWithLogitsLoss(weight=self.weight, reduction=self.reduction)

    def forward(
        self, target: torch.Tensor, _input: torch.Tensor, pred_log_var: torch.Tensor = None  # noqa: MC0001
    ) -> torch.Tensor:
        """Forward pass of :class:`BinaryCrossEntropyLoss`.

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
            BinaryCrossEntropy Loss
        """
        if _input.dim() == 3:
            # if _input.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            _input = _input.unsqueeze(0) if _input.shape[-3] == self.num_segmentation_classes else _input.unsqueeze(1)
        if target.dim() == 3:
            # if target.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            target = target.unsqueeze(0) if target.shape[-3] == self.num_segmentation_classes else target.unsqueeze(1)

        self.binary_cross_entropy.weight = (
            self.binary_cross_entropy.weight.to(target).clone()
            if self.binary_cross_entropy.weight is not None
            else None
        )
        n_pred_ch = _input.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y = True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background = False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                _input = _input[:, 1:]

        if self.mc_samples == 1 or pred_log_var is None:
            return self.binary_cross_entropy(_input.float(), target)

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        loss = (
            self.binary_cross_entropy(noisy_pred, tiled_target).view(self.mc_samples, -1, *_input.shape[-2:]).mean(0)
        )
        return loss
