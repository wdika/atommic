# coding=utf-8
__author__ = "Tim Paquaij, Dimitris Karkalousos"

# Taken and adapted from:
# https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/losses/focal_loss.py

import warnings
from typing import Any, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor

from atommic.collections.segmentation.losses.utils import one_hot
from atommic.core.classes.loss import Loss


class FocalLoss(Loss):
    """FocalLoss is an extension of BCEWithLogitsLoss that down-weights loss from  high confidence correct predictions.

    Reimplementation of the Focal Loss described in:

        - ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002), T. Lin et al., ICCV 2017
        - "AnatomyNet: Deep learning for fast and fully automated whole-volume segmentation of head and neck anatomy",
          Zhu et al., Medical Physics 2018

    """

    def __init__(
        self,
        include_background: bool = True,
        gamma: float = 2.0,
        alpha: float | None = None,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        reduction: str = "mean",
        use_softmax: bool = False,
        to_onehot_y: bool = False,
        num_segmentation_classes: int = None,
    ) -> None:
        """Inits :class:`FocalLoss`

        Parameters
        ----------
        include_background : bool
            whether to include the computation on the first channel of the predicted output. Default is ``True``.
        gamma : float, optional
            Value of the exponent gamma in the definition of the Focal loss. Default is 2
        alpha : float, optional
            Value of the alpha: [0,1] in the definition of the alpha-balanced Focal loss. Default is None
        weight : torch.Tensor, optional
            Weight for each class. Default is None
        reduction : Union[str, None]
            Specifies the reduction to apply:
            ``none``: no reduction will be applied.
            ``mean``: reduction with averaging over both batch and channel dimensions if input is 2D, or batch
            dimension only if input is 1D
            ``sum``: reduction with summing over both batch and channel dimensions if input is 2D, or batch dimension
            only if input is 1D
            Default is ``mean``.
        use_softmax : bool, optional
            option to compute the focal loss as a categorical cross-entropy. Default is ``False``
        to_onehot_y : bool
            Whether to convert `y` into the one-hot format. Default is ``False``.
        num_segmentation_classes: int
            Total number of segmentation classes. Default is ``None``.
        """
        super().__init__()
        self.include_background = include_background
        self.gamma = gamma
        self.alpha = alpha
        self.weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", self.weight)
        self.reduction = reduction
        self.use_softmax = use_softmax
        self.to_onehot_y = to_onehot_y
        self.num_segmentation_classes = num_segmentation_classes

    def forward(self, target: torch.Tensor, _input: torch.Tensor) -> Tuple[Union[Tensor, Any], Tensor]:  # noqa: MC0001
        """Forward pass of :class:`FocalLoss`.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor. Shape: (batch_size, num_classes, *spatial_dims)
        _input : torch.Tensor
            Prediction tensor. Shape: (batch_size, num_classes, *spatial_dims)

        Returns
        -------
        torch.Tensor
            Focal Loss
        """
        unsqueezed = False
        if _input.dim() == 3:
            # if _input.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            _input = _input.unsqueeze(0) if _input.shape[-3] == self.num_segmentation_classes else _input.unsqueeze(1)
        if target.dim() == 3:
            # if target.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            target = target.unsqueeze(0) if target.shape[-3] == self.num_segmentation_classes else target.unsqueeze(1)
            unsqueezed = True

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

        if target.shape != _input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({_input.shape})")

        loss: Optional[torch.Tensor] = None
        _input = _input.float()
        target = target.float()
        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn("`include_background = False`, `alpha` ignored when using softmax.")
            loss = softmax_focal_loss(_input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss(_input, target, self.gamma, self.alpha)

        num_of_classes = target.shape[1]
        if (self.class_weight is not None) and (  # type: ignore  # pylint: disable=access-member-before-definition
            num_of_classes != 1
        ):
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:  # type: ignore  # pylint: disable=access-member-before-definition
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)  # type: ignore
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes. If
                        `include_background = False`, the weight should not include the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            self.class_weight = self.class_weight.to(loss)
            broadcast_dims = [-1] + [1] * len(target.shape[2:])
            self.class_weight = self.class_weight.view(broadcast_dims)
            loss = self.class_weight * loss

        focal_score = loss.clone()

        # some elements might be Nan (if ground truth y was missing (zeros)), we need to account for it
        nans = torch.isnan(loss)
        not_nans = (~nans).float()
        t_zero = torch.zeros(1, device=loss.device, dtype=loss.dtype)
        loss[nans] = 0
        if self.reduction == "mean":
            not_nans = not_nans.sum(dim=3)
            loss = torch.where(not_nans > 0, loss.sum(dim=3) / not_nans, t_zero)  # second spatial dim average
            not_nans = not_nans.sum(dim=2)
            loss = torch.where(not_nans > 0, loss.sum(dim=2) / not_nans, t_zero)  # first spatial dim average
            not_nans = not_nans.sum(dim=1)
            loss = torch.where(not_nans > 0, loss.sum(dim=1) / not_nans, t_zero)  # channel average
            not_nans = (not_nans > 0).float().sum(dim=0)
            loss = torch.where(not_nans > 0, loss.sum(dim=0) / not_nans, t_zero)  # batch average
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none" and unsqueezed:
            loss = loss.squeeze(1)
            focal_score = focal_score.squeeze(1)

        return focal_score, loss


def softmax_focal_loss(
    _input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None  # noqa: MC0001
) -> torch.Tensor:
    """Softmax operation for focal loss

    Parameters
        ----------
    _input : torch.Tensor
        Prediction tensor. Shape: (batch_size, num_classes, *spatial_dims)
    target : torch.Tensor
        Target tensor. Shape: (batch_size, num_classes, *spatial_dims)
    gamma : float
        Value of the exponent gamma in the definition of the Focal loss. Default is ``2``.
    alpha : float, optional
        Value of the alpha: [0,1] in the definition of the alpha-balanced Focal loss. Default is ``None``.

    Returns
    -------
    torch.Tensor
        Focal Loss
    """
    input_ls = _input.log_softmax(1)
    loss: torch.Tensor = -(1 - input_ls.exp()).pow(gamma) * input_ls * target
    if alpha is not None:
        # (1-alpha) for the background class and alpha for the other classes
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss
    return loss


def sigmoid_focal_loss(
    _input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None  # noqa: MC0001
) -> torch.Tensor:
    """Sigmoid operation for focal loss

    Parameters
    ----------
    _input : torch.Tensor
        Prediction tensor. Shape: (batch_size, num_classes, *spatial_dims)
    target : torch.Tensor
        Target tensor. Shape: (batch_size, num_classes, *spatial_dims)
    gamma : float
        Value of the exponent gamma in the definition of the Focal loss. Default is ``2``.
    alpha : float, optional
        Value of the alpha: [0,1] in the definition of the alpha-balanced Focal loss. Default is ``None``.

    Returns
    -------
    torch.Tensor
        Focal Loss
    """
    # computing binary cross entropy with logits
    # equivalent to F.binary_cross_entropy_with_logits(_input, target, reduction = 'none')
    # see also https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp#L363
    loss: torch.Tensor = _input - _input * target - F.logsigmoid(_input)
    # sigmoid(-i) if t == 1; sigmoid(i) if t == 0 <=>
    # 1-sigmoid(i) if t == 1; sigmoid(i) if t == 0 <=>
    # 1-p if t == 1; p if t == 0  <=>
    # pfac, that is, the term (1 - pt)
    invprobs = F.logsigmoid(-_input * (target * 2 - 1))  # reduced chance of overflow
    # (pfac.log() * gamma).exp() <=>
    # pfac.log().exp() ^ gamma <=>
    # pfac ^ gamma
    loss = (invprobs * gamma).exp() * loss
    if alpha is not None:
        # alpha if t == 1; (1-alpha) if t == 0
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss
    return loss
