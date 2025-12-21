# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/dice.py

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from atommic.collections.common.parts.utils import is_none
from atommic.collections.segmentation.losses.utils import do_metric_reduction, one_hot
from atommic.core.classes.loss import Loss


class Dice(Loss):
    """Wrapper for :py:class:`monai.losses.DiceLoss`. Computes the average Dice loss between two tensors.

    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    Examples
    --------
    >>> import torch
    >>> from atommic.collections.segmentation.losses.dice import Dice
    >>> pred = torch.tensor([[[[0.1, 0.2, 0.3, 0.4, 0.5],
    ...                        [0.1, 0.2, 0.3, 0.4, 0.5],
    ...                        [0.1, 0.2, 0.3, 0.4, 0.5],
    ...                        [0.1, 0.2, 0.3, 0.4, 0.5],
    ...                        [0.1, 0.2, 0.3, 0.4, 0.5]]],
    ...                       [[[0.1, 0.2, 0.3, 0.4, 0.5],
    ...                         [0.1, 0.2, 0.3, 0.4, 0.5],
    ...                         [0.1, 0.2, 0.3, 0.4, 0.5],
    ...                         [0.1, 0.2, 0.3, 0.4, 0.5],
    ...                         [0.1, 0.2, 0.3, 0.4, 0.5]]]])
    >>> target = torch.tensor([[[[0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0]]],
    ...                         [[[1, 1, 1, 1, 1],
    ...                           [1, 1, 1, 1, 1],
    ...                           [1, 1, 1, 1, 1],
    ...                           [1, 1, 1, 1, 1],
    ...                           [1, 1, 1, 1, 1]]]])
    >>> dice = Dice(include_background=False, to_onehot_y=True, sigmoid=False, softmax=False)
    >>> dice(pred, target)
    tensor(0.5000)
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        flatten: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = True,
        num_segmentation_classes: int = None,
    ):
        """Inits :class:`Dice`.

        Parameters
        ----------
        include_background : bool
            whether to skip Dice computation on the first channel of the predicted output. Default is ``True``.
        to_onehot_y : bool
            Whether to convert `y` into the one-hot format. Default is ``False``.
        sigmoid : bool
            Whether to add sigmoid function to the input data. Default is ``True``.
        softmax : bool
            Whether to add softmax function to the input data. Default is ``False``.
        other_act : Callable
            Use this parameter if you want to apply another type of activation layer. Default is ``None``.
        squared_pred : bool
            Whether to square the prediction before calculating Dice. Default is ``False``.
        jaccard : bool
            Whether to compute Jaccard Index as a loss. Default is ``False``.
        flatten : bool
            Whether to flatten input data. Default is ``False``.
        reduction : Union[str, None]
            Specifies the reduction to apply:
            ``none``: no reduction will be applied.
            ``mean``: reduction with averaging over both batch and channel dimensions if input is 2D, or batch
            dimension only if input is 1D
            ``sum``: reduction with summing over both batch and channel dimensions if input is 2D, or batch dimension
            only if input is 1D
            Default is ``mean``.
        smooth_nr : float
            A small constant added to the numerator to avoid `nan` when all items are 0. Default is ``1e-5``.
        smooth_dr : float
            A small constant added to the denominator to avoid `nan` when all items are 0. Default is ``1e-5``.
        batch : bool
            If True, compute Dice loss for each batch and return a tensor with shape (batch_size,).
            If False, compute Dice loss for the whole batch and return a tensor with shape (1,).
            Default is ``True``.
        num_segmentation_classes: int
            Total number of segmentation classes. Default is ``None``.
        """
        super().__init__()
        other_act = None if is_none(other_act) else other_act
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.flatten = flatten
        self.reduction = reduction
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.num_segmentation_classes = num_segmentation_classes

    def forward(self, target: torch.Tensor, _input: torch.Tensor) -> Tuple[Union[Tensor, Any], Tensor]:  # noqa: MC0001
        """Forward pass of :class:`Dice`.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor. Shape: (batch_size, num_classes, *spatial_dims) or (batch_size, 1, *spatial_dims)
        _input : torch.Tensor
            Prediction tensor. Shape: (batch_size, num_classes, *spatial_dims)

        Returns
        -------
        torch.Tensor
            Dice Loss
        """
        if isinstance(_input, np.ndarray):
            _input = torch.from_numpy(_input)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        if _input.dim() == 3:
            # if _input.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            _input = _input.unsqueeze(0) if _input.shape[-3] == self.num_segmentation_classes else _input.unsqueeze(1)
        if target.dim() == 3:
            # if target.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            target = target.unsqueeze(0) if target.shape[-3] == self.num_segmentation_classes else target.unsqueeze(1)

        if self.flatten:
            if target.dim() == 4:
                segmentation_classes_dim = 1
            else:
                segmentation_classes_dim = 0
            target = target.reshape(target.shape[segmentation_classes_dim], 1, -1)
            _input = _input.reshape(_input.shape[segmentation_classes_dim], 1, -1)

        if self.sigmoid:
            _input = torch.sigmoid(_input.float())

        n_pred_ch = _input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                _input = torch.softmax(_input.float(), 1).to(_input)

        if self.other_act is not None:
            _input = self.other_act(_input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                _input = _input[:, 1:]

        if target.shape != _input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from _input ({_input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        n_len = len(_input.shape)
        reduce_axis: List[int] = torch.arange(2, n_len).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * _input, dim=reduce_axis)
        if self.squared_pred:
            target = torch.pow(target, 2)
            _input = torch.pow(_input, 2)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(_input, dim=reduce_axis)
        denominator = ground_o + pred_o
        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)
        dice_score = (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        dice_score = torch.where(denominator > 0, dice_score, torch.tensor(1.0).to(pred_o.device))
        dice_score, _ = do_metric_reduction(dice_score, reduction=self.reduction)
        f: torch.Tensor = 1.0 - dice_score
        return dice_score, f


class GeneralisedDice(Loss):
    """Compute the Generalised Dice loss, as presented in [Sudre2017]_.

    Adapted from: https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279

    References
    ----------
    .. Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        w_type: str = "square",
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = True,
        num_segmentation_classes: int = None,
    ) -> None:
        """Inits :class:`GeneralisedDiceLoss`.

        Parameters
        ----------
        include_background : bool
            whether to skip Dice computation on the first channel of the predicted output. Default is ``True``.
        to_onehot_y : bool
            Whether to convert `y` into the one-hot format. Default is ``False``.
        sigmoid : bool
            Whether to add sigmoid function to the input data. Default is ``True``.
        softmax : bool
            Whether to add softmax function to the input data. Default is ``False``.
        other_act : Callable
            Use this parameter if you want to apply another type of activation layer. Default is ``None``.
        w_type: {``"square"``, ``"simple"``, ``"uniform"``}
            Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
        reduction : Union[str, None]
            Specifies the reduction to apply:
            ``none``: no reduction will be applied.
            ``mean``: reduction with averaging over both batch and channel dimensions if input is 2D, or batch
            dimension only if input is 1D
            ``sum``: reduction with summing over both batch and channel dimensions if input is 2D, or batch dimension
            only if input is 1D
            Default is ``mean``.
        smooth_nr : float
            A small constant added to the numerator to avoid `nan` when all items are 0. Default is ``1e-5``.
        smooth_dr : float
            A small constant added to the denominator to avoid `nan` when all items are 0. Default is ``1e-5``.
        batch : bool
            If True, compute Dice loss for each batch and return a tensor with shape (batch_size,).
            If False, compute Dice loss for the whole batch and return a tensor with shape (1,).
            Default is ``True``.
        num_segmentation_classes: int
            Total number of segmentation classes. Default is ``None``.
        """
        super().__init__()
        other_act = None if is_none(other_act) else other_act
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.w_type = w_type
        self.reduction = reduction
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.num_segmentation_classes = num_segmentation_classes

    def w_func(self, grnd):
        """Compute the weight factor for the Generalised Dice loss."""
        if self.w_type == "simple":
            return torch.reciprocal(grnd)
        if self.w_type == "square":
            return torch.reciprocal(grnd * grnd)
        return torch.ones_like(grnd)

    def forward(self, target: torch.Tensor, _input: torch.Tensor) -> Tuple[Union[Tensor, Any], Tensor]:  # noqa: MC0001
        """Forward pass of :class:`GeneralisedDice`.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor. Shape: (batch_size, num_classes, *spatial_dims) or (batch_size, 1, *spatial_dims)
        _input : torch.Tensor
            Prediction tensor. Shape: (batch_size, num_classes, *spatial_dims)

        Returns
        -------
        torch.Tensor
            GeneralizedDice Loss
        """
        if isinstance(_input, np.ndarray):
            _input = torch.from_numpy(_input)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        if _input.dim() == 3:
            # if _input.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            _input = _input.unsqueeze(0) if _input.shape[-3] == self.num_segmentation_classes else _input.unsqueeze(1)
        if target.dim() == 3:
            # if target.shape[-3] == self.num_segmentation_classes then we need dummy batch dim, else dummy channel dim
            target = target.unsqueeze(0) if target.shape[-3] == self.num_segmentation_classes else target.unsqueeze(1)

        if self.sigmoid:
            _input = torch.sigmoid(_input.float())

        n_pred_ch = _input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                _input = torch.softmax(_input.float(), 1).to(_input)

        if self.other_act is not None:
            _input = self.other_act(_input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                _input = _input[:, 1:]

        if target.shape != _input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({_input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        n_len = len(_input.shape)
        reduce_axis: List[int] = torch.arange(2, n_len).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * _input, reduce_axis)
        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(_input, reduce_axis)
        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        infs = torch.isinf(w)
        if not self.batch:
            w[infs] = 0.0
            w = w + infs * torch.max(w)
        else:
            w[infs] = 0.0
            max_values = torch.max(w, dim=0)[0].unsqueeze(dim=0)
            w = w + infs * max_values

        final_reduce_dim = 0 if intersection.dim() <= 1 else 1
        numer = 2.0 * (intersection * w).sum(final_reduce_dim, keepdim=True) + self.smooth_nr
        denom = (denominator * w).sum(final_reduce_dim, keepdim=True) + self.smooth_dr
        gendice_score = numer / denom
        gendice_score = torch.where(denominator > 0, gendice_score, torch.tensor(1.0).to(pred_o.device))
        gendice_score, _ = do_metric_reduction(gendice_score, reduction=self.reduction)
        f: torch.Tensor = 1.0 - gendice_score
        return gendice_score, f
