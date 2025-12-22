# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/utils.py

import warnings
from typing import Any, Tuple

import torch
from torch import Tensor

from atommic.collections.common.parts.utils import check_one_hot


def do_metric_reduction(f: torch.Tensor, reduction: str = "mean") -> Tuple[Tensor, Any]:
    """Utility function to perform metric reduction.

    Parameters
    ----------
    f : torch.Tensor
        the metric to reduce.
    reduction : Union[str, None]
        Specifies the reduction to apply:
        ``none``: no reduction will be applied.
        ``mean``: reduction with averaging over both batch and channel dimensions if input is 2D, or batch dimension
        only if input is 1D
        ``sum``: reduction with summing over both batch and channel dimensions if input is 2D, or batch dimension
        only if input is 1D
        Default is ``mean``.

    Returns
    -------
    torch.Tensor or Any
        the reduced metric.
    Any
        NaNs if there are any NaNs in the input, otherwise 0.

    Examples
    --------
    >>> import torch
    >>> from atommic.collections.segmentation.losses.utils import do_metric_reduction
    >>> f = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> do_metric_reduction(f, "mean")
    (tensor(6.5000), 0)
    >>> do_metric_reduction(f, "sum")
    (tensor(78), 0)
    """
    # some elements might be Nan (if ground truth y was missing (zeros)), we need to account for it
    nans = torch.isnan(f)
    not_nans = (~nans).float()
    t_zero = torch.zeros(1, device=f.device, dtype=f.dtype)
    if reduction is None:
        return f, not_nans
    f[nans] = 0
    if reduction == "mean":
        # first mean by channel (accounting for nans) if 2D, then by batch
        if f.dim() == 2:
            not_nans = not_nans.sum(dim=1)
            f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average
        not_nans = (not_nans > 0).float().sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average
    elif reduction == "sum":
        # first sum by channel (accounting for nans) if 2D, then by batch
        if f.dim() == 2:
            not_nans = not_nans.sum(dim=1)
            f = f.sum(dim=1)  # channel summation
        not_nans = (not_nans > 0).float().sum(dim=0)
        f = f.sum(dim=0)  # batch summation
    elif reduction == "none":
        pass
    else:
        raise ValueError(f'Unsupported reduction: {reduction}, available options are ["mean", "sum", "none"].')
    return f, not_nans


def reduce_mean_dim(
    f: torch.Tensor, not_nans: torch.Tensor, t_zero: torch.Tensor, dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Utility function to reduce the mean of a tensor along a specific dimension.

    Parameters
    ----------
    f : torch.Tensor
        the tensor to reduce.
    not_nans : torch.Tensor
        the tensor with the non-NaN values.
    t_zero : torch.Tensor
        the tensor with zeros.
    dim : int
        the dimension to reduce.

    Returns
    -------
    torch.Tensor
        the reduced tensor.
    torch.Tensor
        the non-NaN values.
    """
    not_nans = not_nans.sum(dim=dim)
    return torch.where(not_nans > 0, f.sum(dim=dim) / not_nans, t_zero), not_nans


def reduce_sum_dim(f: torch.Tensor, not_nans: torch.Tensor, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Utility function to reduce the sum of a tensor along a specific dimension.

    Parameters
    ----------
    f : torch.Tensor
        the tensor to reduce.
    not_nans : torch.Tensor
        the tensor with the non-NaN values.
    dim : int
        the dimension to reduce.

    Returns
    -------
    torch.Tensor
        the reduced tensor.
    torch.Tensor
        the non-NaN values.
    """
    return f.sum(dim=dim), not_nans.sum(dim=dim)


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """Convert labels to one-hot representation.

    Parameters
    ----------
    labels: torch.Tensor
        the labels of shape [BNHW[D]].
    num_classes: int
        number of classes.
    dtype: torch.dtype
        the data type of the returned tensor.
    dim: int
        the dimension to expand the one-hot tensor.

    Returns
    -------
    torch.Tensor
        The one-hot representation of the labels.

    Examples
    --------
    >>> labels = torch.tensor([[[[0, 1, 2]]]])
    >>> one_hot(labels, num_classes=3)
    tensor([[[[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]]]])
    """
    # if `dim` is bigger, add singleton dim at the end
    if check_one_hot(labels):
        warnings.warn('Tensor is already one-hot encoded, to_onehot_y is redundant!')
        return labels
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)
    sh = list(labels.shape)
    sh[dim] = num_classes
    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)
    return labels
