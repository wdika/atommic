# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/aggregator.py

from typing import List

import torch

from atommic.core.classes.common import typecheck
from atommic.core.classes.loss import Loss
from atommic.core.neural_types.elements import LossType
from atommic.core.neural_types.neural_type import NeuralType

__all__ = ["AggregatorLoss"]


class AggregatorLoss(Loss):
    """Aggregates multiple losses into a single loss.

    Examples
    --------
    >>> from atommic.collections.common.losses.aggregator import AggregatorLoss
    >>> loss = AggregatorLoss(num_inputs=2)
    >>> loss(loss_1=torch.tensor(1.0), loss_2=torch.tensor(2.0))
    tensor(3.)
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {f"loss_{str(i + 1)}": NeuralType(elements_type=LossType()) for i in range(self._num_losses)}

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_inputs: int = 2, weights: List[float] = None):
        """Inits :class:`AggregatorLoss`.

        Parameters
        ----------
        num_inputs : int
            Number of losses to be summed.
        weights : List[float]
            Weights to be applied to each loss. If None, all losses are weighted equally.
        reduction : str
            Reduction method to be applied to the aggregated loss.
        """
        super().__init__()
        self._num_losses = num_inputs
        if weights is not None and len(weights) != num_inputs:
            raise ValueError("Length of weights should be equal to the number of inputs (num_inputs)")
        self._weights = weights

    @typecheck()
    def forward(self, **kwargs):
        """Computes the sum of the losses."""
        values = [kwargs[x] for x in sorted(kwargs.keys())]
        loss = torch.zeros_like(values[0])
        for loss_idx, loss_value in enumerate(values):
            if self._weights is not None:
                loss = loss.add(loss_value, alpha=self._weights[loss_idx])
            else:
                loss = loss.add(loss_value)
        return loss
