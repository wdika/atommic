# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from:
# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/metrics/global_average_loss_metric.py

import torch
from torchmetrics import Metric

__all__ = ["GlobalAverageLossMetric"]


class GlobalAverageLossMetric(Metric):
    """This class is for averaging loss across multiple processes if a distributed backend is used. True average is
    computed not running average. It does not accumulate gradients so the averaged loss cannot be used for
    optimization.

    .. note::
        If ``take_avg_loss`` is ``True``, the :meth:`update` method ``loss`` argument has to be a mean loss. If
        ``take_avg_loss`` is ``False`` then the :meth:`update` method ``loss`` argument has to be a sum of losses. See
        PyTorch Lightning Metrics for the metric usage instruction.

    Examples
    --------
    >>> from atommic.collections.common.metrics.global_average_loss_metric import GlobalAverageLossMetric
    >>> metric = GlobalAverageLossMetric()
    >>> metric.update(torch.tensor(1.0), torch.tensor(1))
    >>> metric.update(torch.tensor(2.0), torch.tensor(1))
    >>> metric.compute()
    tensor(1.5000)
    >>> metric.update(torch.tensor(3.0), torch.tensor(1))
    >>> metric.compute()
    tensor(2.0000)
    """

    full_state_update: bool = True

    def __init__(self, dist_sync_on_step=False, process_group=None, take_avg_loss=True):
        """Inits :class:`GlobalAverageLossMetric`.

        Parameters
        ----------
        dist_sync_on_step : bool
            Synchronize metric state across processes at each method :meth:`forward` call before returning the value at
             the step. Default is ``False``.
        process_group : Any, optional
            Specify the process group on which synchronization is called. default: ``None`` (which selects the entire
            world). Default is ``None``.
        take_avg_loss : bool
            If ``True`` values of :meth:`update` method ``loss`` argument has to be a mean loss. If ``False`` values of
            :meth:`update` method ``loss`` argument has to be a sum of losses. Default is ``True``.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.add_state("loss_sum", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state("num_measurements", torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')
        self.take_avg_loss = take_avg_loss

    def update(self, loss, num_measurements):  # pylint: disable=arguments-differ
        """Updates :attr:`loss_sum` and :attr:`num_measurements`.

        Parameters
        ----------
        loss : torch.Tensor
            A float zero dimensional ``torch.Tensor`` which is either sum or average of losses for processed examples.
            See ``take_avg_loss`` parameter of :meth:`__init__`.
        num_measurements : torch.Tensor
            An integer zero dimensional ``torch.Tensor`` which contains a number of loss measurements. The sum or mean
            of the results of these measurements are in the ``loss`` parameter.
        """
        if self.take_avg_loss:
            self.loss_sum = self.loss_sum + loss.detach() * num_measurements
        else:
            self.loss_sum = self.loss_sum + loss.detach()
        self.num_measurements = self.num_measurements + num_measurements

    def compute(self):
        """Returns mean loss."""
        if self.num_measurements.eq(0):
            return torch.tensor(float("nan"))
        return self.loss_sum / self.num_measurements
