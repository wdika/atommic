# coding=utf-8
__author__ = "Dimitris Karkalousos"

from atommic.collections.common.parts.transforms import MRIDataTransforms

__all__ = ["ReconstructionMRIDataTransforms"]


class ReconstructionMRIDataTransforms(MRIDataTransforms):
    """Transforms for the accelerated-MRI reconstruction task.

    .. note::
        Extends :class:`atommic.collections.common.parts.transforms.MRIDataTransforms`.
    """
