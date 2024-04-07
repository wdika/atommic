# coding=utf-8
__author__ = "Dimitris Karkalousos"

from atommic.collections.multitask.rs.parts.transforms import RSMRIDataTransforms

__all__ = ["SegmentationMRIDataTransforms"]


class SegmentationMRIDataTransforms(RSMRIDataTransforms):
    """Transforms for the MRI segmentation task.

    .. note::
        Extends :class:`atommic.collections.multitask.rs.parts.transforms.RSMRIDataTransforms`.
    """
