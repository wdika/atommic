# coding=utf-8
__author__ = "Dimitris Karkalousos"

from atommic.collections.multitask.rs.parts.transforms import RSCTDataTransforms, RSMRIDataTransforms

__all__ = ["SegmentationMRIDataTransforms", "SegmentationCTDataTransforms"]


class SegmentationMRIDataTransforms(RSMRIDataTransforms):
    """Transforms for the MRI segmentation task.

    .. note::
        Extends :class:`atommic.collections.multitask.rs.parts.transforms.RSMRIDataTransforms`.
    """


class SegmentationCTDataTransforms(RSCTDataTransforms):
    """Transforms for the CT segmentation task.

    .. note::
        Extends :class:`atommic.collections.multitask.rs.parts.transforms.RSCTDataTransforms`.
    """
