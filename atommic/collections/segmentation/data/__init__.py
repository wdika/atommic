# coding=utf-8
__author__ = "Dimitris Karkalousos"

from atommic.collections.segmentation.data.ct_segmentation_loader import SegmentationCTDataset  # noqa: F401
from atommic.collections.segmentation.data.mri_segmentation_loader import (  # noqa: F401
    BraTS2023AdultGliomaSegmentationMRIDataset,
    ISLES2022SubAcuteStrokeSegmentationMRIDataset,
    SegmentationMRIDataset,
    SKMTEASegmentationMRIDataset,
)
