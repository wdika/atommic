# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import yaml  # type: ignore
from nibabel.filebasedimages import FileBasedImage
from sympy.physics.control.control_plots import matplotlib
from torch.utils.data import Dataset

from atommic.collections.common.data.ct_loader import CTDataset
from atommic.collections.common.parts.utils import is_none


class SegmentationCTDataset(CTDataset):
    """A dataset class for CT segmentation.

    Examples
    --------
    >>> from atommic.collections.segmentation.data.ct_segmentation_loader import SegmentationCTDataset
    >>> dataset = SegmentationCTDataset(root='data/train', sample_rate=0.1)
    >>> print(len(dataset))
    100
    >>> target, segmentation_labels, attrs, filename, slice_num = dataset[0]
    >>> print(target.shape)
    np.array([30, 500, 500])

    .. note::
        Extends :class:`atommic.collections.common.data.ct_loader.CTDataset`.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        dataset_format: str = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = None,
        consecutive_slices: int = 1,
        data_saved_per_slice: bool = False,
        log_images_rate: Optional[float] = 1.0,
        transform: Optional[Callable] = None,
        segmentations_root: Union[str, Path, os.PathLike] = None,
        segmentation_classes: int = 2,
        segmentation_classes_to_remove: Optional[Tuple[int]] = None,
        segmentation_classes_to_combine: Optional[Tuple[int]] = None,
        segmentation_classes_to_separate: Optional[Tuple[int]] = None,
        segmentation_classes_thresholds: Optional[Tuple[float]] = None,
        complex_data: bool = True,
        **kwargs,
    ):
        """Inits :class:`SegmentationMRIDataset`.

        Parameters
        ----------
        root : Union[str, Path, os.PathLike]
            Path to the dataset.
        dataset_format : str, optional
            The format of the dataset. For example, ``'custom_dataset'`` or ``'public_dataset_name'``.
        sample_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the slices should be loaded. When creating
            subsampled datasets either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes)
            but not both.
        volume_sample_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the volumes should be loaded. When creating
            subsampled datasets either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes)
            but not both.
        use_dataset_cache : bool, optional
            Whether to cache dataset metadata. This is very useful for large datasets.
        dataset_cache_file : Union[str, Path, os.PathLike, none], optional
            A file in which to cache dataset information for faster load times. If not provided, the cache will be
            stored in the dataset root.
        consecutive_slices : int, optional
            An int (>0) that determine the amount of consecutive slices of the file to be loaded at the same time.
            Default is ``1``, loading single slices.
        data_saved_per_slice : bool, optional
            Whether the data is saved as series of slices. If saved as series of slices, data_saved_per_slice should be
            set to ``True``. Default is ``False``.
        log_images_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the slices should be logged as images. Default is
            ``1.0``.
        transform : Optional[Callable], optional
            A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function
            should take ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. Default is ``None``.
        segmentations_root : Union[str, Path, os.PathLike], optional
            Path to the dataset containing the segmentations.
        segmentation_classes : int, optional
            The number of segmentation classes. Default is ``2``.
        segmentation_classes_to_remove : Optional[Tuple[int]], optional
            A tuple of segmentation classes to remove. For example, if the dataset contains segmentation classes
            0, 1, 2,
            3, and 4, and you want to remove classes 1 and 3, set this to ``(1, 3)``. Default is ``None``.
        segmentation_classes_to_combine : Optional[Tuple[int]], optional
            A tuple of segmentation classes to combine. For example, if the dataset contains segmentation classes
            0, 1, 2, 3, and 4, and you want to combine classes 1 and 3, set this to ``(1, 3)``. Default is ``None``.
        segmentation_classes_to_separate : Optional[Tuple[int]], optional
            A tuple of segmentation classes to separate. For example, if the dataset contains segmentation classes
            0, 1, 2, 3, and 4, and you want to separate class 1 into 2 classes, set this to ``(1, 2)``.
            Default is ``None``.
        segmentation_classes_thresholds : Optional[Tuple[float]], optional
            A tuple of thresholds for the segmentation classes. For example, if the dataset contains segmentation
            classes 0, 1, 2, 3, and 4, and you want to set the threshold for class 1 to 0.5, set this to
            ``(0.5, 0.5, 0.5, 0.5, 0.5)``. Default is ``None``.
        complex_data : bool, optional
            Whether the data is complex. If ``False``, the data is assumed to be magnitude only. Default is ``True``.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(
            root,
            dataset_format,
            sample_rate,
            volume_sample_rate,
            use_dataset_cache,
            dataset_cache_file,
            consecutive_slices,
            data_saved_per_slice,
            log_images_rate,
            transform,
            **kwargs,
        )

        self.segmentations_root = segmentations_root
        self.consecutive_slices = consecutive_slices
        self.segmentation_classes = segmentation_classes
        self.segmentation_classes_to_remove = segmentation_classes_to_remove
        self.segmentation_classes_to_combine = segmentation_classes_to_combine
        self.segmentation_classes_to_separate = segmentation_classes_to_separate
        self.segmentation_classes_thresholds = segmentation_classes_thresholds
        self.complex_data = complex_data

    def process_segmentation_labels(self, segmentation_labels: np.ndarray) -> np.ndarray:  # noqa: MC0001
        """Process segmentation labels to remove, combine, and separate classes.

        Parameters
        ----------
        segmentation_labels : np.ndarray
            The segmentation labels. The shape should be (num_slices, height, width) or (height, width).

        Returns
        -------
        np.ndarray
            The processed segmentation labels.
        """
        # find the dimension with the segmentation classes
        segmentation_labels_dim = segmentation_labels.ndim - 1
        for dim in range(segmentation_labels.ndim):
            if segmentation_labels.shape[dim] == self.segmentation_classes:
                segmentation_labels_dim = dim

        # move it to the last dimension
        segmentation_labels = np.moveaxis(segmentation_labels, segmentation_labels_dim, -1)

        # if we have a single slice, add a new dimension
        if segmentation_labels.ndim == 2:
            segmentation_labels = np.expand_dims(segmentation_labels, axis=0)

        # check if we need to remove any classes, e.g. background
        if self.segmentation_classes_to_remove is not None:
            segmentation_labels = np.delete(segmentation_labels, self.segmentation_classes_to_remove, axis=-1)

        # check if we need to combine any classes, e.g. White Matter and Gray Matter
        if self.segmentation_classes_to_combine is not None:
            segmentation_labels_to_combine = np.sum(
                segmentation_labels[..., self.segmentation_classes_to_combine], axis=-1, keepdims=True
            )
            segmentation_labels_to_keep = np.delete(segmentation_labels, self.segmentation_classes_to_combine, axis=-1)

            if self.segmentation_classes_to_remove is not None and 0 in self.segmentation_classes_to_remove:
                # if background is removed, we can stack the combined labels with the rest straight away
                segmentation_labels = np.concatenate(
                    [segmentation_labels_to_combine, segmentation_labels_to_keep], axis=-1
                )
            else:
                # if background is not removed, we need to add it back as new background channel
                segmentation_labels = np.concatenate(
                    [segmentation_labels[..., 0:1], segmentation_labels_to_combine, segmentation_labels_to_keep],
                    axis=-1,
                )

        # check if we need to separate any classes, e.g. pathologies
        if self.segmentation_classes_to_separate is not None:
            for x in self.segmentation_classes_to_separate:
                segmentation_class_to_separate = segmentation_labels[..., x]
                for i in range(segmentation_labels.shape[-1]):
                    if i == x:
                        continue
                    segmentation_labels[..., i][segmentation_class_to_separate > 0] = 0

        # threshold probability maps if any threshold is given
        if self.segmentation_classes_thresholds is not None:
            for i, voxel_thres in enumerate(self.segmentation_classes_thresholds):
                if voxel_thres is not None:
                    segmentation_labels[..., i][segmentation_labels[..., i] < voxel_thres] = 0
                    segmentation_labels[..., i][segmentation_labels[..., i] >= voxel_thres] = 1

        if self.consecutive_slices == 1:
            # bring the segmentation classes dimension back to the first dimension
            segmentation_labels = np.moveaxis(segmentation_labels, -1, 0)
        elif self.consecutive_slices > 1:
            # bring the segmentation classes dimension back to the second dimension
            segmentation_labels = np.moveaxis(segmentation_labels, -1, 1)

        return segmentation_labels

    def __getitem__(self, i: int):  # noqa: MC0001
        """Get item from :class:`SegmentationCTDataset`."""
        fname, dataslice, metadata = self.examples[i]

        image = sitk.ReadImage(str(fname))
        image = sitk.GetArrayFromImage(image)
        image = image[dataslice].astype(np.float32)

        if self.segmentations_root is not None and self.segmentations_root != "None":
            # read the segmentation labels
            segmentation_labels = sitk.ReadImage(f"{self.segmentations_root}/{fname.name}")
            segmentation_labels = sitk.GetArrayFromImage(segmentation_labels)
            segmentation_labels = np.asarray(self.get_consecutive_slices(segmentation_labels, dataslice))
            segmentation_labels = self.process_segmentation_labels(segmentation_labels)
        else:
            raise ValueError("Please provide the 'segmentations_path' to load the segmentation labels.")

        attrs = dict()
        attrs.update(metadata)
        attrs["log_image"] = bool(dataslice in self.indices_to_log)

        return (
            (
                image,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                image,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
        )
