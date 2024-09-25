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
import yaml  # type: ignore
from nibabel.filebasedimages import FileBasedImage
from torch.utils.data import Dataset

from atommic.collections.common.data.mri_loader import MRIDataset
from atommic.collections.common.parts.utils import is_none


class SegmentationMRIDataset(MRIDataset):
    """A dataset class for MRI segmentation.

    Examples
    --------
    >>> from atommic.collections.segmentation.data.mri_segmentation_loader import SegmentationMRIDataset
    >>> dataset = SegmentationMRIDataset(root='data/train', sample_rate=0.1)
    >>> print(len(dataset))
    100
    >>> kspace, imspace, coil_sensitivities, mask, initial_prediction, segmentation_labels, attrs, filename, \
    slice_num = dataset[0]
    >>> print(kspace.shape)
    np.array([30, 640, 368])

    .. note::
        Extends :class:`atommic.collections.common.data.mri_loader.MRIDataset`.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        coil_sensitivity_maps_root: Union[str, Path, os.PathLike] = None,
        mask_root: Union[str, Path, os.PathLike] = None,
        noise_root: Union[str, Path, os.PathLike] = None,
        initial_predictions_root: Union[str, Path, os.PathLike] = None,
        dataset_format: str = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = None,
        num_cols: Optional[Tuple[int]] = None,
        consecutive_slices: int = 1,
        data_saved_per_slice: bool = False,
        n2r_supervised_rate: Optional[float] = 0.0,
        complex_target: bool = False,
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
        sense_root : Union[str, Path, os.PathLike], optional
            Path to the coil sensitivities maps dataset, if stored separately.
        mask_root : Union[str, Path, os.PathLike], optional
            Path to stored masks, if stored separately.
        noise_root : Union[str, Path, os.PathLike], optional
            Path to stored noise, if stored separately (in json format).
        initial_predictions_root : Union[str, Path, os.PathLike], optional
            Path to the dataset containing the initial predictions. If provided, the initial predictions will be used
            as the input of the reconstruction network. Default is ``None``.
        dataset_format : str, optional
            The format of the dataset. For example, ``'custom_dataset'`` or ``'public_dataset_name'``.
            Default is ``None``.
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
        dataset_cache_file : Union[str, Path, os.PathLike], optional
            A file in which to cache dataset information for faster load times.
        num_cols : Optional[Tuple[int]], optional
            If provided, only slices with the desired number of columns will be considered.
        consecutive_slices : int, optional
            An int (>0) that determine the amount of consecutive slices of the file to be loaded at the same time.
            Default is ``1``, loading single slices.
        data_saved_per_slice : bool, optional
            Whether the data is saved per slice or per volume.
        n2r_supervised_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the subjects should be loaded for Noise to
            Reconstruction (N2R) supervised loss, if N2R is enabled. Default is ``0.0``.
        complex_target : bool, optional
            Whether the target is complex. Default is ``False``.
        log_images_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the subjects should be logged as images. Default is
            ``1.0``.
        transform : Optional[Callable], optional
            A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function
            should take ``kspace``, ``coil sensitivity maps``, ``mask``, ``initial prediction``, ``segmentation``,
            ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test
            data. Default is ``None``.
        segmentations_root : Union[str, Path, os.PathLike], optional
            Path to the dataset containing the segmentations.
        segmentation_classes : int, optional
            The number of segmentation classes. Default is ``2``.
        segmentation_classes_to_remove : Optional[Tuple[int]], optional
            A tuple of segmentation classes to remove. For example, if the dataset contains segmentation classes
            0, 1, 2, 3, and 4, and you want to remove classes 1 and 3, set this to ``(1, 3)``. Default is ``None``.
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
            coil_sensitivity_maps_root,
            mask_root,
            noise_root,
            initial_predictions_root,
            dataset_format,
            sample_rate,
            volume_sample_rate,
            use_dataset_cache,
            dataset_cache_file,
            num_cols,
            consecutive_slices,
            data_saved_per_slice,
            n2r_supervised_rate,
            complex_target,
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
            if isinstance(self.segmentation_classes_to_combine[0], int):
                segmentation_labels_to_combine = np.stack(
                    [segmentation_labels[..., x] for x in self.segmentation_classes_to_combine], axis=-1
                ).sum(axis=-1, keepdims=True)
                segmentation_labels_to_keep = np.delete(
                    segmentation_labels, self.segmentation_classes_to_combine, axis=-1
                )
            else:
                # In case we want to combine more classes separately
                segmentation_labels_to_combine = []
                for classes in self.segmentation_classes_to_combine:
                    segmentation_labels_to_combine.append(
                        np.stack([segmentation_labels[..., x] for x in classes], axis=-1).sum(axis=-1, keepdims=True)
                    )
                segmentation_labels_to_combine = np.concatenate(segmentation_labels_to_combine, axis=-1)
                segmentation_labels_to_keep = np.delete(
                    segmentation_labels,
                    [x for classes in self.segmentation_classes_to_combine for x in classes],
                    axis=-1,
                )

            if self.segmentation_classes_to_remove is not None and 0 in self.segmentation_classes_to_remove:
                # if background is removed, we can stack the combined labels with the rest straight away
                segmentation_labels = np.concatenate(
                    [segmentation_labels_to_combine, segmentation_labels_to_keep], axis=-1
                )
            elif segmentation_labels[..., 0].sum() == 0:
                # if background is not removed, we need to add it back as new background channel
                segmentation_labels = np.concatenate(
                    [segmentation_labels[..., 0:1], segmentation_labels_to_combine, segmentation_labels_to_keep],
                    axis=-1,
                )
            else:
                segmentation_labels = np.concatenate(
                    [segmentation_labels_to_combine, segmentation_labels_to_keep], axis=-1
                )

            segmentation_labels = segmentation_labels.astype(np.int8)

        # check if we need to separate any classes, e.g. pathologies from White Matter and Gray Matter
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
        """Get item from :class:`SegmentationMRIDataset`."""
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            if self.complex_data:
                kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

                sensitivity_map = np.array([])
                if "sensitivity_map" in hf:
                    sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(
                        np.complex64
                    )
                elif "maps" in hf:
                    sensitivity_map = self.get_consecutive_slices(hf, "maps", dataslice).astype(np.complex64)
                elif self.coil_sensitivity_maps_root is not None and self.coil_sensitivity_maps_root != "None":
                    coil_sensitivity_maps_root = self.coil_sensitivity_maps_root
                    split_dir = str(fname).split("/")
                    # check if exists
                    if not os.path.exists(Path(f"{coil_sensitivity_maps_root}/{split_dir[-2]}/{fname.name}")):
                        # find to what depth the coil_sensitivity_maps_root directory is nested
                        for j in range(len(split_dir)):
                            # get the coil_sensitivity_maps_root directory name
                            coil_sensitivity_maps_root = Path(f"{self.coil_sensitivity_maps_root}/{split_dir[-j]}/")
                            if os.path.exists(coil_sensitivity_maps_root / Path(split_dir[-2]) / fname.name):
                                break
                    # load coil sensitivity maps
                    with h5py.File(Path(coil_sensitivity_maps_root) / Path(split_dir[-2]) / fname.name, "r") as sf:
                        if "sensitivity_map" in sf or "sensitivity_map" in next(iter(sf.keys())):
                            sensitivity_map = (
                                self.get_consecutive_slices(sf, "sensitivity_map", dataslice)
                                .squeeze()
                                .astype(np.complex64)
                            )

                mask = None
                if "mask" in hf:
                    mask = np.asarray(self.get_consecutive_slices(hf, "mask", dataslice))
                    if mask.ndim == 3:
                        mask = mask[dataslice]
                elif self.mask_root is not None and self.mask_root != "None":
                    with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                        mask = np.asarray(self.get_consecutive_slices(mf, "mask", dataslice))

                imspace = np.empty([])

            elif not self.complex_data:
                if "reconstruction" in hf:
                    imspace = self.get_consecutive_slices(hf, "reconstruction", dataslice)
                elif "target" in hf:
                    imspace = self.get_consecutive_slices(hf, "target", dataslice)
                else:
                    raise ValueError(
                        "Complex data has not been selected but no reconstruction or target data found in file. "
                        "Only 'reconstruction' and 'target' keys are supported."
                    )
                kspace = np.empty([])
                sensitivity_map = np.array([])
                mask = np.empty([])

            segmentation_labels = np.empty([])
            if self.segmentations_root is not None and self.segmentations_root != "None":
                with h5py.File(Path(self.segmentations_root) / fname.name, "r") as sf:
                    segmentation_labels = np.asarray(self.get_consecutive_slices(sf, "segmentation", dataslice))
                    segmentation_labels = self.process_segmentation_labels(segmentation_labels)
            elif "segmentation" in hf:
                segmentation_labels = np.asarray(self.get_consecutive_slices(hf, "segmentation", dataslice))
                segmentation_labels = self.process_segmentation_labels(segmentation_labels)

            initial_prediction = np.empty([])
            if not is_none(self.initial_predictions_root):
                with h5py.File(Path(self.initial_predictions_root) / fname.name, "r") as ipf:  # type: ignore
                    if "reconstruction" in hf:
                        initial_prediction = (
                            self.get_consecutive_slices(ipf, "reconstruction", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
                    elif "initial_prediction" in hf:
                        initial_prediction = (
                            self.get_consecutive_slices(ipf, "initial_prediction", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
            else:
                if "reconstruction" in hf:
                    initial_prediction = (
                        self.get_consecutive_slices(hf, "reconstruction", dataslice).squeeze().astype(np.complex64)
                    )
                elif "initial_prediction" in hf:
                    initial_prediction = (
                        self.get_consecutive_slices(hf, "initial_prediction", dataslice).squeeze().astype(np.complex64)
                    )

            attrs = dict(hf.attrs)

            # get noise level for current slice, if metadata["noise_levels"] is not empty
            if "noise_levels" in metadata and len(metadata["noise_levels"]) > 0:
                metadata["noise"] = metadata["noise_levels"][dataslice]
            else:
                metadata["noise"] = 1.0

            attrs.update(metadata)

        if sensitivity_map.shape != kspace.shape and sensitivity_map.ndim > 1:
            if sensitivity_map.ndim == 3:
                sensitivity_map = np.transpose(sensitivity_map, (2, 0, 1))
            elif sensitivity_map.ndim == 4:
                sensitivity_map = np.transpose(sensitivity_map, (0, 3, 1, 2))
            else:
                raise ValueError(
                    f"Sensitivity map has invalid dimensions {sensitivity_map.shape} compared to kspace {kspace.shape}"
                )

        attrs["log_image"] = bool(dataslice in self.indices_to_log)

        return (
            (
                kspace,
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                kspace,
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
        )


class BraTS2023AdultGliomaSegmentationMRIDataset(Dataset):
    """Supports the BraTS2023AdultGlioma dataset for MRI segmentation.

    .. note::
        Extends :class:`torch.utils.data.Dataset`.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        coil_sensitivity_maps_root: Union[str, Path, os.PathLike] = None,  # pylint: disable=unused-argument
        mask_root: Union[str, Path, os.PathLike] = None,  # pylint: disable=unused-argument
        noise_root: Union[str, Path, os.PathLike] = None,  # pylint: disable=unused-argument
        initial_predictions_root: Union[str, Path, os.PathLike] = None,
        dataset_format: str = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = None,
        num_cols: Optional[Tuple[int]] = None,
        consecutive_slices: int = 1,
        data_saved_per_slice: bool = False,
        n2r_supervised_rate: Optional[float] = 0.0,  # pylint: disable=unused-argument
        complex_target: bool = False,
        log_images_rate: Optional[float] = 1.0,
        transform: Optional[Callable] = None,
        segmentations_root: Union[str, Path, os.PathLike] = None,
        segmentation_classes: int = 2,
        segmentation_classes_to_remove: Optional[Tuple[int]] = None,
        segmentation_classes_to_combine: Optional[Tuple[int]] = None,
        segmentation_classes_to_separate: Optional[Tuple[int]] = None,
        segmentation_classes_thresholds: Optional[Tuple[float]] = None,
        complex_data: bool = True,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Inits :class:`BraTS2023AdultGliomaSegmentationMRIDataset`.

        Parameters
        ----------
        root : Union[str, Path, os.PathLike]
            Path to the dataset.
        sense_root : Union[str, Path, os.PathLike], optional
            Path to the coil sensitivities maps dataset, if stored separately.
        mask_root : Union[str, Path, os.PathLike], optional
            Path to stored masks, if stored separately.
        noise_root : Union[str, Path, os.PathLike], optional
            Path to stored noise, if stored separately (in json format).
        initial_predictions_root : Union[str, Path, os.PathLike], optional
            Path to the dataset containing the initial predictions. If provided, the initial predictions will be used
            as the input of the reconstruction network. Default is ``None``.
        dataset_format : str, optional
            The format of the dataset. For example, ``'custom_dataset'`` or ``'public_dataset_name'``.
            Default is ``None``.
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
        dataset_cache_file : Union[str, Path, os.PathLike], optional
            A file in which to cache dataset information for faster load times.
        num_cols : Optional[Tuple[int]], optional
            If provided, only slices with the desired number of columns will be considered.
        consecutive_slices : int, optional
            An int (>0) that determine the amount of consecutive slices of the file to be loaded at the same time.
            Default is ``1``, loading single slices.
        data_saved_per_slice : bool, optional
            Whether the data is saved per slice or per volume.
        n2r_supervised_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the subjects should be loaded for Noise to
            Reconstruction (N2R) supervised loss, if N2R is enabled. Default is ``0.0``.
        complex_target : bool, optional
            Whether the target is complex. Default is ``False``.
        log_images_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the subjects should be logged as images. Default is
            ``1.0``.
        transform : Optional[Callable], optional
            A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function
            should take ``kspace``, ``coil sensitivity maps``, ``mask``, ``initial prediction``, ``segmentation``,
            ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test
            data. Default is ``None``.
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
        super().__init__()
        self.initial_predictions_root = initial_predictions_root
        self.dataset_format = dataset_format

        # set default sampling mode if none given
        if not is_none(sample_rate) and not is_none(volume_sample_rate):
            raise ValueError(
                f"Both sample_rate {sample_rate} and volume_sample_rate {volume_sample_rate} are set. "
                "Please set only one of them."
            )

        if sample_rate is None or sample_rate == "None":
            sample_rate = 1.0

        if volume_sample_rate is None or volume_sample_rate == "None":
            volume_sample_rate = 1.0

        self.dataset_cache_file = None if is_none(dataset_cache_file) else Path(dataset_cache_file)  # type: ignore

        if self.dataset_cache_file is not None and self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = yaml.safe_load(f)
        else:
            dataset_cache = {}

        if consecutive_slices < 1:
            raise ValueError(f"Consecutive slices {consecutive_slices} is out of range, must be > 0.")
        self.consecutive_slices = consecutive_slices
        self.complex_target = complex_target
        self.transform = transform
        self.data_saved_per_slice = data_saved_per_slice

        self.examples = []

        # Check if our dataset is in the cache. If yes, use that metadata, if not, then regenerate the metadata.
        if dataset_cache.get(root) is None or not use_dataset_cache:
            if str(root).endswith(".json"):
                with open(root, "r") as f:  # type: ignore  # pylint: disable=unspecified-encoding
                    examples = json.load(f)
                files = [Path(example) for example in examples]
            else:
                files = list(Path(root).iterdir())

            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                # Specific to SKM-TEA segmentation dataset, we need to remove the first 50 and last 65 slices
                self.examples += [
                    (fname, slice_ind, metadata) for slice_ind in range(num_slices) if 50 < slice_ind < num_slices - 65
                ]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info("Saving dataset cache to %s.", self.dataset_cache_file)
                with open(self.dataset_cache_file, "wb") as f:  # type: ignore
                    yaml.dump(dataset_cache, f)
        else:
            logging.info("Using dataset cache from %s.", self.dataset_cache_file)
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list({f[0].stem for f in self.examples}))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [example for example in self.examples if example[0].stem in sampled_vols]

        if num_cols and not is_none(num_cols):
            self.examples = [ex for ex in self.examples if ex[2]["encoding_size"][1] in num_cols]

        self.indices_to_log = np.random.choice(
            len(self.examples), int(log_images_rate * len(self.examples)), replace=False  # type: ignore
        )

        self.segmentations_root = segmentations_root
        self.consecutive_slices = consecutive_slices
        self.segmentation_classes = segmentation_classes
        self.segmentation_classes_to_remove = segmentation_classes_to_remove
        self.segmentation_classes_to_combine = segmentation_classes_to_combine
        self.segmentation_classes_to_separate = segmentation_classes_to_separate
        self.segmentation_classes_thresholds = segmentation_classes_thresholds
        self.complex_data = complex_data

    @staticmethod
    def __read_nifti__(nifti_path: Union[str, Path]) -> FileBasedImage:
        """Read a nifti file.

        Parameters
        ----------
        nifti_path : Union[str, Path]
            The path to the nifti file.

        Returns
        -------
        nib.Nifti1Image
            The nifti file.
        """
        return nib.load(nifti_path)

    def _retrieve_metadata(self, fname: Union[str, Path]) -> Tuple[Dict, int]:
        """Retrieve metadata from a given file.

        Parameters
        ----------
        fname : Union[str, Path]
            Path to file.

        Returns
        -------
        Tuple[Dict, int]
            Metadata dictionary and number of slices in the file.
        """
        data = self.__read_nifti__(fname)
        num_slices = data.header["dim"][4]
        # compute the mean and std of the data
        metadata = {
            "padding_left": 0,
            "padding_right": 0,
            "encoding_size": 0,
            "recon_size": 0,
            "num_slices": num_slices,
        }
        return metadata, num_slices

    def get_consecutive_slices(self, data: Dict, key: str, dataslice: int) -> np.ndarray:
        """Get consecutive slices from a given data dictionary.

        Parameters
        ----------
        data : dict
            Data to extract slices from.
        key : str
            Key to extract slices from.
        dataslice : int
            Slice to index.

        Returns
        -------
        np.ndarray
            Array of consecutive slices. If ``self.consecutive_slices`` is > 1, then the array will have shape
            ``(self.consecutive_slices, *data[key].shape[1:])``. Otherwise, the array will have shape
            ``data[key].shape[1:]``.

        Examples
        --------
        >>> data = {"kspace": np.random.rand(10, 640, 368)}
        >>> from atommic.collections.common.data.mri_loader import MRIDataset
        >>> MRIDataset.get_consecutive_slices(data, "kspace", 1).shape
        (1, 640, 368)
        >>> MRIDataset.get_consecutive_slices(data, "kspace", 5).shape
        (5, 640, 368)
        """
        # read data
        x = data[key]

        if self.data_saved_per_slice:
            x = np.expand_dims(x, axis=0)

        if self.consecutive_slices == 1:
            if x.shape[0] == 1:
                return x[0]
            if x.ndim != 2:
                return x[dataslice]
            return x

        # get consecutive slices
        num_slices = x.shape[0]

        # If the number of consecutive slices is greater than or equal to the total slices, return the entire stack
        if self.consecutive_slices >= num_slices:
            # pad left and right with zero slices to match the desired number of slices
            slices_to_add_start = (self.consecutive_slices - num_slices) // 2
            slices_to_add_end = self.consecutive_slices - num_slices - slices_to_add_start
            if slices_to_add_start > 0:
                zero_slices = np.zeros((slices_to_add_start, *x.shape[1:]))
                x = np.concatenate((zero_slices, x), axis=0)
            if slices_to_add_end > 0:
                zero_slices = np.zeros((slices_to_add_end, *x.shape[1:]))
                x = np.concatenate((x, zero_slices), axis=0)
            return x

        # Calculate half of the consecutive slices to determine the middle position
        half_slices = self.consecutive_slices // 2

        # Determine the start and end slices based on the middle position
        start_slice = dataslice - half_slices
        end_slice = dataslice + half_slices + 1

        # Handle edge cases
        slices_to_add_start = 0
        slices_to_add_end = 0
        if start_slice < 0:
            slices_to_add_start = abs(start_slice)
            start_slice = 0

        if end_slice > (num_slices - 1):
            slices_to_add_end = end_slice - num_slices
            extracted_slices = x[start_slice:]
        else:
            extracted_slices = x[start_slice:end_slice]

        # Add slices to the start and end if needed
        if slices_to_add_start > 0:
            zero_slices = np.zeros((slices_to_add_start, *extracted_slices.shape[1:]))
            extracted_slices = np.concatenate((zero_slices, extracted_slices), axis=0)
        if slices_to_add_end > 0:
            zero_slices = np.zeros((slices_to_add_end, *extracted_slices.shape[1:]))
            extracted_slices = np.concatenate((extracted_slices, zero_slices), axis=0)

        return extracted_slices

    def __len__(self):
        """Length of :class:`MRIDataset`."""
        return len(self.examples)

    def __getitem__(self, i: int):
        """Get item from :class:`BraTS2023AdultGliomaSegmentationMRIDataset`."""
        fname, dataslice, metadata = self.examples[i]

        imspace = self.get_consecutive_slices(
            {"target": np.moveaxis(self.__read_nifti__(fname).get_fdata(), -1, 0)}, "target", dataslice
        ).astype(np.float32)

        segmentation_path = Path(self.segmentations_root) / Path(  # type: ignore
            str(fname.name).replace(".nii.gz", "-seg.nii.gz")
        )

        segmentation_labels = self.get_consecutive_slices(
            {"segmentation": np.moveaxis(self.__read_nifti__(segmentation_path).get_fdata(), -1, 0)},
            "segmentation",
            dataslice,
        )

        # Necrotic Tumor Core (NCR - label 1)
        ncr = np.zeros_like(segmentation_labels)
        ncr[segmentation_labels == 1] = 1
        # Peritumoral Edematous/Invaded Tissue (ED - label 2)
        ed = np.zeros_like(segmentation_labels)
        ed[segmentation_labels == 2] = 1
        # GD-Enhancing Tumor (ET - label 3)
        et = np.zeros_like(segmentation_labels)
        et[segmentation_labels == 3] = 1
        # Whole Tumor (WT — label 1, 2, or 3)
        wt = np.zeros_like(segmentation_labels)
        wt[segmentation_labels != 0] = 1

        segmentation_labels = np.stack([ncr, ed, et, wt], axis=0).astype(np.float32)

        if self.consecutive_slices > 1:
            segmentation_labels = np.moveaxis(segmentation_labels, 0, 1)

        kspace = np.empty([])
        target = imspace
        sensitivity_map = np.empty([])
        mask = np.empty([])
        initial_prediction = target

        attrs = {
            "log_image": bool(dataslice in self.indices_to_log),
            "noise": 1.0,
        }
        attrs.update(metadata)

        return (
            (
                kspace,
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                kspace,
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
        )


class ISLES2022SubAcuteStrokeSegmentationMRIDataset(SegmentationMRIDataset):
    """Supports the ISLES2022SubAcuteStroke dataset for MRI segmentation.

    .. note::
        Extends :class:`atommic.collections.segmentation.data.mri_segmentation_loader.SegmentationMRIDataset`.
    """

    @staticmethod
    def __read_nifti__(nifti_path: Union[str, Path]) -> FileBasedImage:
        """Read a nifti file.

        Parameters
        ----------
        nifti_path : Union[str, Path]
            The path to the nifti file.

        Returns
        -------
        nib.Nifti1Image
            The nifti file.
        """
        return nib.load(nifti_path)

    def _retrieve_metadata(self, fname: Union[str, Path]) -> Tuple[Dict, int]:
        """Retrieve metadata from a given file.

        Parameters
        ----------
        fname : Union[str, Path]
            Path to file.

        Returns
        -------
        Tuple[Dict, int]
            Metadata dictionary and number of slices in the file.
        """
        data = self.__read_nifti__(fname)
        num_slices = data.header["dim"][4]
        metadata = {
            "padding_left": 0,
            "padding_right": 0,
            "encoding_size": 0,
            "recon_size": 0,
            "num_slices": num_slices,
        }
        return metadata, num_slices

    def __getitem__(self, i: int):
        """Get item from :class:`ISLES2022SubAcuteStrokeSegmentationMRIDataset`."""
        fname, dataslice, metadata = self.examples[i]

        imspace = self.get_consecutive_slices(
            {"target": np.moveaxis(self.__read_nifti__(fname).get_fdata(), -1, 0)}, "target", dataslice
        ).astype(np.float32)

        if self.consecutive_slices > 1:
            imspace = np.moveaxis(imspace, 0, 1)

        # imspace has 3 channels, normalize each by its min and max values
        max_val = np.max(imspace[0]) if np.max(imspace[0]) > 0 else 1
        imspace[0] = (imspace[0] - np.min(imspace[0])) / (max_val - np.min(imspace[0]))
        max_val = np.max(imspace[1]) if np.max(imspace[1]) > 0 else 1
        imspace[1] = (imspace[1] - np.min(imspace[1])) / (max_val - np.min(imspace[1]))
        max_val = np.max(imspace[2]) if np.max(imspace[2]) > 0 else 1
        imspace[2] = (imspace[2] - np.min(imspace[2])) / (max_val - np.min(imspace[2]))
        # normalize all by min and max values of all channels
        imspace = (imspace - np.min(imspace)) / (np.max(imspace) - np.min(imspace))

        segmentation_path = Path(self.segmentations_root) / Path(  # type: ignore
            str(fname.name).replace(".nii.gz", "-seg.nii.gz")
        )
        segmentation_labels = self.get_consecutive_slices(
            {"segmentation": np.moveaxis(self.__read_nifti__(segmentation_path).get_fdata(), -1, 0)},
            "segmentation",
            dataslice,
        ).astype(np.float32)

        # Lesions (label 1)
        lesions = np.zeros_like(segmentation_labels)
        lesions[segmentation_labels == 1] = 1

        # stack lesions as a new channel
        segmentation_labels = np.stack([lesions], axis=0).astype(np.float32)

        if self.consecutive_slices > 1:
            # bring the segmentation classes dimension back to the first dimension
            imspace = np.moveaxis(imspace, 0, 1)
            segmentation_labels = np.moveaxis(segmentation_labels, 0, 1)

        kspace = np.empty([])
        target = imspace
        sensitivity_map = np.empty([])
        mask = np.empty([])
        initial_prediction = target

        attrs = {"log_image": bool(dataslice in self.indices_to_log), "noise": 1.0}
        attrs.update(metadata)

        return (
            (
                kspace,
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                kspace,
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
        )


class SKMTEASegmentationMRIDataset(Dataset):
    """Supports the SKM-TEA dataset for MRI segmentation.

    .. note::
        Extends :class:`torch.utils.data.Dataset`.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        coil_sensitivity_maps_root: Union[str, Path, os.PathLike] = None,  # pylint: disable=unused-argument
        mask_root: Union[str, Path, os.PathLike] = None,  # pylint: disable=unused-argument
        noise_root: Union[str, Path, os.PathLike] = None,  # pylint: disable=unused-argument
        initial_predictions_root: Union[str, Path, os.PathLike] = None,
        dataset_format: str = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = None,
        num_cols: Optional[Tuple[int]] = None,
        consecutive_slices: int = 1,
        data_saved_per_slice: bool = False,
        n2r_supervised_rate: Optional[float] = 0.0,  # pylint: disable=unused-argument
        complex_target: bool = False,
        log_images_rate: Optional[float] = 1.0,
        transform: Optional[Callable] = None,
        segmentations_root: Union[str, Path, os.PathLike] = None,
        segmentation_classes: int = 2,
        segmentation_classes_to_remove: Optional[Tuple[int]] = None,
        segmentation_classes_to_combine: Optional[Tuple[int]] = None,
        segmentation_classes_to_separate: Optional[Tuple[int]] = None,
        segmentation_classes_thresholds: Optional[Tuple[float]] = None,
        complex_data: bool = True,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Inits :class:`SKMTEASegmentationMRIDataset`.

        Parameters
        ----------
        root : Union[str, Path, os.PathLike]
            Path to the dataset.
        sense_root : Union[str, Path, os.PathLike], optional
            Path to the coil sensitivities maps dataset, if stored separately.
        mask_root : Union[str, Path, os.PathLike], optional
            Path to stored masks, if stored separately.
        noise_root : Union[str, Path, os.PathLike], optional
            Path to stored noise, if stored separately (in json format).
        initial_predictions_root : Union[str, Path, os.PathLike], optional
            Path to the dataset containing the initial predictions. If provided, the initial predictions will be used
            as the input of the reconstruction network. Default is ``None``.
        dataset_format : str, optional
            The format of the dataset. For example, ``'custom_dataset'`` or ``'public_dataset_name'``.
            Default is ``None``.
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
        dataset_cache_file : Union[str, Path, os.PathLike], optional
            A file in which to cache dataset information for faster load times.
        num_cols : Optional[Tuple[int]], optional
            If provided, only slices with the desired number of columns will be considered.
        consecutive_slices : int, optional
            An int (>0) that determine the amount of consecutive slices of the file to be loaded at the same time.
            Default is ``1``, loading single slices.
        data_saved_per_slice : bool, optional
            Whether the data is saved per slice or per volume.
        n2r_supervised_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the subjects should be loaded for Noise to
            Reconstruction (N2R) supervised loss, if N2R is enabled. Default is ``0.0``.
        complex_target : bool, optional
            Whether the target is complex. Default is ``False``.
        log_images_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the subjects should be logged as images. Default is
            ``1.0``.
        transform : Optional[Callable], optional
            A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function
            should take ``kspace``, ``coil sensitivity maps``, ``mask``, ``initial prediction``, ``segmentation``,
            ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test
            data. Default is ``None``.
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
        super().__init__()
        self.initial_predictions_root = initial_predictions_root
        self.dataset_format = dataset_format

        # set default sampling mode if none given
        if not is_none(sample_rate) and not is_none(volume_sample_rate):
            raise ValueError(
                f"Both sample_rate {sample_rate} and volume_sample_rate {volume_sample_rate} are set. "
                "Please set only one of them."
            )

        if sample_rate is None or sample_rate == "None":
            sample_rate = 1.0

        if volume_sample_rate is None or volume_sample_rate == "None":
            volume_sample_rate = 1.0

        self.dataset_cache_file = None if is_none(dataset_cache_file) else Path(dataset_cache_file)  # type: ignore

        if self.dataset_cache_file is not None and self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = yaml.safe_load(f)
        else:
            dataset_cache = {}

        if consecutive_slices < 1:
            raise ValueError(f"Consecutive slices {consecutive_slices} is out of range, must be > 0.")
        self.consecutive_slices = consecutive_slices
        self.complex_target = complex_target
        self.transform = transform
        self.data_saved_per_slice = data_saved_per_slice

        self.examples = []

        # Check if our dataset is in the cache. If yes, use that metadata, if not, then regenerate the metadata.
        if dataset_cache.get(root) is None or not use_dataset_cache:
            if str(root).endswith(".json"):
                with open(root, "r") as f:  # type: ignore  # pylint: disable=unspecified-encoding
                    examples = json.load(f)
                files = [Path(example) for example in examples]
            else:
                files = list(Path(root).iterdir())

            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                # Specific to SKM-TEA segmentation dataset, we need to remove the first and last 30 slices
                self.examples += [
                    (fname, slice_ind, metadata) for slice_ind in range(num_slices) if 30 < slice_ind < num_slices - 30
                ]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info("Saving dataset cache to %s.", self.dataset_cache_file)
                with open(self.dataset_cache_file, "wb") as f:  # type: ignore
                    yaml.dump(dataset_cache, f)
        else:
            logging.info("Using dataset cache from %s.", self.dataset_cache_file)
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list({f[0].stem for f in self.examples}))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [example for example in self.examples if example[0].stem in sampled_vols]

        if num_cols and not is_none(num_cols):
            self.examples = [ex for ex in self.examples if ex[2]["encoding_size"][1] in num_cols]

        self.indices_to_log = np.random.choice(
            len(self.examples), int(log_images_rate * len(self.examples)), replace=False  # type: ignore
        )

        self.segmentations_root = segmentations_root
        self.consecutive_slices = consecutive_slices
        self.segmentation_classes = segmentation_classes
        self.segmentation_classes_to_remove = segmentation_classes_to_remove
        self.segmentation_classes_to_combine = segmentation_classes_to_combine
        self.segmentation_classes_to_separate = segmentation_classes_to_separate
        self.segmentation_classes_thresholds = segmentation_classes_thresholds
        self.complex_data = complex_data

    def _retrieve_metadata(self, fname: Union[str, Path]) -> Tuple[Dict, int]:
        """Override the ``_retrieve_metadata`` method to handle the SKM-TEA dataset.

        .. note::
            Overrides :meth:`atommic.collections.common.data.mri_loader.MRIDataset._retrieve_metadata`.
        """
        with h5py.File(fname, "r") as hf:
            shape = hf["seg"].shape
        num_slices = shape[2]
        metadata = {
            "padding_left": 0,
            "padding_right": 0,
            "encoding_size": 0,
            "recon_size": 0,
            "num_slices": num_slices,
        }
        return metadata, num_slices

    def get_consecutive_slices(self, data: Dict, key: str, dataslice: int) -> np.ndarray:
        """Override the ``get_consecutive_slices`` method to handle the SKM-TEA dataset.

        .. note::
            Overrides :meth:`atommic.collections.common.data.mri_loader.MRIDataset.get_consecutive_slices`.
        """
        x = data[key]

        if self.consecutive_slices == 1:
            if x.shape[2] == 1:
                return x[:, :, 0]
            if x.ndim != 2:
                return x[:, :, dataslice]
            return x

        # get consecutive slices
        num_slices = x.shape[2]

        # If the number of consecutive slices is greater than or equal to the total slices, return the entire stack
        if self.consecutive_slices >= num_slices:
            # pad left and right with zero slices to match the desired number of slices
            slices_to_add_start = (self.consecutive_slices - num_slices) // 2
            slices_to_add_end = self.consecutive_slices - num_slices - slices_to_add_start
            if slices_to_add_start > 0:
                zero_slices = np.zeros((x.shape[0], x.shape[1], slices_to_add_start))
                x = np.concatenate((zero_slices, x), axis=2)
            if slices_to_add_end > 0:
                zero_slices = np.zeros((x.shape[0], x.shape[1], slices_to_add_end))
                x = np.concatenate((x, zero_slices), axis=2)
            return x

        # Calculate half of the consecutive slices to determine the middle position
        half_slices = self.consecutive_slices // 2

        # Determine the start and end slices based on the middle position
        start_slice = dataslice - half_slices
        end_slice = dataslice + half_slices + 1

        # Handle edge cases
        slices_to_add_start = 0
        slices_to_add_end = 0
        if start_slice < 0:
            slices_to_add_start = abs(start_slice)
            start_slice = 0

        if end_slice > (num_slices - 1):
            slices_to_add_end = end_slice - num_slices
            extracted_slices = x[:, :, start_slice:]
        else:
            extracted_slices = x[:, :, start_slice:end_slice]

        # Add slices to the start and end if needed
        if slices_to_add_start > 0:
            zero_slices = np.zeros((x.shape[0], x.shape[1], slices_to_add_start))
            extracted_slices = np.concatenate((zero_slices, extracted_slices), axis=2)
        if slices_to_add_end > 0:
            zero_slices = np.zeros((x.shape[0], x.shape[1], slices_to_add_end))
            extracted_slices = np.concatenate((extracted_slices, zero_slices), axis=2)

        return extracted_slices

    def __len__(self):
        """Length of :class:`MRIDataset`."""
        return len(self.examples)

    def __getitem__(self, i: int):
        """Get item from :class:`SKMTEASegmentationMRIDataset`."""
        fname, dataslice, metadata = self.examples[i]
        dataset_format = self.dataset_format.lower()  # type: ignore
        with h5py.File(fname, "r") as hf:
            attrs = dict(hf.attrs)
            stats = hf["stats"]

            target = None
            metadata = {}

            if dataset_format == "skm-tea-echo1":
                target_key = "echo1"
            elif dataset_format == "skm-tea-echo2":
                target_key = "echo2"
            else:
                if dataset_format == "skm-tea-echo1+echo2":
                    target = np.abs(
                        self.get_consecutive_slices(hf, "echo1", dataslice).squeeze().astype(np.float32)
                        + self.get_consecutive_slices(hf, "echo2", dataslice).squeeze().astype(np.float32)
                    )
                elif dataset_format == "skm-tea-echo1+echo2-mc":
                    target = np.concatenate(
                        [
                            np.abs(self.get_consecutive_slices(hf, "echo1", dataslice).squeeze()).astype(np.float32),
                            np.abs(self.get_consecutive_slices(hf, "echo2", dataslice).squeeze()).astype(np.float32),
                        ],
                        axis=-1,
                    )
                elif dataset_format == "skm-tea-echo1+echo2-rss":
                    target = np.sqrt(
                        self.get_consecutive_slices(hf, "echo1", dataslice).squeeze().astype(np.float32) ** 2
                        + self.get_consecutive_slices(hf, "echo2", dataslice).squeeze().astype(np.float32) ** 2
                    )
                target_key = "rss"

            min_val = stats[target_key]["min"][()]
            max_val = stats[target_key]["max"][()]
            mean_val = stats[target_key]["mean"][()]
            std_val = stats[target_key]["std"][()]

            if target is None:
                target = np.abs(self.get_consecutive_slices(hf, target_key, dataslice).squeeze()).astype(np.float32)

            # Get the segmentation labels. They are stacked in the last dimension as follows:
            # 0: Patellar Cartilage, 1: Femoral Cartilage, 2: Lateral Tibial Cartilage, 3: Medial Tibial Cartilage,
            # 4: Lateral Meniscus, 5: Medial Meniscus
            segmentation_labels = self.get_consecutive_slices(hf, "seg", dataslice).astype(np.float32)

            # combine label 2 and 3 (Lateral Tibial Cartilage and Medial Tibial Cartilage)
            tibial_cartilage = segmentation_labels[..., 2] + segmentation_labels[..., 3]
            # combine label 4 and 5 (Lateral Meniscus and Medial Meniscus)
            medial_meniscus = segmentation_labels[..., 4] + segmentation_labels[..., 5]

            # stack the labels
            segmentation_labels = np.stack(
                [segmentation_labels[..., 0], segmentation_labels[..., 1], tibial_cartilage, medial_meniscus],
                axis=0,
            )

        if self.consecutive_slices > 1:
            # bring the consecutive slices dimension to the first dimension
            target = np.moveaxis(target, -1, 0)
            segmentation_labels = np.moveaxis(segmentation_labels, -1, 0)

        kspace = np.empty([])
        imspace = target
        sensitivity_map = np.empty([])
        mask = np.empty([])
        initial_prediction = target

        attrs.update(metadata)
        # set noise level to 1.0 as we handle fully sampled data
        attrs["noise"] = 1.0
        attrs["log_image"] = bool(dataslice in self.indices_to_log)

        # add min, max, mean, and std to attrs
        attrs["min"] = min_val
        attrs["max"] = max_val
        attrs["mean"] = mean_val
        attrs["std"] = std_val

        return (
            (
                kspace,
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                kspace,
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
        )
