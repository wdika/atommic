# coding=utf-8
__author__ = "Dimitris Karkalousos"

import os
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import h5py
import nibabel as nib
import numpy as np

from atommic.collections.common.data.mri_loader import MRIDataset
from atommic.collections.common.parts.utils import is_none


class RSMRIDataset(MRIDataset):
    """A dataset class for accelerated-MRI reconstruction and MRI segmentation.

    Examples
    --------
    >>> from atommic.collections.multitask.rs.data.mrirs_loader import RSMRIDataset
    >>> dataset = RSMRIDataset(root='data/train', sample_rate=0.1)
    >>> print(len(dataset))
    100
    >>> kspace, imspace, coil_sensitivities, mask, initial_prediction, segmentation_labels, attrs, filename, \
    slice_num = dataset[0]
    >>> print(kspace.shape)
    np.array([30, 640, 368])

    .. note::
        Extends :class:`atommic.collections.common.data.MRIDataset`.
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
        """Inits :class:`RSMRIDataset`.

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
            Whether to use a complex target or not. Default is ``False``.
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

        # Create random number generator used for consecutive slice selection and set consecutive slice amount
        self.consecutive_slices = consecutive_slices
        self.segmentation_classes = segmentation_classes
        self.segmentation_classes_to_remove = segmentation_classes_to_remove
        self.segmentation_classes_to_combine = segmentation_classes_to_combine
        self.segmentation_classes_to_separate = segmentation_classes_to_separate
        self.segmentation_classes_thresholds = segmentation_classes_thresholds
        self.complex_data = complex_data

    def process_segmentation_labels(self, segmentation_labels: np.ndarray) -> np.ndarray:  # noqa: MC0001
        """Processes segmentation labels to remove, combine, and separate classes.

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
        """Get item from :class:`RSMRIDataset`."""
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
                if "reconstruction_rss" in hf:
                    imspace = self.get_consecutive_slices(hf, "reconstruction_rss", dataslice)
                elif "reconstruction_sense" in hf:
                    imspace = self.get_consecutive_slices(hf, "reconstruction_sense", dataslice)
                elif "reconstruction" in hf:
                    imspace = self.get_consecutive_slices(hf, "reconstruction", dataslice)
                elif "target" in hf:
                    imspace = self.get_consecutive_slices(hf, "target", dataslice)
                else:
                    raise ValueError(
                        "Complex data has not been selected but no reconstruction data found in file. "
                        "Only 'reconstruction' key is supported."
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


class SKMTEARSMRIDataset(RSMRIDataset):
    """Supports the SKM-TEA dataset for multitask accelerated MRI reconstruction and MRI segmentation.

    .. note::
        Extends :class:`atommic.collections.multitask.rs.data.mrirs_loader.RSMRIDataset`.
    """

    def __getitem__(self, i: int):  # noqa: MC0001
        """Get item from :class:`SKMTEARSMRIDataset`."""
        if not is_none(self.dataset_format):
            dataset_format = self.dataset_format.lower()  # type: ignore
            masking = "default"
            if "custom_masking" in dataset_format:
                masking = "custom"
                dataset_format = dataset_format.replace("custom_masking", "").strip("_")
        else:
            dataset_format = None
            masking = "custom"

        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

            if not is_none(dataset_format) and dataset_format == "skm-tea-echo1":
                kspace = kspace[:, :, 0, :]
            elif not is_none(dataset_format) and dataset_format == "skm-tea-echo2":
                kspace = kspace[:, :, 1, :]
            elif not is_none(dataset_format) and dataset_format == "skm-tea-echo1+echo2":
                kspace = kspace[:, :, 0, :] + kspace[:, :, 1, :]
            elif not is_none(dataset_format) and dataset_format == "skm-tea-echo1+echo2-mc":
                kspace = np.concatenate([kspace[:, :, 0, :], kspace[:, :, 1, :]], axis=-1)
            else:
                warnings.warn(
                    f"Dataset format {dataset_format} is either not supported or set to None. "
                    "Using by default only the first echo."
                )
                kspace = kspace[:, :, 0, :]

            sensitivity_map = self.get_consecutive_slices(hf, "maps", dataslice).astype(np.complex64)

            if self.consecutive_slices > 1:
                sensitivity_map = sensitivity_map[:, 48:-48, 40:-40]
                kspace = kspace[:, 48:-48, 40:-40]
            else:
                sensitivity_map = sensitivity_map[48:-48, 40:-40]
                kspace = kspace[48:-48, 40:-40]

            if masking == "custom":
                mask = np.array([])
            else:
                masks = hf["masks"]
                mask = {}
                for key, val in masks.items():
                    mask[key.split("_")[-1].split(".")[0]] = np.asarray(val)

            # get the file format of the segmentation files
            segmentation_labels = nib.load(
                Path(self.segmentations_root) / Path(str(fname.name.split(".")[0]) + ".nii.gz")  # type: ignore
            ).get_fdata()

            # get a slice
            segmentation_labels = self.get_consecutive_slices({"seg": segmentation_labels}, "seg", dataslice)

            # Get the segmentation labels. They are valued as follows:
            # 0: Patellar Cartilage
            patellar_cartilage = np.zeros_like(segmentation_labels)
            patellar_cartilage[segmentation_labels == 1] = 1
            # 1: Femoral Cartilage
            femoral_cartilage = np.zeros_like(segmentation_labels)
            femoral_cartilage[segmentation_labels == 2] = 1
            # 2: Lateral Tibial Cartilage
            lateral_tibial_cartilage = np.zeros_like(segmentation_labels)
            lateral_tibial_cartilage[segmentation_labels == 3] = 1
            # 3: Medial Tibial Cartilage
            medial_tibial_cartilage = np.zeros_like(segmentation_labels)
            medial_tibial_cartilage[segmentation_labels == 4] = 1
            # 4: Lateral Meniscus
            lateral_meniscus = np.zeros_like(segmentation_labels)
            lateral_meniscus[segmentation_labels == 5] = 1
            # 5: Medial Meniscus
            medial_meniscus = np.zeros_like(segmentation_labels)
            medial_meniscus[segmentation_labels == 6] = 1
            # combine Lateral Tibial Cartilage and Medial Tibial Cartilage
            tibial_cartilage = lateral_tibial_cartilage + medial_tibial_cartilage
            # combine Lateral Meniscus and Medial Meniscus
            medial_meniscus = lateral_meniscus + medial_meniscus

            if self.consecutive_slices > 1:
                segmentation_labels_dim = 1
            else:
                segmentation_labels_dim = 0

            # stack the labels in the last dimension
            segmentation_labels = np.stack(
                [patellar_cartilage, femoral_cartilage, tibial_cartilage, medial_meniscus],
                axis=segmentation_labels_dim,
            )

            # TODO: This is hardcoded on the SKM-TEA side, how to generalize this?
            # We need to crop the segmentation labels in the frequency domain to reduce the FOV.
            segmentation_labels = np.fft.fftshift(np.fft.fft2(segmentation_labels))
            segmentation_labels = segmentation_labels[:, 48:-48, 40:-40]
            segmentation_labels = np.fft.ifft2(np.fft.ifftshift(segmentation_labels)).real
            segmentation_labels = np.where(segmentation_labels > 0.5, 1.0, 0.0)  # Make sure the labels are binary.

            imspace = np.empty([])

            initial_prediction = np.empty([])
            attrs = dict(hf.attrs)

            # get noise level for current slice, if metadata["noise_levels"] is not empty
            if "noise_levels" in metadata and len(metadata["noise_levels"]) > 0:
                metadata["noise"] = metadata["noise_levels"][dataslice]
            else:
                metadata["noise"] = 1.0

            attrs.update(metadata)

        # TODO: check this
        if not is_none(dataset_format) and dataset_format == "skm-tea-echo1+echo2":
            if self.consecutive_slices > 1:
                segmentation_labels = np.transpose(segmentation_labels, (0, 3, 1, 2))
                kspace = np.transpose(kspace, (3, 0, 4, 1, 2))
                sensitivity_map = np.transpose(sensitivity_map, (4, 0, 3, 1, 2))
            else:
                segmentation_labels = np.transpose(segmentation_labels, (2, 0, 1))
                kspace = np.transpose(kspace, (2, 3, 0, 1))
                sensitivity_map = np.transpose(sensitivity_map, (3, 2, 0, 1))
        elif self.consecutive_slices > 1 and not is_none(dataset_format) and dataset_format != "skm-tea-echo1+echo2":
            segmentation_labels = np.transpose(segmentation_labels, (0, 3, 1, 2))
            kspace = np.transpose(kspace, (0, 3, 1, 2))
            sensitivity_map = np.transpose(sensitivity_map.squeeze(), (0, 3, 1, 2))
        else:
            segmentation_labels = np.transpose(segmentation_labels, (2, 0, 1))
            kspace = np.transpose(kspace, (2, 0, 1))
            sensitivity_map = np.transpose(sensitivity_map.squeeze(), (2, 0, 1))

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
