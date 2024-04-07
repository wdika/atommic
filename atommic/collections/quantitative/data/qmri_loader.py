# coding=utf-8
__author__ = "Dimitris Karkalousos"

import os
import re
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import h5py
import numpy as np

from atommic.collections.common.data.mri_loader import MRIDataset
from atommic.collections.common.parts.utils import is_none


class qMRIDataset(MRIDataset):
    """A dataset class for quantitative MRI.

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
        sequence: str = None,
        segmentation_mask_root: Union[str, Path, os.PathLike] = None,
        kspace_scaling_factor: float = 1.0,
        **kwargs,
    ):
        """Inits :class:`qMRIDataset`.

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
            should take ``kspace``, ``coil sensitivity maps``, ``quantitative maps``, ``mask``, ``initial prediction``,
            ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test
            data. Default is ``None``.
        sequence : str, optional
            Sequence of the dataset. For example, ``MEGRE`` or ``FUTURE_SEQUENCES``.
        segmentation_mask_root : Union[str, Path, os.PathLike], optional
            Path to stored segmentation masks, if stored separately.
        kspace_scaling_factor : float, optional
            A float that scales the kspace. Default is ``1.0``.
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
        if sequence not in ("MEGRE", "FUTURE_SEQUENCES"):
            warnings.warn(
                'Sequence should be either "MEGRE" or "FUTURE_SEQUENCES". '
                f'Found {sequence}. If you are using this dataset for reconstruction, ignore this warning.'
                'If you are using this dataset for quantitative mapping, please use the correct sequence.'
            )
        self.sequence = sequence
        self.segmentation_mask_root = segmentation_mask_root
        self.kspace_scaling_factor = kspace_scaling_factor

    def __getitem__(self, i: int):  # noqa: MC0001
        """Get item from :class:`qMRIDataset`."""
        raise NotImplementedError


class AHEADqMRIDataset(qMRIDataset):
    """Supports the AHEAD dataset for quantitative MRI.

    .. note::
        Extends :class:`atommic.collections.quantitative.data.qMRIDataset`.
    """

    def __getitem__(self, i: int):  # noqa: MC0001
        """Get item from :class:`AHEADqMRIDataset`."""
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

            kspace = kspace / self.kspace_scaling_factor

            if "sensitivity_map" in hf:
                sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(np.complex64)
            elif not is_none(self.coil_sensitivity_maps_root):
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

                with h5py.File(
                    Path(coil_sensitivity_maps_root) / Path(split_dir[-2]) / fname.name, "r"  # type: ignore
                ) as sf:
                    if "sensitivity_map" in sf or "sensitivity_map" in next(iter(sf.keys())):
                        sensitivity_map = (
                            self.get_consecutive_slices(sf, "sensitivity_map", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
            else:
                sensitivity_map = np.array([])

            if "mask" in hf:
                mask = np.asarray(self.get_consecutive_slices(hf, "mask", dataslice))
                if mask.ndim == 3:
                    mask = mask[dataslice]
            elif not is_none(self.mask_root):
                with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:  # type: ignore
                    mask = np.asarray(self.get_consecutive_slices(mf, "mask", dataslice))
            else:
                mask = np.empty([])

            if "anatomy_mask" in hf:
                anatomy_mask = np.asarray(self.get_consecutive_slices(hf, "anatomy_mask", dataslice))
                if anatomy_mask.ndim == 3:
                    anatomy_mask = anatomy_mask[dataslice]
            elif not is_none(self.segmentation_mask_root):
                with h5py.File(Path(self.segmentation_mask_root) / fname.name, "r") as mf:  # type: ignore
                    anatomy_mask = np.asarray(self.get_consecutive_slices(mf, "anatomy_mask", dataslice))
            else:
                anatomy_mask = np.empty([])

            mask = [mask, anatomy_mask]

            prediction = np.empty([])
            if not is_none(self.initial_predictions_root):
                with h5py.File(Path(self.initial_predictions_root) / fname.name, "r") as ipf:  # type: ignore
                    rkey = "reconstruction" if "reconstruction" in ipf else "initial_prediction"
                    prediction = self.get_consecutive_slices(ipf, rkey, dataslice).squeeze().astype(np.complex64)
            elif "reconstruction" in hf or "initial_prediction" in hf:
                rkey = "reconstruction" if "reconstruction" in hf else "initial_prediction"
                prediction = self.get_consecutive_slices(hf, rkey, dataslice).squeeze().astype(np.complex64)

            # find key containing "reconstruction_"
            rkey = re.findall(r"reconstruction_(.*)", str(hf.keys()))  # type: ignore
            self.recons_key = "reconstruction_" + rkey[0] if rkey else "target"
            if "reconstruction_rss" in self.recons_key:
                self.recons_key = "reconstruction_rss"
            elif "reconstruction_sense" in hf:
                self.recons_key = "reconstruction_sense"

            if self.complex_target:
                target = None
            else:
                # find key containing "reconstruction_"
                rkey = re.findall(r"reconstruction_(.*)", str(hf.keys()))  # type: ignore
                self.recons_key = "reconstruction_" + rkey[0] if rkey else "target"
                if "reconstruction_rss" in self.recons_key:
                    self.recons_key = "reconstruction_rss"
                elif "reconstruction_sense" in hf:
                    self.recons_key = "reconstruction_sense"
                target = self.get_consecutive_slices(hf, self.recons_key, dataslice) if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            # get noise level for current slice, if metadata["noise_levels"] is not empty
            if "noise_levels" in metadata and len(metadata["noise_levels"]) > 0:
                metadata["noise"] = metadata["noise_levels"][dataslice]
            else:
                metadata["noise"] = 1.0

            attrs.update(metadata)

        if self.data_saved_per_slice:
            # arbitrary slice number for logging purposes
            dataslice = str(fname.name)  # type: ignore
            if "h5" in dataslice:  # type: ignore
                dataslice = dataslice.split(".h5")[0]  # type: ignore
            dataslice = int(dataslice.split("_")[-1])  # type: ignore

        attrs["log_image"] = bool(dataslice in self.indices_to_log) if not self.data_saved_per_slice else True

        if not is_none(self.sequence):
            return (
                (
                    kspace,
                    sensitivity_map,
                    mask,
                    prediction,
                    target,
                    attrs,
                    fname.name,
                    dataslice,
                )
                if self.transform is None
                else self.transform(
                    kspace,
                    sensitivity_map,
                    mask,
                    prediction,
                    target,
                    attrs,
                    fname.name,
                    dataslice,
                )
            )
        return (
            (
                kspace,
                sensitivity_map,
                np.empty([]),
                prediction,
                target,
                attrs,
                fname.name,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                kspace,
                sensitivity_map,
                np.empty([]),
                prediction,
                target,
                attrs,
                fname.name,
                dataslice,
            )
        )
