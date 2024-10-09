# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import json
import logging
import os
import random
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import yaml  # type: ignore
from defusedxml.ElementTree import fromstring
from torch.utils.data import Dataset

from atommic.collections.common.data.mri_loader import MRIDataset, et_query
from atommic.collections.common.parts.utils import is_none


class ReconstructionMRIDataset(MRIDataset):
    """A dataset class for accelerated MRI reconstruction.

    Examples
    --------
    >>> from atommic.collections.reconstruction.data.mri_reconstruction_loader import ReconstructionMRIDataset
    >>> dataset = ReconstructionMRIDataset(root='data/train', sample_rate=0.1)
    >>> print(len(dataset))
    100
    >>> kspace, coil_sensitivities, mask, initial_prediction, target, attrs, filename, slice_num = dataset[0]
    >>> print(kspace.shape)
    np.array([30, 640, 368])

    .. note::
        Extends :class:`atommic.collections.common.data.mri_loader.MRIDataset`.
    """

    def __getitem__(self, i: int):  # noqa: MC0001
        """Get item from :class:`ReconstructionMRIDataset`."""
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            min_val = hf["min"][()] if "min" in hf else None
            max_val = hf["max"][()] if "max" in hf else None
            mean_val = hf["mean"][()] if "mean" in hf else None
            std_val = hf["std"][()] if "std" in hf else None

            kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

            sensitivity_map = np.array([])
            if "sensitivity_map" in hf:
                sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(np.complex64)
            elif "maps" in hf:
                sensitivity_map = self.get_consecutive_slices(hf, "maps", dataslice).astype(np.complex64)
            elif self.coil_sensitivity_maps_root is not None and self.coil_sensitivity_maps_root != "None":
                coil_sensitivity_maps_root = self.coil_sensitivity_maps_root
                split_dir = str(fname).split("/")
                for j in range(len(split_dir)):
                    coil_sensitivity_maps_root = Path(f"{self.coil_sensitivity_maps_root}/{split_dir[-j]}/")
                    if os.path.exists(coil_sensitivity_maps_root / Path(split_dir[-2]) / fname.name):
                        break
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

            prediction = np.empty([])
            if not is_none(self.initial_predictions_root):
                with h5py.File(Path(self.initial_predictions_root) / fname.name, "r") as ipf:  # type: ignore
                    if "reconstruction" in hf:
                        prediction = (
                            self.get_consecutive_slices(ipf, "reconstruction", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
                    elif "initial_prediction" in hf:
                        prediction = (
                            self.get_consecutive_slices(ipf, "initial_prediction", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
            else:
                if "reconstruction" in hf:
                    prediction = (
                        self.get_consecutive_slices(hf, "reconstruction", dataslice).squeeze().astype(np.complex64)
                    )
                elif "initial_prediction" in hf:
                    prediction = (
                        self.get_consecutive_slices(hf, "initial_prediction", dataslice).squeeze().astype(np.complex64)
                    )

            if self.complex_target:
                target = None
            else:
                # find key containing "reconstruction_"
                rkey = re.findall(r"reconstruction_(.*)", str(hf.keys()))
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

        if min_val is not None:
            attrs["min"] = min_val
        if max_val is not None:
            attrs["max"] = max_val
        if mean_val is not None:
            attrs["mean"] = mean_val
        if std_val is not None:
            attrs["std"] = std_val

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


class CC359ReconstructionMRIDataset(Dataset):
    """Supports the CC359 dataset for accelerated MRI reconstruction.

    .. note::
        Similar to :class:`atommic.collections.common.data.mri_loader.MRIDataset`. It does not extend it because we
        need to override the ``__init__`` and ``__getitem__`` methods.
    """

    def __init__(  # noqa: MC0001
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
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Inits :class:`CC359ReconstructionMRIDataset`.

        Parameters
        ----------
        root : Union[str, Path, os.PathLike]
            Path to the dataset.
        coil_sensitivity_maps_root : Union[str, Path, os.PathLike], optional
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
            A float between 0 and 1. This controls what fraction of the slices should be logged as images. Default is
            ``1.0``.
        transform : Optional[Callable], optional
            A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function
            should take ``kspace``, ``coil sensitivity maps``, ``quantitative maps``, ``mask``, ``initial prediction``,
            ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test
            data. Default is ``None``.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__()
        self.coil_sensitivity_maps_root = coil_sensitivity_maps_root
        self.mask_root = mask_root

        if str(noise_root).endswith(".json"):
            with open(noise_root, "r") as f:  # type: ignore  # pylint: disable=unspecified-encoding
                noise_root = [json.loads(line) for line in f.readlines()]  # type: ignore
        else:
            noise_root = None

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

        self.recons_key = "reconstruction"
        self.examples = []

        # Check if our dataset is in the cache. If yes, use that metadata, if not, then regenerate the metadata.
        if dataset_cache.get(root) is None or not use_dataset_cache:
            if str(root).endswith(".json"):
                with open(root, "r") as f:  # pylint: disable=unspecified-encoding
                    examples = json.load(f)
                files = [Path(example) for example in examples]
            else:
                files = list(Path(root).iterdir())

            if n2r_supervised_rate != 0.0:
                # randomly select a subset of files for N2R supervised loss based on n2r_supervised_rate
                n2r_supervised_files = random.sample(
                    files, int(np.round(n2r_supervised_rate * len(files)))  # type: ignore
                )

            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                metadata["noise_levels"] = (
                    self.__parse_noise__(noise_root, fname) if noise_root is not None else []  # type: ignore
                )
                metadata["n2r_supervised"] = False
                if n2r_supervised_rate != 0.0:
                    #  Use lazy % formatting in logging
                    logging.info("%s files are selected for N2R supervised loss.", n2r_supervised_files)
                    if fname in n2r_supervised_files:
                        metadata["n2r_supervised"] = True

                if not is_none(num_slices) and not is_none(consecutive_slices):
                    num_slices = num_slices - (consecutive_slices - 1)

                # Specific to CC359 dataset, we need to remove the first and last 50 slices
                self.examples += [
                    (fname, slice_ind, metadata) for slice_ind in range(num_slices) if 50 < slice_ind < num_slices - 50
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

        Examples
        --------
        >>> metadata, num_slices = _retrieve_metadata("file.h5")
        >>> metadata
        {'padding_left': 0, 'padding_right': 0, 'encoding_size': 0, 'recon_size': (0, 0)}
        >>> num_slices
        1
        """
        with h5py.File(fname, "r") as hf:
            if "ismrmrd_header" in hf:
                et_root = fromstring(hf["ismrmrd_header"][()])

                enc = ["encoding", "encodedSpace", "matrixSize"]
                enc_size = (
                    int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                    int(et_query(et_root, enc + ["z"])),
                )
                rec = ["encoding", "reconSpace", "matrixSize"]
                recon_size = (
                    int(et_query(et_root, rec + ["x"])),
                    int(et_query(et_root, rec + ["y"])),
                    int(et_query(et_root, rec + ["z"])),
                )

                params = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
                enc_limits_center = int(et_query(et_root, params + ["center"]))
                enc_limits_max = int(et_query(et_root, params + ["maximum"])) + 1

                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max
            else:
                padding_left = 0
                padding_right = 0
                enc_size = (0, 0, 0)
                recon_size = (0, 0, 0)

            if "kspace" in hf:
                shape = hf["kspace"].shape
            elif "reconstruction" in hf:
                shape = hf["reconstruction"].shape
            elif "target" in hf:
                shape = hf["target"].shape
            else:
                raise ValueError(f"{fname} does not contain kspace, reconstruction, or target data.")

        num_slices = 1 if self.data_saved_per_slice else shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
            "num_slices": num_slices,
        }

        return metadata, num_slices

    @staticmethod
    def __parse_noise__(noise: str, fname: Path) -> List[str]:
        """Parse noise type from filename.

        Parameters
        ----------
        noise : str
            json string of noise type.
        fname : Path
            Filename to parse noise type from.

        Returns
        -------
        List[str]
            List of noise values.
        """
        return [noise[i]["noise"] for i in range(len(noise)) if noise[i]["fname"] == fname.name]  # type: ignore

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

    def __getitem__(self, i: int):  # noqa: MC0001
        """Get item from :class:`CC359ReconstructionMRIDataset`."""
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

            kspace = np.transpose(kspace[..., ::2] + 1j * kspace[..., 1::2], (2, 0, 1))

            sensitivity_map = np.array([])
            if "sensitivity_map" in hf:
                sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(np.complex64)
            elif "maps" in hf:
                sensitivity_map = self.get_consecutive_slices(hf, "maps", dataslice).astype(np.complex64)
            elif self.coil_sensitivity_maps_root is not None and self.coil_sensitivity_maps_root != "None":
                coil_sensitivity_maps_root = self.coil_sensitivity_maps_root
                split_dir = str(fname).split("/")
                for j in range(len(split_dir)):
                    coil_sensitivity_maps_root = Path(f"{self.coil_sensitivity_maps_root}/{split_dir[-j]}/")
                    if os.path.exists(coil_sensitivity_maps_root / Path(split_dir[-2]) / fname.name):
                        break
                with h5py.File(Path(coil_sensitivity_maps_root) / Path(split_dir[-2]) / fname.name, "r") as sf:
                    if "sensitivity_map" in sf or "sensitivity_map" in next(iter(sf.keys())):
                        sensitivity_map = (
                            self.get_consecutive_slices(sf, "sensitivity_map", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )

            if self.mask_root is not None and self.mask_root != "None":
                mask = []
                with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                    for key in mf.keys():
                        mask.append(np.asarray(self.get_consecutive_slices(mf, key, dataslice)))
            else:
                mask = None

            prediction = np.empty([])
            if not is_none(self.initial_predictions_root):
                with h5py.File(Path(self.initial_predictions_root) / fname.name, "r") as ipf:  # type: ignore
                    if "reconstruction" in hf:
                        prediction = (
                            self.get_consecutive_slices(ipf, "reconstruction", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
                    elif "initial_prediction" in hf:
                        prediction = (
                            self.get_consecutive_slices(ipf, "initial_prediction", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
            else:
                if "reconstruction" in hf:
                    prediction = (
                        self.get_consecutive_slices(hf, "reconstruction", dataslice).squeeze().astype(np.complex64)
                    )
                elif "initial_prediction" in hf:
                    prediction = (
                        self.get_consecutive_slices(hf, "initial_prediction", dataslice).squeeze().astype(np.complex64)
                    )

            if self.complex_target:
                target = None
            else:
                # find key containing "reconstruction_"
                rkey = re.findall(r"reconstruction_(.*)", str(hf.keys()))
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


class SKMTEAReconstructionMRIDataset(MRIDataset):
    """Supports the SKM-TEA dataset for accelerated MRI reconstruction.

    .. note::
        Extends :class:`atommic.collections.reconstruction.data.mri_reconstruction_loader.ReconstructionMRIDataset`.
    """

    def __getitem__(self, i: int):  # noqa: MC0001
        """Get item from :class:`SKMTEAReconstructionMRIDataset`."""
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
            elif not is_none(dataset_format) and dataset_format == "skm-tea-echo1-echo2":
                kspace = kspace
            else:
                warnings.warn(
                    f"Dataset format {dataset_format} is either not supported or set to None. "
                    "Using by default only the first echo."
                )
                kspace = kspace[:, :, 0, :]

            kspace = kspace[48:-48, 40:-40]

            sensitivity_map = self.get_consecutive_slices(hf, "maps", dataslice).astype(np.complex64)
            sensitivity_map = sensitivity_map[..., 0]

            sensitivity_map = sensitivity_map[48:-48, 40:-40]

            if masking == "custom":
                mask = np.array([])
            else:
                masks = hf["masks"]
                mask = {}
                for key, val in masks.items():
                    mask[key.split("_")[-1].split(".")[0]] = np.asarray(val)

            prediction = np.empty([])
            if not is_none(self.initial_predictions_root):
                if "reconstruction" in hf:
                    with h5py.File(Path(self.initial_predictions_root) / fname.name, "r") as ipf:  # type: ignore
                        prediction = (
                            self.get_consecutive_slices(ipf, "reconstruction", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
                elif "initial_prediction" in hf:
                    with h5py.File(Path(self.initial_predictions_root) / fname.name, "r") as ipf:  # type: ignore
                        prediction = (
                            self.get_consecutive_slices(ipf, "initial_prediction", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
            else:
                if "reconstruction" in hf:
                    prediction = (
                        self.get_consecutive_slices(hf, "reconstruction", dataslice).squeeze().astype(np.complex64)
                    )
                elif "initial_prediction" in hf:
                    prediction = (
                        self.get_consecutive_slices(hf, "initial_prediction", dataslice).squeeze().astype(np.complex64)
                    )

            if self.complex_target:
                target = None
            else:
                # find key containing "reconstruction_"
                self.recons_key = "target"
                target = self.get_consecutive_slices(hf, self.recons_key, dataslice) if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            # get noise level for current slice, if metadata["noise_levels"] is not empty
            if "noise_levels" in metadata and len(metadata["noise_levels"]) > 0:
                metadata["noise"] = metadata["noise_levels"][dataslice]
            else:
                metadata["noise"] = 1.0

            attrs.update(metadata)

        kspace = np.transpose(kspace, (2, 0, 1))
        sensitivity_map = np.transpose(sensitivity_map.squeeze(), (2, 0, 1))

        attrs["log_image"] = bool(dataslice in self.indices_to_log)

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


class StanfordKneesReconstructionMRIDataset(MRIDataset):
    """Supports the Stanford Knees 2019 dataset for accelerated MRI reconstruction.

    .. note::
        Extends :class:`atommic.collections.reconstruction.data.mri_reconstruction_loader.ReconstructionMRIDataset`.
    """

    def __getitem__(self, i: int):
        """Get item from :class:`StanfordKneesReconstructionMRIDataset`."""
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

            attrs = dict(hf.attrs)

        sensitivity_map = np.array([])
        if "sensitivity_map" in hf:
            sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(np.complex64)
        elif "maps" in hf:
            sensitivity_map = self.get_consecutive_slices(hf, "maps", dataslice).astype(np.complex64)
        elif self.coil_sensitivity_maps_root is not None and self.coil_sensitivity_maps_root != "None":
            coil_sensitivity_maps_root = self.coil_sensitivity_maps_root
            split_dir = str(fname).split("/")
            for j in range(len(split_dir)):
                coil_sensitivity_maps_root = Path(f"{self.coil_sensitivity_maps_root}/{split_dir[-j]}/")
                if os.path.exists(coil_sensitivity_maps_root / Path(split_dir[-2]) / fname.name):
                    break
            with h5py.File(Path(coil_sensitivity_maps_root) / Path(split_dir[-2]) / fname.name, "r") as sf:
                if "sensitivity_map" in sf or "sensitivity_map" in next(iter(sf.keys())):
                    sensitivity_map = (
                        self.get_consecutive_slices(sf, "sensitivity_map", dataslice).squeeze().astype(np.complex64)
                    )

        # get noise level for current slice, if metadata["noise_levels"] is not empty
        metadata["noise"] = (
            metadata["noise_levels"][dataslice]
            if "noise_levels" in metadata and len(metadata["noise_levels"]) > 0
            else 1.0
        )
        attrs.update(metadata)
        attrs["log_image"] = bool(dataslice in self.indices_to_log)

        mask = None
        prediction = None
        target = np.array([])

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
