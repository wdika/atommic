# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import yaml  # type: ignore
from defusedxml.ElementTree import fromstring
from torch.utils.data import Dataset

from atommic.collections.common.parts import utils


def et_query(root: str, qlist: Sequence[str], namespace: str = "http://www.ismrm.org/ISMRMRD") -> str:
    """Query an XML element for a list of attributes.

    Parameters
    ----------
    root : str
        The root element of the XML tree.
    qlist : list
        A list of strings, each of which is an attribute name.
    namespace : str, optional
        The namespace of the XML tree.

    Returns
    -------
    str
        A string containing the value of the last attribute in the list.
    """
    s = "."
    prefix = "ismrmrd_namespace"
    ns = {prefix: namespace}
    for el in qlist:
        s += f"//{prefix}:{el}"
    value = root.find(s, ns)  # type: ignore
    if value is None:
        return "0"
    return str(value.text)  # type: ignore


class MRIDataset(Dataset):
    """A generic class for loading an MRI dataset for any task.

    .. note::
        Extends :class:`torch.utils.data.Dataset`.
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
        """Inits :class:`MRIDataset`.

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
        if not utils.is_none(sample_rate) and not utils.is_none(volume_sample_rate):
            raise ValueError(
                f"Both sample_rate {sample_rate} and volume_sample_rate {volume_sample_rate} are set. "
                "Please set only one of them."
            )

        if sample_rate is None or sample_rate == "None":
            sample_rate = 1.0

        if volume_sample_rate is None or volume_sample_rate == "None":
            volume_sample_rate = 1.0

        self.dataset_cache_file = (
            None if utils.is_none(dataset_cache_file) else Path(dataset_cache_file)  # type: ignore
        )

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

                self.examples += [(fname, slice_ind, metadata) for slice_ind in range(num_slices)]

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

        if num_cols and not utils.is_none(num_cols):
            self.examples = [ex for ex in self.examples if ex[2]["encoding_size"][1] in num_cols]

        self.indices_to_log = np.random.choice(
            [example[1] for example in self.examples],
            int(log_images_rate * len(self.examples)),  # type: ignore
            replace=False,
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

    def __getitem__(self, i: int):
        """Get item from :class:`MRIDataset`."""
        raise NotImplementedError
