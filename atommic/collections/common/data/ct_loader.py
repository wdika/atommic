# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk  # type: ignore
import yaml  # type: ignore
from torch.utils.data import Dataset

from atommic.collections.common.parts import utils


class CTDataset(Dataset):
    """A generic class for loading a CT dataset for any task.

    .. note::
        Extends :class:`torch.utils.data.Dataset`.
    """

    def __init__(  # noqa: MC0001
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
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Inits :class:`CTDataset`.

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
        **kwargs
            Additional keyword arguments.
        """
        super().__init__()

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
        self.transform = transform
        self.data_saved_per_slice = data_saved_per_slice

        self.examples = []

        # Check if our dataset is in the cache. If yes, use that metadata, if not, then regenerate the metadata.
        if dataset_cache.get(root) is None or not use_dataset_cache:
            if str(root).endswith(".json"):
                with open(root, "rb") as f:  # pylint: disable=unspecified-encoding
                    examples = json.load(f)
                files = [Path(example) for example in examples]
            else:
                files = list(Path(root).iterdir())

            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
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
        >>> from atommic.collections.common.data.ct_loader import CTDataset
        >>> metadata, num_slices = CTDataset._retrieve_metadata("file.nii.gz")
        >>> metadata
        >>> ({"spacing": (1.0, 1.0, 1.0), "origin": (0.0, 0.0, 0.0),
        >>> "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), "num_slices": 10})
        >>> num_slices
        10
        """
        # Read the file
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(fname))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        # Get the metadata
        spacing = reader.GetSpacing()
        origin = reader.GetOrigin()
        direction = reader.GetDirection()
        num_slices = reader.GetSize()[2]

        metadata = {
            "spacing": spacing,
            "origin": origin,
            "direction": direction,
            "num_slices": num_slices,
        }

        return metadata, num_slices

    def get_consecutive_slices(self, labels: np.array, dataslice: int) -> np.ndarray:
        """Get consecutive slices from a given data dictionary.

        Parameters
        ----------
        labels : np.array
            The labels array.
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
        >>> labels = {np.random.rand(10, 5, 512, 512)}
        >>> from atommic.collections.common.data.ct_loader import CTDataset
        >>> CTDataset.get_consecutive_slices(labels, 1).shape
        (1, 512, 512)
        >>> CTDataset.get_consecutive_slices(labels, 5).shape
        (5, 512, 512)
        """
        x = labels

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
        """Length of :class:`CTDataset`."""
        return len(self.examples)

    def __getitem__(self, i: int):
        """Get item from :class:`CTDataset`."""
        raise NotImplementedError
