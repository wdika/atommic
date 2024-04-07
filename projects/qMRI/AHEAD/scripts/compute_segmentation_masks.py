# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from tqdm import tqdm


def compute_masks(target_image: np.ndarray, fname: str) -> np.ndarray:
    """
    Compute the brain and head masks.

    Parameters
    ----------
    target_image : np.ndarray
        The target image.
    fname : str
        The filename of the target image.

    Returns
    -------
    brain_mask : np.ndarray
        The brain mask.
    head_mask : np.ndarray
        The head mask.
    """
    # compute head and brain mask
    brain_masks = []
    for _slice_idx_ in tqdm(range(target_image.shape[0])):
        # calculate otsu's threshold
        threshold = threshold_otsu(np.abs(target_image[_slice_idx_]))
        # get the connected components and apply threshold
        target_image_cc = measure.label(np.abs(target_image[_slice_idx_]) > threshold * 2) + measure.label(
            np.abs(target_image[_slice_idx_]) > threshold
        )
        # Get mask
        skull_mask = np.where(target_image_cc != 0, 1, 0)
        # get the convex hull
        brain_mask = convex_hull_image(skull_mask) * (1 - skull_mask)
        # perform binary erosion to remove skull
        if (
            'mp2rageme_003_axial.h5' in str(fname)
            or 'mp2rageme_007_axial.h5' in str(fname)
            or 'mp2rageme_008_axial.h5' in str(fname)
            or 'mp2rageme_009_axial.h5' in str(fname)
        ):
            brain_mask = binary_erosion(brain_mask, iterations=1)
        elif 'mp2rageme_010_axial.h5' in str(fname):
            brain_mask = binary_erosion(brain_mask, iterations=3)
        else:
            brain_mask = binary_erosion(brain_mask, iterations=4)
        # get the convex hull of the brain mask
        brain_mask = convex_hull_image(brain_mask)
        # threshold the brain mask
        brain_mask = np.where(np.abs(target_image[_slice_idx_]) * brain_mask > threshold / 2, 1, 0)
        # perform binary erosion to remove skull
        brain_mask = binary_erosion(brain_mask, iterations=4)
        # perform binary dilation to get the brain mask
        brain_mask = binary_dilation(brain_mask, iterations=4)
        # fill holes in the brain mask
        brain_mask = binary_fill_holes(brain_mask)
        # get the convex hull of the brain mask
        brain_mask = convex_hull_image(brain_mask)
        brain_masks.append(brain_mask)
    brain_mask = np.stack(brain_masks, axis=0)
    return brain_mask.astype(np.float32)


def main(args):
    output_path = Path(args.output_path)
    if not os.path.exists(output_path):
        output_path.mkdir(parents=True, exist_ok=True)
    # get all files
    files = list(Path(args.data_path).iterdir())
    # iterate over all subjects
    for fname in tqdm(files):
        print(fname)
        # load the target
        target = h5py.File(fname, "r")["target"][()]
        # masks are the same for all echoes
        anatomy_mask = compute_masks(target[:, 0], str(fname.name))
        # save the masks
        with h5py.File(output_path / fname.name, "w") as f:
            f.create_dataset("anatomy_mask", data=anatomy_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, default="data/ahead_data")
    parser.add_argument("output_path", type=Path, default="data/ahead_data_preprocessed")
    args = parser.parse_args()
    main(args)
