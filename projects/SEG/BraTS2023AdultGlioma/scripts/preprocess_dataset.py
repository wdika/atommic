# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def normalizer(data):
    """Normalize the data to zero mean and unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    if np.isscalar(mean):
        if mean == 0.0:
            mean = 1.0
    elif isinstance(mean, np.ndarray):
        mean[std == 0.0] = 1.0
    return (data - mean) / std


def crop_to_brain(data):
    """Crop the data to the brain region."""
    # crop from left to right until brain is found
    min_x = 0
    for i in range(data.shape[0]):
        if np.sum(data[i, :, :]) > 0:
            min_x = i
            break

    # crop from right to left until brain is found
    max_x = data.shape[0] - 1
    for i in range(data.shape[0] - 1, -1, -1):
        if np.sum(data[i, :, :]) > 0:
            max_x = i
            break

    # crop from top to bottom until brain is found
    min_y = 0
    for i in range(data.shape[1]):
        if np.sum(data[:, i, :]) > 0:
            min_y = i
            break

    # crop from bottom to top until brain is found
    max_y = data.shape[1] - 1
    for i in range(data.shape[1] - 1, -1, -1):
        if np.sum(data[:, i, :]) > 0:
            max_y = i
            break

    # add 15% margin
    margin_x = int((max_x - min_x) * 0.15)
    margin_y = int((max_y - min_y) * 0.15)

    min_x = max(0, min_x - margin_x)
    max_x = min(data.shape[0], max_x + margin_x)
    min_y = max(0, min_y - margin_y)
    max_y = min(data.shape[1], max_y + margin_y)

    return min_x, max_x, min_y, max_y


def main(args):
    train_path = Path(args.data_path) / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    output_train_data_path = Path(args.output_path) / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    if not os.path.exists(output_train_data_path):
        output_train_data_path.mkdir(parents=True, exist_ok=True)
    output_train_segmentations_path = (
        Path(args.output_path) / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingSegmentations"
    )
    if not os.path.exists(output_train_segmentations_path):
        output_train_segmentations_path.mkdir(parents=True, exist_ok=True)

    # iterate over all subjects
    train_subjects = list(train_path.glob("*/"))

    # each subject dir contains five files, each starting with the subject name and ending with "seg.nii.gz",
    # "t1c.nii.gz", "t1n.nii.gz", "t2f.nii.gz", or "t2w.nii.gz"
    for subj in tqdm(train_subjects):
        # get all files inside the subject dir if not seg
        t1c = nib.load(subj / f"{subj.name}-t1c.nii.gz")
        t1n = nib.load(subj / f"{subj.name}-t1n.nii.gz")
        t2f = nib.load(subj / f"{subj.name}-t2f.nii.gz")
        t2w = nib.load(subj / f"{subj.name}-t2w.nii.gz")

        # get data affine
        affine = t1c.affine

        # get data
        t1c_data = t1c.get_fdata().astype(np.float32)
        t1n_data = t1n.get_fdata().astype(np.float32)
        t2f_data = t2f.get_fdata().astype(np.float32)
        t2w_data = t2w.get_fdata().astype(np.float32)

        # get seg
        seg = nib.load(subj / f"{subj.name}-seg.nii.gz")

        # get seg affine
        seg_affine = seg.affine

        # get seg data
        seg_data = seg.get_fdata().astype(np.float32)

        # crop to brain
        t1c_min_x, t1c_max_x, t1c_min_y, t1c_max_y = crop_to_brain(t1c_data)
        t1n_min_x, t1n_max_x, t1n_min_y, t1n_max_y = crop_to_brain(t1n_data)
        t2f_min_x, t2f_max_x, t2f_min_y, t2f_max_y = crop_to_brain(t2f_data)
        t2w_min_x, t2w_max_x, t2w_min_y, t2w_max_y = crop_to_brain(t2w_data)

        # get max of min slices and min of max slices to ensure that all modalities have the same number of slices
        # containing the tumor
        min_x = max(t1c_min_x, t1n_min_x, t2f_min_x, t2w_min_x)
        max_x = min(t1c_max_x, t1n_max_x, t2f_max_x, t2w_max_x)
        min_y = max(t1c_min_y, t1n_min_y, t2f_min_y, t2w_min_y)
        max_y = min(t1c_max_y, t1n_max_y, t2f_max_y, t2w_max_y)

        # crop the data and seg
        t1c_data = t1c_data[min_x : max_x + 1, min_y : max_y + 1, :]
        t1n_data = t1n_data[min_x : max_x + 1, min_y : max_y + 1, :]
        t2f_data = t2f_data[min_x : max_x + 1, min_y : max_y + 1, :]
        t2w_data = t2w_data[min_x : max_x + 1, min_y : max_y + 1, :]
        seg_data = seg_data[min_x : max_x + 1, min_y : max_y + 1, :]

        # normalize again
        t1c_data = normalizer(t1c_data)
        t1n_data = normalizer(t1n_data)
        t2f_data = normalizer(t2f_data)
        t2w_data = normalizer(t2w_data)

        # update the header
        hdr = t1c.header.copy()
        hdr["dim"][1] = 4
        hdr["dim"][2] = t1c_data.shape[0]
        hdr["dim"][3] = t1c_data.shape[1]
        hdr["dim"][4] = t1c_data.shape[2]

        # save the stacked modalities
        all_modalities_nii = nib.Nifti1Image(
            np.stack([t1c_data, t1n_data, t2f_data, t2w_data], axis=0), affine=affine, header=hdr
        )
        nib.save(all_modalities_nii, output_train_data_path / f"{subj.name}.nii.gz")

        # update the seg header
        seg_hdr = seg.header.copy()
        seg_hdr["dim"][1] = 1
        seg_hdr["dim"][2] = seg_data.shape[0]
        seg_hdr["dim"][3] = seg_data.shape[1]
        seg_hdr["dim"][4] = seg_data.shape[2]

        # save the seg file to the output dir
        seg_nii = nib.Nifti1Image(seg_data, affine=seg_affine, header=seg_hdr)
        nib.save(seg_nii, output_train_segmentations_path / f"{subj.name}-seg.nii.gz")

    val_path = Path(args.data_path) / "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    output_val_data_path = Path(args.output_path) / "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    if not os.path.exists(output_val_data_path):
        output_val_data_path.mkdir(parents=True, exist_ok=True)

    # iterate over all subjects
    val_subjects = list(val_path.glob("*/"))

    # each subject dir contains five files, each starting with the subject name and ending with "t1c.nii.gz",
    # "t1n.nii.gz", "t2f.nii.gz", or "t2w.nii.gz". Validation data don't include seg.nii.gz files.
    for subj in tqdm(val_subjects):
        # get all files inside the subject dir if not seg
        t1c = nib.load(subj / f"{subj.name}-t1c.nii.gz")
        t1n = nib.load(subj / f"{subj.name}-t1n.nii.gz")
        t2f = nib.load(subj / f"{subj.name}-t2f.nii.gz")
        t2w = nib.load(subj / f"{subj.name}-t2w.nii.gz")

        # get affine
        affine = t1c.affine

        t1c_data = t1c.get_fdata().astype(np.float32)
        t1n_data = t1n.get_fdata().astype(np.float32)
        t2f_data = t2f.get_fdata().astype(np.float32)
        t2w_data = t2w.get_fdata().astype(np.float32)

        t1c_data = normalizer(t1c_data)
        t1n_data = normalizer(t1n_data)
        t2f_data = normalizer(t2f_data)
        t2w_data = normalizer(t2w_data)

        # save the stacked modalities
        all_modalities_nii = nib.Nifti1Image(np.stack([t1c_data, t1n_data, t2f_data, t2w_data], axis=0), affine=affine)

        nib.save(all_modalities_nii, output_val_data_path / f"{subj.name}.nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    main(args)
