# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def resample_flair(flair, target, resample_method=sitk.sitkBSpline):
    """
    Resample the image to the same space as the target image.

    Parameters
    ----------
    flair : Path
        The path to the flair image.
    target : Path
        The path to the target image.
    resample_method : sitk.sitkLinear
        The resample method.
    """
    flair = sitk.ReadImage(flair)
    target = sitk.ReadImage(target)
    if not flair.GetSize() == target.GetSize():
        # set target size
        target_origin = target.GetOrigin()
        target_direction = target.GetDirection()
        target_spacing = target.GetSpacing()
        target_size = target.GetSize()

        # initialize resampler
        resampler_image = sitk.ResampleImageFilter()
        # set the parameters of image
        resampler_image.SetReferenceImage(flair)  # set resampled image meta data same to origin data
        resampler_image.SetOutputOrigin(target_origin)
        resampler_image.SetOutputDirection(target_direction)  # set target image space
        resampler_image.SetOutputSpacing(target_spacing)  # set target image space
        resampler_image.SetSize(target_size)  # set target image size
        if resample_method == sitk.sitkNearestNeighbor:
            resampler_image.SetOutputPixelType(sitk.sitkUInt8)
        else:
            resampler_image.SetOutputPixelType(sitk.sitkFloat32)
        resampler_image.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler_image.SetInterpolator(resample_method)

        # launch the resampler
        resampled_image = resampler_image.Execute(flair)
        # convert to numpy array
        resampled_image = sitk.GetArrayFromImage(resampled_image)
        return resampled_image
    return flair


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


def main(args):
    output_data_path = Path(args.output_path) / "data"
    if not os.path.exists(output_data_path):
        output_data_path.mkdir(parents=True, exist_ok=True)
    output_segmentations_path = Path(args.output_path) / "segmentations"
    if not os.path.exists(output_segmentations_path):
        output_segmentations_path.mkdir(parents=True, exist_ok=True)

    # get all files
    derivatives = [
        j
        for f in list((Path(args.data_path) / "derivatives").iterdir())
        for i in list(f.iterdir())
        for j in list(i.iterdir())
        if j.name.endswith(".nii.gz")
    ]
    rawdata = [
        j
        for f in list((Path(args.data_path) / "rawdata").iterdir())
        for i in list(f.iterdir())
        for j in list(i.iterdir())
        if j.name.endswith(".nii.gz")
    ]

    # create a dictionary with all the derivatives
    derivatives_subjects_files = {}
    for file in derivatives:
        fname = file.name.replace("_ses-0001_msk.nii.gz", "")
        derivatives_subjects_files[fname] = file

    # create a dictionary with all the rawdata
    rawdata_adc_files = {}
    rawdata_dwi_files = {}
    rawdata_flair_files = {}
    for file in rawdata:
        if "adc" in file.name:
            fname = file.name.replace("_ses-0001_adc.nii.gz", "")
            rawdata_adc_files[fname] = file
        if "dwi" in file.name:
            fname = file.name.replace("_ses-0001_dwi.nii.gz", "")
            rawdata_dwi_files[fname] = file
        if "flair" in file.name:
            fname = file.name.replace("_ses-0001_flair.nii.gz", "")
            rawdata_flair_files[fname] = file

    # iterate over all the subjects and derivatives
    subjects = {}
    for fname, files in derivatives_subjects_files.items():
        subjects[fname] = {
            "mask": derivatives_subjects_files[fname],
            "adc": rawdata_adc_files[fname],
            "dwi": rawdata_dwi_files[fname],
            "flair": rawdata_flair_files[fname],
        }

    # read the data
    for fname, files in tqdm(subjects.items()):
        # Segmentation
        seg_data = nib.load(files["mask"]).get_fdata()

        # find which slices contain the lesion
        lesion_slices = [i for i in range(seg_data.shape[2]) if np.sum(seg_data[:, :, i]) > 0]

        if len(lesion_slices) == 0:
            continue

        # keep only the slices that contain the lesion
        seg_data = np.stack([seg_data[:, :, i] for i in range(seg_data.shape[2]) if i in lesion_slices], axis=-1)

        seg_data = np.transpose(seg_data, (1, 0, 2))

        # get the seg affine
        seg = nib.load(files["mask"])
        seg_affine = seg.affine

        # update the seg header
        seg_hdr = seg.header.copy()
        seg_hdr["dim"][0] = 1
        seg_hdr["dim"][1] = seg_data.shape[0]
        seg_hdr["dim"][2] = seg_data.shape[1]
        seg_hdr["dim"][3] = seg_data.shape[2]

        # save the seg file to the output dir
        seg_nii = nib.Nifti1Image(seg_data, affine=seg_affine, header=seg_hdr)
        nib.save(seg_nii, output_segmentations_path / f"{fname}-seg.nii.gz")

        # ADC
        adc_nii = nib.load(files["adc"])
        adc_data = adc_nii.get_fdata().astype(np.float32)

        # DWI
        dwi_nii = nib.load(files["dwi"])
        dwi_affine = dwi_nii.affine
        dwi_header = dwi_nii.header
        dwi_data = dwi_nii.get_fdata().astype(np.float32)

        # FLAIR
        flair_data = np.transpose(resample_flair(files["flair"], files["dwi"]), (2, 1, 0)).astype(np.float32)

        adc_data = np.clip(adc_data, 0.0, adc_data.max())
        dwi_data = np.clip(dwi_data, 0.0, dwi_data.max())
        flair_data = np.clip(flair_data, 0.0, flair_data.max())

        # keep only the slices that contain the lesion
        adc_data = np.stack([adc_data[:, :, i] for i in range(adc_data.shape[2]) if i in lesion_slices], axis=-1)
        dwi_data = np.stack([dwi_data[:, :, i] for i in range(dwi_data.shape[2]) if i in lesion_slices], axis=-1)
        flair_data = np.stack([flair_data[:, :, i] for i in range(flair_data.shape[2]) if i in lesion_slices], axis=-1)

        # normalize
        # adc_data = normalizer(adc_data)
        # dwi_data = normalizer(dwi_data)
        # flair_data = normalizer(flair_data)

        # get correct orientation
        adc_data = np.transpose(adc_data, (1, 0, 2))
        dwi_data = np.transpose(dwi_data, (1, 0, 2))
        flair_data = np.transpose(flair_data, (1, 0, 2))

        # get the dwi header
        hdr = dwi_header.copy()
        hdr["dim"][0] = 3
        hdr["dim"][1] = dwi_data.shape[0]
        hdr["dim"][2] = dwi_data.shape[1]
        hdr["dim"][3] = dwi_data.shape[2]

        # save the stacked modalities
        all_modalities_nii = nib.Nifti1Image(
            np.stack([adc_data, dwi_data, flair_data], axis=0), affine=dwi_affine, header=hdr
        )
        nib.save(all_modalities_nii, output_data_path / f"{fname}.nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    main(args)
