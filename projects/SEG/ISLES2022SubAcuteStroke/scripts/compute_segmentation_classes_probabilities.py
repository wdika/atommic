# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def main(args):
    # get all files
    derivatives = [
        j
        for f in list((Path(args.data_path) / "derivatives").iterdir())
        for i in list(f.iterdir())
        for j in list(i.iterdir())
        if j.name.endswith(".nii.gz")
    ]

    # create a dictionary with all the derivatives
    derivatives_subjects_files = {}
    for file in derivatives:
        fname = file.name.replace("_ses-0001_msk.nii.gz", "")
        derivatives_subjects_files[fname] = file

    # iterate over all the subjects and derivatives
    subjects = {}
    for fname, files in derivatives_subjects_files.items():
        subjects[fname] = {"mask": derivatives_subjects_files[fname]}

    bgs = []
    lesions = []
    total_slices = 0

    # read the data
    for fname, files in tqdm(subjects.items()):
        # get segmentation
        segmentation_labels = nib.load(files["mask"]).get_fdata().astype(np.float32)

        # Lesion (label 1)
        lesion = np.zeros_like(segmentation_labels)
        lesion[segmentation_labels == 1] = 1

        # find how many slices contain each class
        bg_slices = np.sum(
            [1 for i in range(segmentation_labels.shape[2]) if np.sum(segmentation_labels[:, :, i]) == 0]
        )
        lesion_slices = np.sum([1 for i in range(lesion.shape[2]) if np.sum(lesion[:, :, i]) > 0])

        bgs.append(bg_slices)
        lesions.append(lesion_slices)

        total_slices += segmentation_labels.shape[2]

    # compute probabilities for each class
    bg_prob = np.sum(bgs, axis=0) / total_slices
    lesion_prob = np.sum(lesions, axis=0) / total_slices

    # sum and compute 100% probability
    total_prob = bg_prob + lesion_prob
    bg_prob /= total_prob
    lesion_prob /= total_prob

    # round to 3 decimals
    bg_prob = np.round(bg_prob, 3)
    lesion_prob = np.round(lesion_prob, 3)

    print(f"Probabilities {bg_prob + lesion_prob}. " f"Background: {bg_prob}, " f"Lesions: {lesion_prob}, ")

    # create output dir
    output_path = Path(args.output_path)
    if not os.path.exists(output_path):
        output_path.mkdir(parents=True, exist_ok=True)

    # save probabilities as json
    with open(output_path / "probabilities.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "bg_prob": bg_prob.tolist(),
                "lesion_prob": lesion_prob.tolist(),
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    main(args)
