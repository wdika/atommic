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
    # iterate over all subjects
    train_subjects = list(
        (Path(args.data_path) / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingSegmentations').glob("*/")
    )

    bgs = []
    ncrs = []
    eds = []
    ets = []
    wts = []
    total_slices = 0

    # parse all "seg.nii.gz",
    for subj in tqdm(train_subjects):
        # get segmentation
        segmentation_labels = nib.load(subj).get_fdata().astype(np.float32)

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

        # find how many slices contain each class
        bg_slices = np.sum(
            [1 for i in range(segmentation_labels.shape[2]) if np.sum(segmentation_labels[:, :, i]) == 0]
        )
        ncr_slices = np.sum([1 for i in range(ncr.shape[2]) if np.sum(ncr[:, :, i]) > 0])
        ed_slices = np.sum([1 for i in range(ed.shape[2]) if np.sum(ed[:, :, i]) > 0])
        et_slices = np.sum([1 for i in range(et.shape[2]) if np.sum(et[:, :, i]) > 0])
        wt_slices = np.sum([1 for i in range(wt.shape[2]) if np.sum(wt[:, :, i]) > 0])

        bgs.append(bg_slices)
        ncrs.append(ncr_slices)
        eds.append(ed_slices)
        ets.append(et_slices)
        wts.append(wt_slices)

        total_slices += segmentation_labels.shape[2]

    # compute probabilities for each class
    bg_prob = np.sum(bgs, axis=0) / total_slices
    ncr_prob = np.sum(ncrs, axis=0) / total_slices
    ed_prob = np.sum(eds, axis=0) / total_slices
    et_prob = np.sum(ets, axis=0) / total_slices
    wt_prob = np.sum(wts, axis=0) / total_slices

    # sum and compute 100% probability
    total_prob = bg_prob + ncr_prob + ed_prob + et_prob + wt_prob
    bg_prob /= total_prob
    ncr_prob /= total_prob
    ed_prob /= total_prob
    et_prob /= total_prob
    wt_prob /= total_prob

    # round to 3 decimals
    bg_prob = np.round(bg_prob, 3)
    ncr_prob = np.round(ncr_prob, 3)
    ed_prob = np.round(ed_prob, 3)
    et_prob = np.round(et_prob, 3)
    wt_prob = np.round(wt_prob, 3)

    print(
        f"Probabilities {bg_prob + ncr_prob + ed_prob + et_prob + wt_prob}. "
        f"Background: {bg_prob}, "
        f"NCR: {ncr_prob}, "
        f"ED: {ed_prob}, "
        f"ET: {et_prob}, "
        f"WT: {wt_prob}."
    )

    # create output dir
    output_path = Path(args.output_path)
    if not os.path.exists(output_path):
        output_path.mkdir(parents=True, exist_ok=True)

    # save probabilities as json
    with open(output_path / "probabilities.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "bg_prob": bg_prob.tolist(),
                "ncr_prob": ncr_prob.tolist(),
                "ed_prob": ed_prob.tolist(),
                "et_prob": et_prob.tolist(),
                "wt_prob": wt_prob.tolist(),
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    main(args)
