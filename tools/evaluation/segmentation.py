# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import os
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

from atommic.collections.common.parts import center_crop
from atommic.collections.segmentation.metrics.segmentation_metrics import (
    SegmentationMetrics,
    binary_cross_entropy_with_logits_metric,
    dice_metric,
    f1_per_class_metric,
    hausdorff_distance_95_metric,
    iou_metric,
    precision_metric,
    recall_metric,
)

METRIC_FUNCS = {
    "BCE": binary_cross_entropy_with_logits_metric,
    "DICE": dice_metric,
    "F1": f1_per_class_metric,
    "HD95": lambda x, y: hausdorff_distance_95_metric(x, y, batched=False, sum_method="sum"),
    "IOU": iou_metric,
    "PRE": precision_metric,
    "REC": recall_metric,
}


def main(args):  # noqa: MC0001
    if args.flatten_dice:
        METRIC_FUNCS["DICE"] = lambda x, y: dice_metric(x, y, flatten=True)

    # if json file
    if args.targets_dir.endswith(".json"):
        with open(args.targets_dir, "r", encoding="utf-8") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_dir).iterdir())

    crop_size = args.crop_size
    dataset_format = args.dataset_format
    evaluation_type = args.evaluation_type
    exclude_prediction_background = args.exclude_prediction_background

    scores = SegmentationMetrics(METRIC_FUNCS, ddof=1 if evaluation_type == "per_slice" else 0)
    for target in tqdm(targets):
        fname = str(target).rsplit("/", maxsplit=1)[-1]
        if ".h5" in fname:
            fname = fname.split(".h5")[0]
        elif ".nii" in fname:
            fname = fname.split(".nii")[0]

        predictions = h5py.File(Path(args.segmentations_dir) / fname, "r")["segmentation"][()].squeeze()
        predictions = np.abs(predictions.astype(np.float32))
        predictions = np.where(predictions > 0.5, 1, 0)
        if args.sum_classes_method == "argmax":
            predictions = np.where(predictions.argmax(axis=1) > 0.5, 1, 0)
        elif args.sum_classes_method == "sum":
            predictions = np.where(predictions.sum(axis=1) > 0.5, 1, 0)

        if dataset_format == 'skm-tea':
            with h5py.File(target, "r") as hf:
                segmentation_labels = hf["seg"][()].squeeze()

                # combine label 2 and 3 (Lateral Tibial Cartilage and Medial Tibial Cartilage)
                tibial_cartilage = segmentation_labels[..., 2] + segmentation_labels[..., 3]
                # combine label 4 and 5 (Lateral Meniscus and Medial Meniscus)
                medial_meniscus = segmentation_labels[..., 4] + segmentation_labels[..., 5]

                # stack the labels
                target = np.stack(
                    [segmentation_labels[..., 0], segmentation_labels[..., 1], tibial_cartilage, medial_meniscus],
                    axis=0,
                )
                target = np.moveaxis(target, -1, 0)[30:-31]
        elif dataset_format == 'brats':
            target = str(target).replace("TrainingData", "TrainingSegmentations").replace(".nii.gz", "-seg.nii.gz")

            target = np.moveaxis(nib.load(target).get_fdata(), -1, 0)
            # remove the first 50 and last 65 slices
            target = target[51:-65]

            # Necrotic Tumor Core (NCR - label 1)
            ncr = np.zeros_like(target)
            ncr[target == 1] = 1
            # Peritumoral Edematous/Invaded Tissue (ED - label 2)
            ed = np.zeros_like(target)
            ed[target == 2] = 1
            # GD-Enhancing Tumor (ET - label 3)
            et = np.zeros_like(target)
            et[target == 3] = 1
            # Whole Tumor (WT — label 1, 2, or 3)
            wt = np.zeros_like(target)
            wt[target != 0] = 1
            target = np.stack([ncr, ed, et, wt], axis=1).astype(np.float32)

        target = np.abs(target.astype(np.float32))
        target = np.where(target > 0.5, 1, 0)
        if args.sum_classes_method == "argmax":
            target = np.where(target.argmax(axis=1) > 0.5, 1, 0)
        elif args.sum_classes_method == "sum":
            target = np.where(target.sum(axis=1) > 0.5, 1, 0)

        if crop_size is not None:
            crop_size[0] = target.shape[-2] if target.shape[-2] < int(crop_size[0]) else int(crop_size[0])
            crop_size[1] = target.shape[-1] if target.shape[-1] < int(crop_size[1]) else int(crop_size[1])
            crop_size[0] = predictions.shape[-2] if predictions.shape[-2] < int(crop_size[0]) else int(crop_size[0])
            crop_size[1] = predictions.shape[-1] if predictions.shape[-1] < int(crop_size[1]) else int(crop_size[1])

            target = center_crop(target, crop_size)
            predictions = center_crop(predictions, crop_size)

        if exclude_prediction_background:
            predictions = predictions[:, 1:, ...]

        if evaluation_type == "per_slice":
            target = np.expand_dims(target, axis=1)
            predictions = np.expand_dims(predictions, axis=1)
            for sl in range(target.shape[0]):
                scores.push(target[sl], predictions[sl])
        elif evaluation_type == "per_volume":
            scores.push(target, predictions)

    model = args.segmentations_dir.split("/")
    model = model[-4] if model[-4] != "default" else model[-5]
    print(f"{model}: {repr(scores)}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # if file exists dont' overwrite, but append in a new line
        with open(output_dir / "results.txt", "a", encoding="utf-8") as f:
            f.write(f"{model}: {repr(scores)}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("targets_dir", type=str)
    parser.add_argument("segmentations_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--crop_size", nargs="+", type=int)
    parser.add_argument("--dataset_format", choices=["skm-tea", "brats", "private"], default="private")
    parser.add_argument("--evaluation_type", choices=["per_slice", "per_volume"], default="per_slice")
    parser.add_argument("--sum_classes_method", choices=["sum", "argmax", "none"], default="none")
    parser.add_argument("--flatten_dice", action="store_true")
    parser.add_argument("--exclude_prediction_background", action="store_true")
    parser.add_argument("--fill_target_path", action="store_true")
    parser.add_argument("--fill_pred_path", action="store_true")
    args = parser.parse_args()

    if args.fill_target_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.targets_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        args.targets_dir = os.path.join(input_dir, "segmentations")

    if args.fill_pred_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.segmentations_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        args.segmentations_dir = os.path.join(input_dir, "segmentations")

    main(args)
