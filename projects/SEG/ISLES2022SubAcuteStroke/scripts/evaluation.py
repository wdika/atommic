# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Metrics functions taken and adapted from : https://github.com/ezequieldlrosa/isles22

import json
import os
import warnings
from pathlib import Path

import cc3d  # pylint: disable=import-error
import h5py
import nibabel as nib
import numpy as np
from runstats import Statistics

from atommic.collections.common.parts import center_crop


def compute_dice(im1, im2, voxel_volume=0.0, empty_value=1.0):  # pylint: disable=unused-argument
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as im1. If not boolean, it will be converted.
    voxel_volume : scalar, float (ml)
    empty_value : scalar, float.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.

    This function has been adapted from the Verse Challenge repository:
    https://github.com/anjany/verse/blob/main/utils/eval_utilities.py
    """

    # binarize im1 and im2
    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_value

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


def compute_absolute_volume_difference(im1, im2, voxel_volume=0.0):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_volume : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)
    voxel_volume = np.asarray(voxel_volume).astype(np.float32)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_volume
    prediction_volume = np.sum(im2) * voxel_volume
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff


def compute_absolute_lesion_difference(
    ground_truth, prediction, voxel_volume=0.0, connectivity=26  # pylint: disable=unused-argument
):
    """
    Computes the absolute lesion difference between two masks. The number of lesions are counted for
    each volume, and their absolute difference is computed.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_volume : scalar, float (ml)
    connectivity : scalar, int.

    Returns
    -------
    abs_les_diff : int
        Absolute lesion difference as integer.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    """
    ground_truth = np.asarray(ground_truth).astype(np.bool_)
    prediction = np.asarray(prediction).astype(np.bool_)

    _, ground_truth_numb_lesion = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
    _, prediction_numb_lesion = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    abs_les_diff = abs(ground_truth_numb_lesion - prediction_numb_lesion)

    return abs_les_diff


def compute_lesion_f1_score(
    ground_truth, prediction, voxel_volume=0.0, empty_value=1.0, connectivity=26  # pylint: disable=unused-argument
):
    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_volume : scalar, float (ml)
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction
    image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    ground_truth = np.asarray(ground_truth).astype(np.bool_)
    prediction = np.asarray(prediction).astype(np.bool_)
    tp = 0
    fp = 0
    fn = 0

    # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)

    # Iterate over ground_truth clusters to find tp and fn.
    # tp and fn are only computed if the ground-truth is not empty.
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # iterate over prediction clusters to find fp.
    # fp are only computed if the prediction image is not empty.
    labeled_prediction, N = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    # Define case when both images are empty.
    if tp + fp + fn == 0:
        _, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
        if N == 0:
            f1_score = empty_value
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    return f1_score


METRIC_FUNCS = {
    "DICE": compute_dice,
    "AVD": compute_absolute_volume_difference,
    "ALD": compute_absolute_lesion_difference,
    "L-F1": compute_lesion_f1_score,
}


class ISLES2022SubAcuteStrokeSegmentationMetrics:
    """Maintains running statistics for a given collection of metrics."""

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, segmentations, voxel_volume):
        """
        Pushes a new batch of metrics to the running statistics.
        Args:
            target: target image
            segmentations: predicted segmentation
            voxel_volume: voxel volume in ml
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        for metric, func in METRIC_FUNCS.items():
            self.metrics_scores[metric].push(func(target, segmentations, voxel_volume))

    def means(self):
        """
        Mean of the means of each metric.
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """
        Standard deviation of the means of each metric.
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """
        Representation of the metrics.
        Returns:
            str: A string representation of the metrics.
        """
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = " ".join(f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}" for name in metric_names) + "\n"

        return res


def main(args):
    # if json file
    if args.targets_dir.endswith(".json"):
        with open(args.targets_dir, "r", encoding="utf-8") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_dir).iterdir())

    crop_size = args.crop_size
    evaluation_type = args.evaluation_type

    scores = ISLES2022SubAcuteStrokeSegmentationMetrics(METRIC_FUNCS)
    for target in targets:
        subj = str(target).rsplit("/", maxsplit=1)[-1].split('.')[0]
        predictions = h5py.File(Path(args.segmentations_dir) / subj, "r")["segmentation"][()].squeeze()
        predictions = np.where(np.abs(predictions.astype(np.float32)) > 0.5, 1, 0)

        # Labels are stacked as ADC, DWI, FLAIR
        labels = (
            nib.load(Path(args.targets_segmentations_dir) / Path(f"{subj}-seg.nii.gz")).get_fdata().astype(np.float32)
        )
        labels = np.moveaxis(labels, -1, 0)
        lesions = np.zeros_like(labels)
        lesions[labels == 1] = 1

        # get voxel volume
        voxel_volume = np.prod(nib.load(Path(args.targets_data_dir) / Path(target)).header.get_zooms()) / 1000

        if crop_size is not None:
            crop_size[0] = lesions.shape[-2] if lesions.shape[-2] < int(crop_size[0]) else int(crop_size[0])
            crop_size[1] = lesions.shape[-1] if lesions.shape[-1] < int(crop_size[1]) else int(crop_size[1])
            crop_size[0] = predictions.shape[-2] if predictions.shape[-2] < int(crop_size[0]) else int(crop_size[0])
            crop_size[1] = predictions.shape[-1] if predictions.shape[-1] < int(crop_size[1]) else int(crop_size[1])

            lesions = center_crop(lesions, crop_size)
            predictions = center_crop(predictions, crop_size)

        if evaluation_type == "per_slice":
            lesions = np.expand_dims(lesions, axis=1)
            predictions = np.expand_dims(predictions, axis=1)
            for sl in range(lesions.shape[0]):
                scores.push(lesions[sl], predictions[sl], voxel_volume)
        elif evaluation_type == "per_volume":
            scores.push(lesions, predictions, voxel_volume)

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
    parser.add_argument("targets_data_dir", type=str)
    parser.add_argument("targets_segmentations_dir", type=str)
    parser.add_argument("segmentations_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--crop_size", nargs="+", type=int)
    parser.add_argument("--evaluation_type", choices=["per_slice", "per_volume"], default="per_slice")
    parser.add_argument("--fill_pred_path", action="store_true")
    args = parser.parse_args()

    if args.fill_pred_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.segmentations_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        args.segmentations_dir = os.path.join(input_dir, "segmentations")

    main(args)
