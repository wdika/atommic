# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import os
from pathlib import Path

import h5py
import numpy as np
from runstats import Statistics
from tqdm import tqdm

from atommic.collections.common.parts import center_crop
from atommic.collections.reconstruction.metrics.reconstruction_metrics import mse, nmse, psnr, ssim

METRIC_FUNCS = {"MSE": mse, "NMSE": nmse, "PSNR": psnr, "SSIM": ssim}


class qMRIMetrics:
    """Maintains running statistics for a given collection of metrics."""

    def __init__(self, metric_funcs):
        """Inits :class:`qMRIMetrics`.

        Parameters
        ----------
        metric_funcs : dict
            A dict where the keys are metric names and the values are Python functions for evaluating that metric.
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}

    def push(self, x, y):
        """Pushes a new batch of metrics to the running statistics.

        Parameters
        ----------
        x : np.ndarray
            Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
            images, the first dimension should be 1.
        y : np.ndarray
            Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
            images, the first dimension should be 1.

        Returns
        -------
        dict
            A dict where the keys are metric names and the values are the computed metric scores.
        """
        for metric, func in METRIC_FUNCS.items():
            self.metrics_scores[metric].push(func(x, y))

    def means(self):
        """Mean of the means of each metric."""
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """Standard deviation of the means of each metric."""
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """Representation of the metrics."""
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

    scores = qMRIMetrics(METRIC_FUNCS)
    for target in tqdm(targets):
        fname = str(target).rsplit("/", maxsplit=1)[-1]

        target = h5py.File(target, "r")["qmaps"][()].squeeze()
        prediction = h5py.File(Path(args.predictions_dir) / fname, "r")["qmaps"][()].squeeze()
        anatomy_mask = h5py.File(Path(args.segmentations_masks_dir) / fname, "r")["anatomy_mask"][()].squeeze()

        if crop_size is not None:
            crop_size[0] = target.shape[-2] if target.shape[-2] < int(crop_size[0]) else int(crop_size[0])
            crop_size[1] = target.shape[-1] if target.shape[-1] < int(crop_size[1]) else int(crop_size[1])
            crop_size[0] = prediction.shape[-2] if prediction.shape[-2] < int(crop_size[0]) else int(crop_size[0])
            crop_size[1] = prediction.shape[-1] if prediction.shape[-1] < int(crop_size[1]) else int(crop_size[1])

            target = center_crop(target, crop_size)
            prediction = center_crop(prediction, crop_size)
            anatomy_mask = center_crop(anatomy_mask, crop_size)

        # normalize per slice
        for i in range(target.shape[0]):
            # normalize per echo
            for j in range(target.shape[1]):
                target[i, j] = np.abs(target[i, j] / np.max(np.abs(target[i, j]))) * anatomy_mask[i]
                prediction[i, j] = np.abs(prediction[i, j] / np.max(np.abs(prediction[i, j]))) * anatomy_mask[i]

        if evaluation_type == "per_slice":
            for sl in range(target.shape[0]):
                for qmap_idx in range(prediction.shape[1]):
                    scores.push(target[sl, qmap_idx], prediction[sl, qmap_idx])
        elif evaluation_type == "per_volume":
            for qmap_idx in range(prediction.shape[1]):
                scores.push(target[:, qmap_idx], prediction[:, qmap_idx])

    model = args.predictions_dir.split("/")
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
    parser.add_argument("segmentations_masks_dir", type=str)
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--crop_size", nargs="+", type=int)
    parser.add_argument("--evaluation_type", choices=["per_slice", "per_volume"], default="per_slice")
    parser.add_argument("--fill_target_path", action="store_true")
    parser.add_argument("--fill_pred_path", action="store_true")
    args = parser.parse_args()

    if args.fill_target_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.targets_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        # check if after dir we have "/reconstructions" or "/predictions" dir
        if os.path.exists(os.path.join(input_dir, "reconstructions")):
            args.targets_dir = os.path.join(input_dir, "reconstructions")
        elif os.path.exists(os.path.join(input_dir, "predictions")):
            args.targets_dir = os.path.join(input_dir, "predictions")

    if args.fill_pred_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.predictions_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        # check if after dir we have "/reconstructions" or "/predictions" dir
        if os.path.exists(os.path.join(input_dir, "reconstructions")):
            args.predictions_dir = os.path.join(input_dir, "reconstructions")
        elif os.path.exists(os.path.join(input_dir, "predictions")):
            args.predictions_dir = os.path.join(input_dir, "predictions")

    main(args)
