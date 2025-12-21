# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from atommic.collections.common.parts import center_crop, is_none
from atommic.collections.reconstruction.metrics.reconstruction_metrics import (
    ReconstructionMetrics,
    mse,
    nmse,
    psnr,
    ssim,
)

METRIC_FUNCS = {"MSE": mse, "NMSE": nmse, "PSNR": psnr, "SSIM": ssim}


def main(args):  # noqa: MC0001
    # if json file
    if args.targets_dir.endswith(".json"):
        with open(args.targets_dir, "r", encoding="utf-8") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_dir).iterdir())

    crop_size = args.crop_size
    evaluation_type = args.evaluation_type

    scores = ReconstructionMetrics(METRIC_FUNCS, ddof=1 if evaluation_type == "per_slice" else 0)
    for target in tqdm(targets):
        reconstruction = h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r")[
            "reconstruction"
        ][()].squeeze()

        target_file = h5py.File(target, "r")
        if "reconstruction_sense" in target_file.keys():
            target = target_file["reconstruction_sense"][()].squeeze()
        elif "reconstruction_rss" in target_file.keys():
            target = target_file["reconstruction_rss"][()].squeeze()
        elif "reconstruction" in target_file.keys():
            target = target_file["reconstruction"][()].squeeze()
        else:
            target = target_file["target"][()].squeeze()

        if crop_size is not None:
            crop_size[0] = target.shape[-2] if target.shape[-2] < int(crop_size[0]) else int(crop_size[0])
            crop_size[1] = target.shape[-1] if target.shape[-1] < int(crop_size[1]) else int(crop_size[1])
            crop_size[0] = (
                reconstruction.shape[-2] if reconstruction.shape[-2] < int(crop_size[0]) else int(crop_size[0])
            )
            crop_size[1] = (
                reconstruction.shape[-1] if reconstruction.shape[-1] < int(crop_size[1]) else int(crop_size[1])
            )

            target = center_crop(target, crop_size)
            reconstruction = center_crop(reconstruction, crop_size)

        if "stanford_fullysampled" in args.targets_dir.lower():
            # remove the first 20 and the last 20 slices
            target = target[20:-20]
            reconstruction = reconstruction[20:-20]

        # check if any flipping is needed
        if not is_none(args.flip_target):
            if args.flip_target == "left_right":
                target = np.flip(target, axis=-1)
            elif args.flip_target == "up_down":
                target = np.flip(target, axis=-2)
            elif args.flip_target == "both":
                target = np.flip(np.flip(target, axis=-1), axis=-2)

        if not is_none(args.flip_reconstruction):
            if args.flip_reconstruction == "left_right":
                reconstruction = np.flip(reconstruction, axis=-1)
            elif args.flip_reconstruction == "up_down":
                reconstruction = np.flip(reconstruction, axis=-2)
            elif args.flip_reconstruction == "both":
                reconstruction = np.flip(np.flip(reconstruction, axis=-1), axis=-2)

        # normalize per slice
        for sl in range(target.shape[0]):
            target[sl] = target[sl] / np.max(np.abs(target[sl]))
            reconstruction[sl] = reconstruction[sl] / np.max(np.abs(reconstruction[sl]))
        target = np.abs(target)
        reconstruction = np.abs(reconstruction)

        maxvalue = max(np.max(target) - np.min(target), np.max(reconstruction) - np.min(reconstruction))

        if evaluation_type == "per_slice":
            for sl in range(target.shape[0]):
                scores.push(target[sl], reconstruction[sl], maxval=maxvalue)
        elif evaluation_type == "per_volume":
            scores.push(target, reconstruction, maxval=maxvalue)

    model = args.reconstructions_dir.split("/")
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
    parser.add_argument("reconstructions_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--crop_size", nargs="+", type=int)
    parser.add_argument("--flip_target", choices=["left_right", "up_down", "both", "none"], default="none")
    parser.add_argument("--flip_reconstruction", choices=["left_right", "up_down", "both", "none"], default="none")
    parser.add_argument("--evaluation_type", choices=["per_slice", "per_volume"], default="per_slice")
    parser.add_argument("--fill_target_path", action="store_true")
    parser.add_argument("--fill_pred_path", action="store_true")
    args = parser.parse_args()

    if args.fill_target_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.targets_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        args.targets_dir = os.path.join(input_dir, "reconstructions")

    if args.fill_pred_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.reconstructions_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        args.reconstructions_dir = os.path.join(input_dir, "reconstructions")

    main(args)
