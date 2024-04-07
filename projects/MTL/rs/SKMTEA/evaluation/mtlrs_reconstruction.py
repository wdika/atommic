# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from atommic.collections.reconstruction.metrics.reconstruction_metrics import (
    ReconstructionMetrics,
    mse,
    nmse,
    psnr,
    ssim,
)

METRIC_FUNCS = {"MSE": mse, "NMSE": nmse, "PSNR": psnr, "SSIM": ssim}


def main(args):
    # if json file
    if args.targets_dir.endswith(".json"):
        with open(args.targets_dir, "r", encoding="utf-8") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_dir).iterdir())

    evaluation_type = args.evaluation_type

    scores = ReconstructionMetrics(METRIC_FUNCS)
    for target in tqdm(targets):
        reconstruction = h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r")[
            "reconstruction"
        ][()].squeeze()

        target = h5py.File(target, "r")["reconstruction"][()].squeeze()

        # normalize per slice
        for sl in range(target.shape[0]):
            target[sl] = target[sl] / np.max(np.abs(target[sl]))
            reconstruction[sl] = reconstruction[sl] / np.max(np.abs(reconstruction[sl]))
        reconstruction = np.abs(reconstruction).real.astype(np.float32)
        target = np.abs(target).real.astype(np.float32)

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
        for root, dirs, files in os.walk(args.reconstructions_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        # check if after dir we have "/reconstructions" or "/predictions" dir
        if os.path.exists(os.path.join(input_dir, "reconstructions")):
            args.reconstructions_dir = os.path.join(input_dir, "reconstructions")
        elif os.path.exists(os.path.join(input_dir, "predictions")):
            args.reconstructions_dir = os.path.join(input_dir, "predictions")

    main(args)
