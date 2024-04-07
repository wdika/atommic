# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import json
import random
from pathlib import Path

import numpy as np


def generate_fold(filenames):
    """Generate a train, val and test set from a list of filenames"""
    data_parent_dir = Path(filenames[0]).parent

    # Path to str
    filenames = [str(filename) for filename in filenames]

    # keep only the filename, so drop the "-t1c.nii.gz", "-t1n.nii.gz", "-t2f.nii.gz", or "-t2w.nii.gz"
    filenames = [filename.split("/")[-1] for filename in filenames]
    # keep only the unique filenames
    filenames = np.unique(filenames)

    # shuffle the filenames
    random.shuffle(filenames)

    # split the filenames into train and val with 80% and 20% respectively
    train_fnames = np.array(filenames[: int(len(filenames) * 0.8)]).tolist()
    # remove train filenames from all filenames
    filenames = np.setdiff1d(filenames, train_fnames)
    # since we have already removed the train filenames, we can use the remaining filenames as val
    val_fnames = filenames.tolist()

    # set full path
    train_fnames = [str(data_parent_dir / filename) for filename in train_fnames]
    val_fnames = [str(data_parent_dir / filename) for filename in val_fnames]

    return train_fnames, val_fnames


def main(args):
    # read all nii.gz files in the data directory
    all_filenames = list((Path(args.data_path) / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData").iterdir())

    # create n folds
    folds = [generate_fold(all_filenames) for _ in range(args.nfolds)]

    # create a directory to store the folds
    output_path = Path(args.data_path) / "folds"
    output_path.mkdir(parents=True, exist_ok=True)

    # write each fold to a json file
    for i, fold in enumerate(folds):
        train_set, val_set = fold

        # write the train, val and test filenames to a json file
        with open(output_path / f"fold_{i}_train.json", "w", encoding="utf-8") as f:
            json.dump(train_set, f)
        with open(output_path / f"fold_{i}_val.json", "w", encoding="utf-8") as f:
            json.dump(val_set, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the data directory.")
    parser.add_argument("--nfolds", type=int, default=1, help="Number of folds to create.")
    args = parser.parse_args()
    main(args)
