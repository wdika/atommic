# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import json
import random
from pathlib import Path

import numpy as np


def generate_fold(filenames):
    """Generate a train, val and test set from a list of filenames"""
    # Path to str
    filenames = [str(filename) for filename in filenames]

    # shuffle the filenames
    random.shuffle(filenames)

    # split the filenames into train, val and test with 70%, 15% and 15% respectively
    train_fnames = np.array(filenames[: int(len(filenames) * 0.7)])
    # remove train filenames from all filenames
    filenames = np.setdiff1d(filenames, train_fnames)
    # split the remaining filenames into val and test with 50% and 50% respectively
    val_fnames = np.array(filenames[: int(len(filenames) * 0.5)])
    # remove val filenames from all filenames
    filenames = np.setdiff1d(filenames, val_fnames)
    test_fnames = np.array(filenames)

    return train_fnames.tolist(), val_fnames.tolist(), test_fnames.tolist()


def main(args):
    # read all h5 files in the data directory
    all_filenames = list((Path(args.data_path) / "preprocessed").iterdir())

    # create n folds
    folds = [generate_fold(all_filenames) for _ in range(args.nfolds)]

    # create a directory to store the folds
    output_path = Path(args.data_path) / "folds"
    output_path.mkdir(parents=True, exist_ok=True)

    # write each fold to a json file
    for i, fold in enumerate(folds):
        train_set, val_set, test_set = fold

        # write the train, val and test filenames to a json file
        with open(output_path / f"fold_{i}_train.json", "w", encoding="utf-8") as f:
            json.dump(train_set, f)
        with open(output_path / f"fold_{i}_val.json", "w", encoding="utf-8") as f:
            json.dump(val_set, f)
        with open(output_path / f"fold_{i}_test.json", "w", encoding="utf-8") as f:
            json.dump(test_set, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the data directory.")
    parser.add_argument("--nfolds", type=int, default=1, help="Number of folds to create.")
    args = parser.parse_args()
    main(args)
