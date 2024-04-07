# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import json
from pathlib import Path


def read_h5_files(dataset_path):
    """Read all h5 files in a directory"""
    return list(Path(dataset_path).iterdir())


def main(args):
    # read all h5 files in the data directory
    all_filenames_train = read_h5_files(args.data_path / "PD/multicoil_train")
    all_filenames_train += read_h5_files(args.data_path / "PDFS/multicoil_train")
    all_filenames_train = [str(filename) for filename in all_filenames_train]

    all_filenames_val = read_h5_files(args.data_path / "PD/multicoil_val")
    all_filenames_val += read_h5_files(args.data_path / "PDFS/multicoil_val")
    all_filenames_val = [str(filename) for filename in all_filenames_val]

    print(f"Number of train files: {len(all_filenames_train)}")
    print(f"Number of val files: {len(all_filenames_val)}")

    # create a directory to store the folds
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # write the train, val and test filenames to a json file
    with open(output_path / "multicoil_train.json", "w", encoding="utf-8") as f:
        json.dump(all_filenames_train, f)
    with open(output_path / "multicoil_val.json", "w", encoding="utf-8") as f:
        json.dump(all_filenames_val, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, default=None, help="Path to the data directory.")
    parser.add_argument("output_path", type=Path, default="data/folds", help="Path to the output directory.")
    args = parser.parse_args()
    main(args)
