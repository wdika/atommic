# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import json
from pathlib import Path

FILENAMES_TO_EXCLUDE = [
    "file_brain_AXFLAIR_201_6002914.h5",
    "file_brain_AXFLAIR_201_6003003.h5",
    "file_brain_AXFLAIR_202_6000508.h5",
    "file_brain_AXFLAIR_202_6000531.h5",
    "file_brain_AXFLAIR_202_6000539.h5",
    "file_brain_AXFLAIR_202_6000554.h5",
    "file_brain_AXFLAIR_202_6000420.h5",
    "file_brain_AXFLAIR_202_6000486.h5",
    "file_brain_AXFLAIR_202_6000586.h5",
]


def main(args):
    # read all h5 files in the data directory
    train_set = list((Path(args.data_path) / "multicoil_train").iterdir())
    val_set = list((Path(args.data_path) / "multicoil_val").iterdir())

    # remove the files that we want to exclude
    train_set = [str(f) for f in train_set if f.name not in FILENAMES_TO_EXCLUDE]
    val_set = [str(f) for f in val_set if f.name not in FILENAMES_TO_EXCLUDE]

    # create a directory to store the folds
    output_path = Path(args.data_path) / "json"
    output_path.mkdir(parents=True, exist_ok=True)

    # write the train and val filenames to a json file
    with open(output_path / "multicoil_train.json", "w", encoding="utf-8") as f:
        json.dump(train_set, f)
    with open(output_path / "multicoil_val.json", "w", encoding="utf-8") as f:
        json.dump(val_set, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the data directory.")
    args = parser.parse_args()
    main(args)
