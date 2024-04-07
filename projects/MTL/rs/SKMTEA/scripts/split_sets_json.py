# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import json
from pathlib import Path


def main(args):
    if args.data_type == "raw":
        data_type = "files_recon_calib-24"
    else:
        data_type = "image_files"

    # remove "annotations/v1.0.0/" from args.annotations_path and add "files_recon_calib-24" to get the raw_data_path
    raw_data_path = Path(args.annotations_path).parent.parent / data_type

    # get train.json, val.json and test.json filenames from args.annotations_path
    annotations_sets = list(Path(args.annotations_path).iterdir())
    for annotation_set in annotations_sets:
        set_name = Path(annotation_set).name

        # read json file
        with open(annotation_set, "r", encoding="utf-8") as f:
            annotation_set = json.load(f)

        # read the "images" key and for every instance get the "file_name" key
        filenames = [f'{raw_data_path}/{image["file_name"]}' for image in annotation_set["images"]]

        # create a directory to store the folds
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # write the train, val and test filenames to a json file
        with open(output_path / f"{data_type}_{set_name}", "w", encoding="utf-8") as f:
            json.dump(filenames, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_path", type=Path, default=None, help="Path to the annotations json file.")
    parser.add_argument("output_path", type=Path, default=None, help="Path to the output directory.")
    parser.add_argument("--data_type", choices=["raw", "image"], default="raw", help="Type of data to split.")
    args = parser.parse_args()
    main(args)
