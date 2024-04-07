# coding=utf-8
__author__ = "Dimitris Karkalousos"

import os

import numpy as np


def process_line(line, metrics):
    model = line.split(":")[0]
    if "_SENSE" in model:
        model = model.split("_SENSE")[0]
    if "_128CH" in model:
        model = model.split("_128CH")[0]
    if model in ("ZeroFilled_SENSE", "ZeroFilled_RSS"):
        model = "ZeroFilled"

    result_dict = {}
    for metric in metrics:
        value = line.split(f"{metric} = ")[1]
        mean_value = np.round(float(value.split(" ")[0]), 3)
        if len(str(mean_value)) == 4:
            mean_value = str(mean_value) + "0"
        elif len(str(mean_value)) == 3:
            mean_value = str(mean_value) + "00"
        std_value = np.round(float(value.split("+/- ")[1].split(" ")[0]), 3)
        if len(str(std_value)) == 4:
            std_value = str(std_value) + "0"
        elif len(str(std_value)) == 3:
            std_value = str(std_value) + "00"

        result_dict[metric] = (mean_value, std_value)

    return model, result_dict


def simplify_code(parent_dir, dataset_name, lines, results_file):  # pylint: disable=inconsistent-return-statements
    table_results = []

    if parent_dir == 'REC':
        metrics = ["SSIM", "PSNR"]
    elif parent_dir == 'qMRI':
        metrics = ["SSIM", "PSNR", "NMSE"]
    elif parent_dir == 'SEG':
        metrics = (
            ["ALD", "AVD", "DICE", "L-F1"]
            if 'ISLES2022SubAcuteStroke' in results_file
            else ["DICE", "F1", "HD95", "IOU"]
        )
    else:
        return  # Handle other cases if necessary

    table_results.append(f"{dataset_name} \n")
    table_results.append(f"Model & {' & '.join(metrics)} \n")

    for line in lines:
        if not line.strip() == "":
            model, result_dict = process_line(line, metrics)
            result_str = (
                f"{model} & " f"{' & '.join([f'{result[0]} +/- {result[1]}' for result in result_dict.values()])} \n"
            )
            table_results.append(result_str)

    table_results.append("\n")
    return table_results


def main(args):  # noqa: MC0001
    results_dir = args.out_dir

    # get all subdirs
    subdirs = [x[0] for x in os.walk(results_dir)][1:]

    # get all results.txt files, and store the parent dir together if it is MTL, qMRI, REC, or SEG
    results = {}
    for subdir in subdirs:
        # search for all results.txt files
        files = os.listdir(subdir)
        if 'results.txt' in files:
            # get the path to results.txt
            path_to_results = os.path.join(subdir, 'results.txt')
            if 'MTL' in path_to_results:
                if 'reconstruction' in path_to_results:
                    parent_dir = 'REC'
                elif 'segmentation' in path_to_results:
                    parent_dir = 'SEG'
            elif 'qMRI' in path_to_results:
                parent_dir = 'qMRI'
            elif 'REC' in path_to_results:
                parent_dir = 'REC'
            elif 'SEG' in path_to_results:
                parent_dir = 'SEG'

            results[path_to_results] = parent_dir

    # iterate the results dictionary and read the results.txt files
    table_results = []
    for results_file, parent_dir in results.items():
        dataset_name = results_file.split("evaluation_per_slice/")[1].split("/")[0]
        with open(results_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        table_results.extend(simplify_code(parent_dir, dataset_name, lines, results_file))
    table_results.append("\n")
    table_results.append("\n")
    table_results.append("\n")

    # write the table_results to a file as txt
    with open(os.path.join(results_dir, "ATOMMIC_paper_results.txt"), "w", encoding="utf-8") as f:
        for line in table_results:
            f.write(line)

    # format as latex table
    with open(os.path.join(results_dir, "ATOMMIC_paper_results_latex.txt"), "w", encoding="utf-8") as f:
        for line in table_results:
            f.write(line.replace(" +/- ", " $\pm$ ").replace(" \n", " \\\\ \n"))  # noqa: W605


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.out_dir = "output_data_dir/atommic"
    main(args)
