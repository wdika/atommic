#!/bin/bash
echo "
Preprocessing pipeline for the Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) 2021 Dataset.

For more information, please refer to https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7
and check the following paper https://openreview.net/forum?id=YDMFgD_qJuA.

Generating train, val, and test sets...
"

# Prompt the user to enter the path to the downloaded annotations directory
echo "Please enter the (downloaded) annotations data directory:"
read INPUT_DIR

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "The input directory does not exist. Please try again."
  exit 1
fi

# Prompt the user to enter the output directory for the generated json files
echo "Please enter the output directory for the generated json files:"
read OUTPUT_DIR

# Run the json generation script
python projects/segmentation/SKMTEA/scripts/split_sets_json.py $INPUT_DIR $OUTPUT_DIR --data_type image
echo "Done!"
