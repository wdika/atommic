#!/bin/bash
echo "
Preprocessing pipeline for the ISLES2022SubAcuteStroke dataset.

For more information, please refer to https://isles22.grand-challenge.org/dataset/ and check the following
paper https://www.nature.com/articles/s41597-022-01875-5.

Please make sure you have the following packages installed: argparse, connected-components-3d, json, nibabel, numpy,
pathlib, random, simpleitk, tqdm.

Starting the preprocessing...
"

# Prompt the user to enter the path to the downloaded data
echo "Please enter the (downloaded) data directory:"
read INPUT_DIR

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "The input directory does not exist. Please try again."
  exit 1
fi

# Prompt the user to enter the output directory for the preprocessed data
echo "Please enter the output directory for the preprocessed data:"
read OUTPUT_DIR

# Run the preprocessing pipeline
echo "Running the preprocessing..."
python projects/segmentation/ISLES2022SubAcuteStroke/scripts/preprocess_dataset.py $INPUT_DIR $OUTPUT_DIR
echo "Generating train, val, and test splits..."
python projects/segmentation/ISLES2022SubAcuteStroke/scripts/split_sets_json.py $OUTPUT_DIR
echo "Done!"
