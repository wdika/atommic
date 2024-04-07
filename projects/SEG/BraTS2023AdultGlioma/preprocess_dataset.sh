#!/bin/bash
echo "
Preprocessing pipeline for the BraTS2023AdultGlioma dataset.

For more information, please refer to https://www.synapse.org/#!Synapse:syn51156910/wiki/ and check the following
papers:
- https://arxiv.org/pdf/1811.02629.pdf,
- https://arxiv.org/pdf/2305.17033.pdf.
Data download link (registration required): https://www.synapse.org/#!Synapse:syn51156910/wiki/622351.

Please make sure you have the following packages installed: argparse, json, nibabel, numpy, pathlib, random, tqdm.

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
python projects/segmentation/BraTS2023AdultGlioma/scripts/preprocess_dataset.py $INPUT_DIR $OUTPUT_DIR
echo "Generating train and val splits..."
python projects/segmentation/BraTS2023AdultGlioma/scripts/split_sets_json.py $OUTPUT_DIR
echo "Computing the segmentation classes probabilities..."
python projects/segmentation/BraTS2023AdultGlioma/scripts/compute_segmentation_classes_probabilities.py $OUTPUT_DIR $OUTPUT_DIR
echo "Done!"
