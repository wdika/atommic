#!/bin/bash
echo "
Compute the mask for the Calgary-Campinas 359 dataset.

The data download link is available at: https://sites.google.com/view/calgary-campinas-dataset/home

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

# Prompt the user to enter the path to the downloaded mask data
echo "Please enter the (downloaded) data directory:"
read INPUT_MASK_DIR

# Check if the input mask directory exists
if [ ! -d "$INPUT_MASK_DIR" ]; then
  echo "The input mask directory does not exist. Please try again."
  exit 1
fi

# Prompt the user to enter the output directory for the preprocessed data
echo "Please enter the output directory for the preprocessed data:"
read OUTPUT_DIR

# Prompt the user to enter if 5 or 10 or both accelerations are to be used
echo "Please enter the acceleration factor (5 or 10 or both - Default):"
read ACCELERATION

# Compute the masks
echo "Computing the masks..."
python projects/reconstruction/CC359/scripts/compute_masks.py $INPUT_DIR $INPUT_MASK_DIR $OUTPUT_DIR $ACCELERATION
echo "Done!"
