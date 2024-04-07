#!/bin/bash
echo "
Preprocessing pipeline for the AHEAD dataset.

The data download link is available at: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/IHZGQM.

Please make sure you have ``ismrmrd`` installed.

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
python projects/quantitative/AHEAD/scripts/preprocessing.py $INPUT_DIR $OUTPUT_DIR --plane axial --slice_range 120 171
echo "Computing the segmentation masks..."
python projects/quantitative/AHEAD/scripts/compute_segmentation_masks.py $OUTPUT_DIR $OUTPUT_DIR/segmentation_masks/
echo "Done!"
