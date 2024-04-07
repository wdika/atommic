#!/bin/bash
echo "
Preprocessing pipeline for the Stanford Fullysampled 3D FSE Knees 2019 dataset.

The data download link is available at: http://mridata.org/list?project=Stanford%20Fullysampled%203D%20FSE%20Knees.

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
python projects/reconstruction/StanfordKnees2019/scripts/preprocess_dataset.py $INPUT_DIR $OUTPUT_DIR
echo "Generating train, val, and test splits..."
python projects/reconstruction/StanfordKnees2019/scripts/split_sets_json.py $OUTPUT_DIR
echo "Done!"
