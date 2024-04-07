#!/bin/bash
echo "Enter the path to the parent data directory of the SKMTEA dataset:"
read parent_data_dir_skmtea
echo "Enter the path to the parent data directory of the AHEAD dataset:"
read parent_data_dir_ahead
echo "Enter the path to the parent data directory of the CC359 dataset:"
read parent_data_dir_cc359
echo "Enter the path to the parent data directory of the fastMRIBrainMulticoil dataset:"
read parent_data_dir_fastmri
echo "Enter the path to the parent data directory of the StanfordKnees2019 dataset:"
read parent_data_dir_stanford
echo "Enter the path to the parent data directory of the BraTS2023AdultGlioma dataset:"
read parent_data_dir_brats
echo "Enter the path to the parent data directory of the ISLES2022SubAcuteStroke dataset:"
read parent_data_dir_isles
echo "Enter the path to the parent output directory."
read output_dir

# create the output_dir if it does not exist
mkdir -p ${output_dir}

# go inside projects/ATOMMIC_paper/MTL/SKMTEA/ folder, find all files inside any subfolder and replace parent_data_dir with the given parent_data_dir_skmtea
find projects/ATOMMIC_paper/MTL/SKMTEA/ -type f -exec sed -i "s|parent_data_dir|${parent_data_dir_skmtea}|g" {} +
# go inside projects/ATOMMIC_paper/SEG/SKMTEA/ folder, find all files inside any subfolder and replace parent_data_dir with the given parent_data_dir_skmtea
find projects/ATOMMIC_paper/SEG/SKMTEA/ -type f -exec sed -i "s|parent_data_dir|${parent_data_dir_skmtea}|g" {} +
# go inside projects/ATOMMIC_paper/qMRI/AHEAD/ folder, find all files inside any subfolder and replace parent_data_dir with the given parent_data_dir_skmtea
find projects/ATOMMIC_paper/qMRI/AHEAD/ -type f -exec sed -i "s|parent_data_dir|${parent_data_dir_ahead}|g" {} +
# go inside projects/ATOMMIC_paper/REC/CC359/ folder, find all files inside any subfolder and replace parent_data_dir with the given parent_data_dir_cc359
find projects/ATOMMIC_paper/REC/CC359/ -type f -exec sed -i "s|parent_data_dir|${parent_data_dir_cc359}|g" {} +
# go inside projects/ATOMMIC_paper/REC/fastMRIBrainsMulticoil/ folder, find all files inside any subfolder and replace parent_data_dir with the given parent_data_dir_fastmri
find projects/ATOMMIC_paper/REC/fastMRIBrainsMulticoil/ -type f -exec sed -i "s|parent_data_dir|${parent_data_dir_fastmri}|g" {} +
# go inside projects/ATOMMIC_paper/REC/StanfordKnees2019/ folder, find all files inside any subfolder and replace parent_data_dir with the given parent_data_dir_stanford
find projects/ATOMMIC_paper/REC/StanfordKnees2019/ -type f -exec sed -i "s|parent_data_dir|${parent_data_dir_stanford}|g" {} +
# go inside projects/ATOMMIC_paper/SEG/BraTS2023AdultGlioma/ folder, find all files inside any subfolder and replace parent_data_dir with the given parent_data_dir_brats
find projects/ATOMMIC_paper/SEG/BraTS2023AdultGlioma/ -type f -exec sed -i "s|parent_data_dir|${parent_data_dir_brats}|g" {} +
# go inside projects/ATOMMIC_paper/SEG/ISLES2022SubAcuteStroke/ folder, find all files inside any subfolder and replace parent_data_dir with the given parent_data_dir_isles
find projects/ATOMMIC_paper/SEG/ISLES2022SubAcuteStroke/ -type f -exec sed -i "s|parent_data_dir|${parent_data_dir_isles}|g" {} +
# go inside projects/ATOMMIC_paper/ folder, find all files inside any subfolder and replace output_dir with the given output_dir except the read_data_and_output_paths.sh file
find projects/ATOMMIC_paper/ -type f -not -name "set_paths.sh" -exec sed -i "s|output_data_dir|${output_dir}|g" {} +
