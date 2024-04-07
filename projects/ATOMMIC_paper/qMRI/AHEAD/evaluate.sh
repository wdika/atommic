python tools/evaluation/reconstruction.py \
parent_data_dir/ahead/preprocessed/test output_data_dir/atommic/REC/predictions/AHEAD_gaussian2d_12x_Test/CIRIM/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/qMRI/evaluation_per_slice/AHEAD_gaussian2d_12x_Test/REC --fill_pred_path
python tools/evaluation/reconstruction.py \
parent_data_dir/ahead/preprocessed/test output_data_dir/atommic/REC/predictions/AHEAD_gaussian2d_12x_Test/VarNet/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/qMRI/evaluation_per_slice/AHEAD_gaussian2d_12x_Test/REC --fill_pred_path
python tools/evaluation/qmapping.py \
output_data_dir/atommic/qMRI/targets/AHEAD_gaussian2d_12x_Test/SENSE/default/ parent_data_dir/ahead/segmentation_masks/test/ \
output_data_dir/atommic/qMRI/predictions/AHEAD_gaussian2d_12x_Test/qCIRIM/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/qMRI/evaluation_per_slice/AHEAD_gaussian2d_12x_Test/qMRI --fill_target_path --fill_pred_path
python tools/evaluation/qmapping.py \
output_data_dir/atommic/qMRI/targets/AHEAD_gaussian2d_12x_Test/SENSE/default/ parent_data_dir/ahead/segmentation_masks/test/ \
output_data_dir/atommic/qMRI/predictions/AHEAD_gaussian2d_12x_Test/qVarNet/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/qMRI/evaluation_per_slice/AHEAD_gaussian2d_12x_Test/qMRI --fill_target_path --fill_pred_path
