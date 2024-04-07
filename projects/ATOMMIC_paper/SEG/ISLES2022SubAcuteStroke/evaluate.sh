python projects/SEG/ISLES2022SubAcuteStroke/scripts/evaluation.py \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/folds/fold_0_test.json \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/data/ \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/segmentations/ \
output_data_dir/atommic/SEG/predictions/ISLES2022SubAcuteStroke/AttentionUNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/ISLES2022SubAcuteStroke/ \
--evaluation_type per_slice --fill_pred_path
python projects/SEG/ISLES2022SubAcuteStroke/scripts/evaluation.py \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/folds/fold_0_test.json \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/data/ \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/segmentations/ \
output_data_dir/atommic/SEG/predictions/ISLES2022SubAcuteStroke/DynUNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/ISLES2022SubAcuteStroke/ \
--evaluation_type per_slice --fill_pred_path
python projects/SEG/ISLES2022SubAcuteStroke/scripts/evaluation.py \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/folds/fold_0_test.json \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/data/ \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/segmentations/ \
output_data_dir/atommic/SEG/predictions/ISLES2022SubAcuteStroke/UNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/ISLES2022SubAcuteStroke/ \
--evaluation_type per_slice --fill_pred_path
python projects/SEG/ISLES2022SubAcuteStroke/scripts/evaluation.py \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/folds/fold_0_test.json \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/data/ \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/segmentations/ \
output_data_dir/atommic/SEG/predictions/ISLES2022SubAcuteStroke/UNet3D/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/ISLES2022SubAcuteStroke/ \
--evaluation_type per_slice --fill_pred_path
python projects/SEG/ISLES2022SubAcuteStroke/scripts/evaluation.py \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/folds/fold_0_test.json \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/data/ \
parent_data_dir/ISLES2022SubAcuteStroke/preprocessed/segmentations/ \
output_data_dir/atommic/SEG/predictions/ISLES2022SubAcuteStroke/VNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/ISLES2022SubAcuteStroke/ \
--evaluation_type per_slice --fill_pred_path
