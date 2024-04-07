python tools/evaluation/segmentation.py parent_data_dir/BraTS2023AdultGlioma/preprocessed/folds/fold_0_val.json \
output_data_dir/atommic/SEG/predictions/BraTs23AdultGlioma/AttentionUNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/BraTS2023AdultGlioma/ \
--dataset_format brats --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
python tools/evaluation/segmentation.py parent_data_dir/BraTS2023AdultGlioma/preprocessed/folds/fold_0_val.json \
output_data_dir/atommic/SEG/predictions/BraTs23AdultGlioma/DynUNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/BraTS2023AdultGlioma/ \
--dataset_format brats --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
python tools/evaluation/segmentation.py parent_data_dir/BraTS2023AdultGlioma/preprocessed/folds/fold_0_val.json \
output_data_dir/atommic/SEG/predictions/BraTs23AdultGlioma/UNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/BraTS2023AdultGlioma/ \
--dataset_format brats --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
python tools/evaluation/segmentation.py parent_data_dir/BraTS2023AdultGlioma/preprocessed/folds/fold_0_val.json \
output_data_dir/atommic/SEG/predictions/BraTs23AdultGlioma/UNet3D/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/BraTS2023AdultGlioma/ \
--dataset_format brats --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
python tools/evaluation/segmentation.py parent_data_dir/BraTS2023AdultGlioma/preprocessed/folds/fold_0_val.json \
output_data_dir/atommic/SEG/predictions/BraTs23AdultGlioma/VNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/BraTS2023AdultGlioma/ \
--dataset_format brats --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
