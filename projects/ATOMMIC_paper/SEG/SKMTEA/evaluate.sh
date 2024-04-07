python tools/evaluation/segmentation.py \
parent_data_dir/skm-tea/v1-release/json/image_files_test.json \
output_data_dir/atommic/SEG/predictions/SKMTEA/AttentionUNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/SKMTEA/ \
--dataset_format skm-tea --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
python tools/evaluation/segmentation.py \
parent_data_dir/skm-tea/v1-release/json/image_files_test.json \
output_data_dir/atommic/SEG/predictions/SKMTEA/DynUNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/SKMTEA/ \
--dataset_format skm-tea --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
python tools/evaluation/segmentation.py \
parent_data_dir/skm-tea/v1-release/json/image_files_test.json \
output_data_dir/atommic/SEG/predictions/SKMTEA/UNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/SKMTEA/ \
--dataset_format skm-tea --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
python tools/evaluation/segmentation.py \
parent_data_dir/skm-tea/v1-release/json/image_files_test.json \
output_data_dir/atommic/SEG/predictions/SKMTEA/UNet3D/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/SKMTEA/ \
--dataset_format skm-tea --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
python tools/evaluation/segmentation.py \
parent_data_dir/skm-tea/v1-release/json/image_files_test.json \
output_data_dir/atommic/SEG/predictions/SKMTEA/VNet/default/ \
--output_dir output_data_dir/atommic/SEG/evaluation_per_slice/SKMTEA/ \
--dataset_format skm-tea --evaluation_type per_slice --fill_pred_path --sum_classes_method argmax
