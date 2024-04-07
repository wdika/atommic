python projects/MTL/rs/SKMTEA/evaluation/mtlrs_reconstruction.py \
output_data_dir/atommic/MTL/targets/SKMTEA_Test/SENSE/default/ output_data_dir/atommic/MTL/predictions/SKMTEA/IDSLR_SENSE/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/MTL/evaluation_per_slice/SKMTEA/reconstruction/ --fill_target_path --fill_pred_path
python projects/MTL/rs/SKMTEA/evaluation/mtlrs_reconstruction.py \
output_data_dir/atommic/MTL/targets/SKMTEA_Test/SENSE/default/ output_data_dir/atommic/MTL/predictions/SKMTEA/IDSLRUNET_SENSE/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/MTL/evaluation_per_slice/SKMTEA/reconstruction/ --fill_target_path --fill_pred_path
python projects/MTL/rs/SKMTEA/evaluation/mtlrs_reconstruction.py \
output_data_dir/atommic/MTL/targets/SKMTEA_Test/SENSE/default/ output_data_dir/atommic/MTL/predictions/SKMTEA/SegNet_SENSE/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/MTL/evaluation_per_slice/SKMTEA/reconstruction/ --fill_target_path --fill_pred_path
python projects/MTL/rs/SKMTEA/evaluation/mtlrs_reconstruction.py \
output_data_dir/atommic/MTL/targets/SKMTEA_Test/SENSE/default/ output_data_dir/atommic/MTL/predictions/SKMTEA/MTLRS_SENSE/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/MTL/evaluation_per_slice/SKMTEA/reconstruction/ --fill_target_path --fill_pred_path
python projects/MTL/rs/SKMTEA/evaluation/mtlrs_segmentation.py \
parent_data_dir/skm-tea/v1-release/json/files_recon_calib-24_test.json \
parent_data_dir/skm-tea/v1-release/segmentation_masks/raw-data-track \
output_data_dir/atommic/MTL/predictions/SKMTEA/IDSLR_SENSE/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/MTL/evaluation_per_slice/SKMTEA/segmentation/ --fill_pred_path
python projects/MTL/rs/SKMTEA/evaluation/mtlrs_segmentation.py \
parent_data_dir/skm-tea/v1-release/json/files_recon_calib-24_test.json \
parent_data_dir/skm-tea/v1-release/segmentation_masks/raw-data-track \
output_data_dir/atommic/MTL/predictions/SKMTEA/IDSLRUNET_SENSE/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/MTL/evaluation_per_slice/SKMTEA/segmentation/ --fill_pred_path
python projects/MTL/rs/SKMTEA/evaluation/mtlrs_segmentation.py \
parent_data_dir/skm-tea/v1-release/json/files_recon_calib-24_test.json \
parent_data_dir/skm-tea/v1-release/segmentation_masks/raw-data-track \
output_data_dir/atommic/MTL/predictions/SKMTEA/SegNet_SENSE/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/MTL/evaluation_per_slice/SKMTEA/segmentation/ --fill_pred_path
python projects/MTL/rs/SKMTEA/evaluation/mtlrs_segmentation.py \
parent_data_dir/skm-tea/v1-release/json/files_recon_calib-24_test.json \
parent_data_dir/skm-tea/v1-release/segmentation_masks/raw-data-track \
output_data_dir/atommic/MTL/predictions/SKMTEA/MTLRS_SENSE/default/ \
--evaluation_type per_slice --output_dir output_data_dir/atommic/MTL/evaluation_per_slice/SKMTEA/segmentation/ --fill_pred_path
