python tools/evaluation/reconstruction.py \
data_parent_dir/PD/multicoil_val \
output_dir/atommic/reconstruction/predictions/fastMRIKnees_multicoil_PD/ATOMMIC/VarNet/default/2023-11-10_09-56-29/reconstructions \
--evaluation_type per_slice --output_dir output_dir/atommic/reconstruction/evaluation_per_slice/fastMRIKnees_multicoil_PD/ --crop_size 320 320

python tools/evaluation/reconstruction.py \
data_parent_dir/PD/multicoil_val \
output_dir/atommic/reconstruction/predictions/fastMRIKnees_multicoil_PD/DIRECT \
--evaluation_type per_slice --output_dir output_dir/atommic/reconstruction/evaluation_per_slice/fastMRIKnees_multicoil_PD/ --crop_size 320 320

python tools/evaluation/reconstruction.py \
data_parent_dir/PD/multicoil_val \
output_dir/atommic/reconstruction/predictions/fastMRIKnees_multicoil_PD/fastMRI/varnet/reconstructions \
--evaluation_type per_slice --output_dir output_dir/atommic/reconstruction/evaluation_per_slice/fastMRIKnees_multicoil_PD/ --crop_size 320 320
