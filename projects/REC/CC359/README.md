## **Calgary-Campinas Public Brain MR Dataset (CC359)**

This project folder contains the configuration files, preprocessing, and visualization scripts for the
Calgary-Campinas Public Brain MR Dataset (CC359).

For more information, please refer to https://sites.google.com/view/calgary-campinas-dataset/home

The dataset contains 3D T1-weighted raw data.

### Dataset Folder Structure
The information below describes the folder structure of the dataset and is copied from the dataset website.

```console
CC359/
└── Raw-data
          ├── Multi-channel
                    ├── 12-channel
                    │         ├── test_12_channel.zip -> Undersampled 12-channel test set for R = 5 and R = 10
                    │         └── train_val_12_channel.zip -> Fully sampled 12-channel train and validation data
                    └── 32-channel
                        └── test_32_channel.zip -> Undersampled 32-channel test set for R = 5 and R = 10
```

### Raw Multicoil Data
The information below describes the raw multicoil data and is copied from the dataset website.

We are providing 167 three-dimensional (3D), T1-weighted, gradient-recalled echo, 1 mm isotropic sagittal acquisitions
collected on a clinical MR scanner (Discovery MR750; General Electric (GE) Healthcare, Waukesha, WI). The scans
correspond to presumed healthy subjects (age: 44.5 years +/- 15.5 years [mean +/- standard deviation]; range: 20 years
to 80 years). Datasets were acquired using either a 12-channel (117 scans) or a 32-channel coil (50 scans).
Acquisition parameters were TR/TE/TI = 6.3 ms/ 2.6 ms/ 650 ms (93 scans) and TR/TE/TI = 7.4 ms/ 3.1 ms/ 400 ms
(74 scans), with 170 to 180 contiguous 1.0-mm slices and a field of view of 256 mm x 218 mm. The acquisition matrix
size for each channel was Nx x Ny x Nz = 256 x 218 x [170,180]. In the slice-encoded direction (kz), data were
partially collected up to 85% of its matrix size and then zero filled to Nz= [170,180]. The scanner automatically
applied the inverse Fourier Transform, using the fast Fourier transform (FFT) algorithms, to the kx-ky-kz k-space data
in the frequency-encoded direction, so a hybrid x-ky-kz dataset was saved. This reduces the problem from 3D to
two-dimensions, while still allowing to undersample k-space in the phase encoding and slice encoding directions. The
partial Fourier reference data were reconstructed by taking the channel-wise iFFT of the collected k-spaces for each
slice of the 3D volume and combining the outputs through the conventional sum of squares algorithm. The dataset
train/validation/test split is summarized in the table below .Relevant information

- Healthy subjects (age: 44.5 years ± 15.5 years; range: 20 years to 80 years).
- Acquisition parameters are either: TR/TE/TI = 6.3 ms/2.6 ms/650 ms and TR/TE/TI = 7.4 ms/3.1 ms/400 ms
- Average scan duration ~341 seconds
- Only the undersampled k-spaces for R=5 and R=10 are provided for the test set

Dataset summary:
- 12-channel
 - Train: 47 dataset
 - Validation: 20 datasets
 - Test: 50 datasets
- 32-channel
 - Test: 50 datasets

### **Visualization**
An example notebook for visualizing the data is provided in the
[visualize.ipynb](visualize.ipynb). You just need to set the path where the
dataset is downloaded. The
[original notebook](https://github.com/rmsouza01/MC-MRI-Rec/blob/master/JNotebooks/getting-started/getting_started.ipynb)
is copied from the (https://github.com/rmsouza01/MC-MRI-Rec repository and provided by the CC359 dataset authors.

### **Preprocessing**
The CC359 dataset is supported natively in ``ATOMMIC`` and no preprocessing is required. You just need to convert the
.npy masks to .h5 format if you want to undersampled the data with the provided masks. The conversion script is
implemented in the [compute_masks.sh](compute_masks.sh) script.

```bash
bash ./projects/REC/CC359/compute_masks.sh
```

### **Training/Testing**
For training a model, you just need to set up the data and export paths to the configuration file in
/projects/REC/CC359/conf/train/ of the model you want to train. In `train_ds` and
`validation_ds` please set the `data_path` to the generated json files. In `exp_manager` please set the `exp_dir` to
the path where you want to save the model checkpoints and tensorboard or wandb logs.

You can train a model with the following command:
`atommic run -c /projects/REC/CC359/conf/train/{model}.yaml`

For testing a model, you just need to set up the data and export paths to the configuration file in
/projects/REC/CC359/conf/test/ of the model you want to test. In `checkpoint`
(line 2) set the path the trained model checkpoint and in `test_ds` please set the `data_path`. In `exp_manager` please
set the `exp_dir` to the path where the predictions and logs will be saved.

You can test a model with the following command:
`atommic run -c /projects/REC/CC359/conf/test/{model}.yaml`

**Note:** The default logger is tensorboard.
