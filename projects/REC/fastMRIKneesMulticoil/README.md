## **fastMRI Knees Multicoil Dataset**

This project folder contains the configuration files and visualization scripts for the fastMRI Knees Multicoil
dataset.

For more information, please refer to https://fastmri.med.nyu.edu/.

### **Visualization**
An example notebook for visualizing the data is provided in the
[visualize.ipynb](visualize.ipynb). You just need to set the path where
the dataset is downloaded.

### **Preprocessing**
The fastMRI datasets are supported natively in ``ATOMMIC`` and no preprocessing is required.

### **Training/Testing**
For training a model, you just need to set up the data and export paths to the configuration file in
/projects/REC/fastMRIKneesMulticoil/conf/train/ of the model you want to train. In `train_ds` and
`validation_ds` please set the `data_path` to the generated json files. In `exp_manager` please set the `exp_dir` to
the path where you want to save the model checkpoints and tensorboard or wandb logs.

You can train a model with the following command:
`atommic run -c /projects/REC/fastMRIKneesMulticoil/conf/train/{model}.yaml`

For testing a model, you just need to set up the data and export paths to the configuration file in
/projects/REC/fastMRIKneesMulticoil/conf/test/ of the model you want to test. In `checkpoint`
(line 2) set the path the trained model checkpoint and in `test_ds` please set the `data_path`. In `exp_manager` please
set the `exp_dir` to the path where the predictions and logs will be saved.

You can test a model with the following command:
`atommic run -c /projects/REC/fastMRIKneesMulticoil/conf/test/{model}.yaml`

**Note:** The default logger is tensorboard.
