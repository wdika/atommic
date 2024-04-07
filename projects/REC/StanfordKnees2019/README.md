## **Stanford Fullysampled 3D FSE Knees 2019 Dataset**

This project folder contains the configuration files, preprocessing, and visualization scripts for the Stanford
Fullysampled 3D FSE Knees 2019 dataset.

For more information, please refer to http://mridata.org/list?project=Stanford%20Fullysampled%203D%20FSE%20Knees.

**Note:** When running the preprocessing scripts please make sure you have the ``ismrmrd`` package installed. You
can install it with the following command:
```bash
pip install -r requirements/requirements-ahead_stanfordknees.txt
```

### **Visualization**
An example notebook for visualizing the data is provided in the
[visualize.ipynb](visualize.ipynb). You just need to set the path where the
dataset is downloaded.

### **Preprocessing**
The preprocessing pipeline is implemented in the
[preprocess_dataset.sh](preprocess_dataset.sh) script, consisting of the
following steps:
1. Convert the data from ISMRMRD to HDF5 format.
2. Split the dataset into training and validation sets.

The preprocessing script can be run with the following command:
```bash
bash ./projects/REC/StanfordKnees2019/preprocess_dataset.sh
```

### **Training/Testing**
For training a model, you just need to set up the data and export paths to the configuration file in
/projects/REC/StanfordKnees2019/conf/train/ of the model you want to train. In `train_ds` and
`validation_ds` please set the `data_path` to the generated json files. In `exp_manager` please set the `exp_dir` to
the path where you want to save the model checkpoints and tensorboard or wandb logs.

You can train a model with the following command:
`atommic run -c /projects/REC/StanfordKnees2019/conf/train/{model}.yaml`

For testing a model, you just need to set up the data and export paths to the configuration file in
/projects/REC/StanfordKnees2019/conf/test/ of the model you want to test. In `checkpoint`
(line 2) set the path the trained model checkpoint and in `test_ds` please set the `data_path`. In `exp_manager` please
set the `exp_dir` to the path where the predictions and logs will be saved.

You can test a model with the following command:
`atommic run -c /projects/REC/StanfordKnees2019/conf/test/{model}.yaml`

**Note:** The default logger is tensorboard.
