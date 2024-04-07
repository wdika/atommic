## **Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) 2021 Dataset**

This project folder contains the configuration files, preprocessing, and visualization scripts for the
Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) 2021 dataset.

For more information, please refer to https://github.com/StanfordMIMI/skm-tea.

Related papers:
- https://openreview.net/forum?id=YDMFgD_qJuA.

### **Visualization**
An example notebook for visualizing the data is provided in the
[visualize.ipynb](visualize.ipynb). You just need to set the path where the
dataset is downloaded. The
[original notebook](https://colab.research.google.com/drive/1PluqK77pobD5dXE7zzBLEAeBgaaeGKXa) is copied from the
https://github.com/StanfordMIMI/skm-tea repository and provided by the SKMTEA authors.

### **Preprocessing**
No preprocessing is needed for the SKMTEA dataset. You just need to generate train, val, and test sets depending on
the task you use the dataset for. For example, for the segmentation task, you need to run the
[generate_sets.sh](generate_sets.sh) script.

The preprocessing script can be run with the following command:
```bash
bash ./projects/SEG/SKMTEA/preprocess_dataset.sh
```

### **Training/Testing**
For training a model, you just need to set up the data and export paths to the configuration file in
/projects/SEG/SKMTEA/conf/train/ of the model you want to train. In `train_ds` and
`validation_ds` please set the `data_path` to the generated json files. In `exp_manager` please set the `exp_dir` to
the path where you want to save the model checkpoints and tensorboard or wandb logs.

You can train a model with the following command:
`atommic run -c /projects/SEG/SKMTEA/conf/train/{model}.yaml`

For testing a model, you just need to set up the data and export paths to the configuration file in
/projects/SEG/SKMTEA/conf/test/ of the model you want to test. In `checkpoint`
(line 2) set the path the trained model checkpoint and in `test_ds` please set the `data_path`. In `exp_manager` please
set the `exp_dir` to the path where the predictions and logs will be saved.

You can test a model with the following command:
`atommic run -c /projects/SEG/SKMTEA/conf/test/{model}.yaml`

**Note:** The default logger is tensorboard.
