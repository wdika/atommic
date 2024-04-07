BraTS 2023 Adult Glioma
=======================

This project folder contains the configuration files, preprocessing, and visualization scripts for the
BraTS2023AdultGlioma dataset.

For more information, please refer to https://www.synapse.org/#!Synapse:syn51156910/wiki/.

Related papers:

* https://arxiv.org/pdf/1811.02629.pdf,
* https://arxiv.org/pdf/2305.17033.pdf.

Data need to be downloaded manually due to required registration. Download link:
https://www.synapse.org/#!Synapse:syn51156910/wiki/622351.

.. note::
    When running the preprocessing scripts please make sure you have the following packages installed: argparse, json,
    nibabel, numpy, pathlib, random, tqdm. Those packages are installed by default if atommic is installed.

**Visualization**
An example notebook for visualizing the data is provided in the
`visualize <https://github.com/wdika/atommic/tree/main/projects/SEG/BraTS2023AdultGlioma/visualize.ipynb>`_
notebook. You just need to set the path where the dataset is downloaded.

**Preprocessing**
The preprocessing pipeline is implemented in the
`preprocess_dataset.sh <https://github.com/wdika/atommic/tree/main/projects/SEG/BraTS2023AdultGlioma/preprocess_dataset.sh>`_
script, consisting of the following steps:
1. Crop to the brain region, as there is a lot of background around the brain resulting is slower training.
Important note: the cropping is done only for the training set.
2. Normalize the images to zero mean and unit variance.
3. Updates headers and save to NIfTI format.
4. Split the dataset into training and validation sets.
5. Compute the probabilities for each segmentation class.

**Training/Testing**

.. important::
    The ``BraTS2023AdultGlioma`` dataset is natively supported in ``atommic``. Therefore, you do not need to create a
    custom dataset class. You just need to set the ``dataset_format`` argument in the configuration file to
    ``BraTS2023AdultGlioma``. For example:

    .. code-block:: bash

        train_ds:
            dataset_format: BraTS2023AdultGlioma

        validation_ds:
            dataset_format: BraTS2023AdultGlioma

        test_ds:
            dataset_format: BraTS2023AdultGlioma

For training a model, you just need to set up the data and export paths to the
`configuration <https://github.com/wdika/atommic/tree/main/projects/SEG/BraTS2023AdultGlioma/conf/train/>`_
file of the model you want to train. In ``train_ds`` and `validation_ds` please set the ``data_path`` to the generated
json files. In ``exp_manager`` please set the ``exp_dir`` to the path where you want to save the model checkpoints and
tensorboard or wandb logs.

You can train a model with the following command:

.. code-block:: bash

    atommic run -c /projects/SEG/BraTS2023AdultGlioma/conf/train/{model}.yaml

For testing a model, you just need to set up the data and export paths to the
`configuration <https://github.com/wdika/atommic/tree/main/projects/SEG/BraTS2023AdultGlioma/conf/test/>`_ file
model you want to test. In ``checkpoint`` (line 2) set the path the trained model checkpoint and in ``test_ds`` please
set the ``data_path``. In ``exp_manager`` please set the ``exp_dir`` to the path where the predictions and logs will
be saved.

You can test a model with the following command:

.. code-block:: bash

    atommic run -c /projects/SEG/BraTS2023AdultGlioma/conf/test/{model}.yaml

**Note:** The default logger is tensorboard.
