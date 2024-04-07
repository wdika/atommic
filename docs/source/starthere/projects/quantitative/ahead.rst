AHEAD
=====

This project folder contains the configuration files, preprocessing, and visualization scripts for the Amsterdam
Ultra-high field adult lifespan database (AHEAD) dataset.

Data were scanned using the MP2RAGEME sequence for T1, T2* and Quantitative Susceptibility Mapping in one sequence at 7 Tesla.
Data are motion-corrected using Fat navigators (FatNavs), and defaced in image-domain. In total 77 subjects are
included, scanned with a resolution of 0.7mm isotropic. Data of the MP2RAGEME-sequence are stored according to the
ISMRMRD-standard in h5-format (https://ismrmrd.github.io/). Detailed scanner parameters are included in the h5-files
of all subjects. Coil sensitivity maps per subjects are included in native h5-format. Demographics of all subjects are
included in a separate csv-file, being sex and age decade, covering the life span.

For more information and dataset download link for the AHEAD project, please check
https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/IHZGQM.

.. note::
    When running the preprocessing scripts please make sure you have the ``ismrmrd`` package installed. You can
    install it with the following command:

    .. code-block:: bash

        pip install -r requirements/requirements-ahead_stanfordknees.txt

**Visualization**
An example notebook for visualizing and preprocessing the data is provided in the
`visualize <https://github.com/wdika/atommic/tree/main/projects/qMRI/AHEAD/getting-started.ipynb>`_. You just
need to set the path where the dataset is downloaded.

**Preprocessing**
The preprocessing pipeline is implemented in the
`batch_preprocessing.sh <https://github.com/wdika/atommic/tree/main/projects/qMRI/AHEAD/batch_preprocessing.sh>`_
script, consisting of the following steps:
1. Read the raw data in ``ISMRMRD`` format.
2. Preprocess the coil sensitivity maps.
3. Compute the imspace and ground-truth target data.
4. Compute the masks.
5. Compute the quantitative maps.
6. Store the data in ``HDF5`` format.

The preprocessing script can be run with the following command:

.. code-block:: bash

    bash projects/qMRI/AHEAD/batch_preprocessing.sh

**Training/Testing**
For training a model, you just need to set up the data and export paths to the
`configuration <https://github.com/wdika/atommic/tree/main/qMRI/AHEAD/conf/>`_ file of the model you want
to train. In `train_ds` and `validation_ds` please set the `data_path` to the generated json files. In `exp_manager`
please set the `exp_dir` to the path where you want to save the model checkpoints and tensorboard or wandb logs.

.. code-block:: bash

    atommic run -c /projects/qMRI/AHEAD/conf/train/{model}.yaml`

For testing a model, you just need to set up the data and export paths to the
`configuration <https://github.com/wdika/atommic/tree/main/qMRI/AHEAD/conf/>`_ file of the model you want
to test. In `checkpoint` (line 2) set the path the trained model checkpoint and in `test_ds` please set the `data_path`.
In `exp_manager` please set the `exp_dir` to the path where the predictions and logs will be saved.

You can test a model with the following command:

.. code-block:: bash

    atommic run -c /projects/qMRI/AHEAD/conf/test/{model}.yaml`

**Note:** The default logger is tensorboard.
