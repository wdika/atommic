Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) 2021 Dataset
===============================================================

This project folder contains the configuration files, preprocessing, and visualization scripts for the Stanford Knee
MRI Multi-Task Evaluation (SKM-TEA) 2021 dataset.

Related papers:

* https://openreview.net/forum?id=YDMFgD_qJuA.

**Visualization**
An example notebook for visualizing the data is provided in the
`visualize <https://github.com/wdika/atommic/tree/main/projects/REC/SKMTEA/visualize.ipynb>`_ notebook. You
just need to set the path where the dataset is downloaded. The
`original notebook <https://colab.research.google.com/drive/1PluqK77pobD5dXE7zzBLEAeBgaaeGKXa>`_ is copied from the
https://github.com/StanfordMIMI/skm-tea repository and provided by the SKMTEA authors.

**Preprocessing**
No preprocessing is needed for the SKMTEA dataset. You just need to generate train, val, and test sets depending on
the task you use the dataset for. For example, for the reconstruction task, you need to run the
`generate_sets.sh <https://github.com/wdika/atommic/tree/main/projects/REC/SKMTEA/generate_sets.sh>`_
script.

**Training/Testing**

.. important::
    The ``SKM-TEA`` dataset is natively supported in ``atommic``. Therefore, you do not need to create a custom
    dataset class. You just need to set the ``dataset_format`` argument in the configuration file to the desired
    ``SKM-TEA`` dataset version. Also the FFT needs to be centered. For example:

    .. code-block:: bash

        model:
            fft_centered: true
            fft_normalization: ortho

            train_ds:
                dataset_format: skm-tea-echo1
                fft_centered: true
                fft_normalization: ortho

            validation_ds:
                dataset_format: skm-tea-echo1+echo2
                fft_centered: true
                fft_normalization: ortho

            test_ds:
                dataset_format: skm-tea-echo1+echo2-mc
                fft_centered: true
                fft_normalization: ortho

The ``skm-tea-echo1`` dataset contains only the first echo of the multi-echo data. The ``skm-tea-echo2`` dataset
contains only the second echo of the multi-echo data. The ``skm-tea-echo1+echo2`` dataset sums the first and second
echoes of the multi-echo data. The ``skm-tea-echo1+echo2-mc`` dataset stacks the first and second echoes of the
multi-echo data as channels.

For training a model, you just need to set up the data and export paths to the
`configuration <https://github.com/wdika/atommic/tree/main/projects/REC/SKMTEA/conf/train/>`_ file of the
model you want to train. In ``train_ds`` and `validation_ds` please set the ``data_path`` to the generated json files.
In ``exp_manager`` please set the ``exp_dir`` to the path where you want to save the model checkpoints and tensorboard
or wandb logs.

You can train a model with the following command:

.. code-block:: bash

    atommic run -c /projects/REC/SKMTEA/conf/train/{model}.yaml

For testing a model, you just need to set up the data and export paths to the
`configuration <https://github.com/wdika/atommic/tree/main/projects/REC/SKMTEA/conf/train/>`_ file of the
model you want to test. In ``checkpoint`` (line 2) set the path the trained model checkpoint and in ``test_ds`` please
set the ``data_path``. In ``exp_manager`` please set the ``exp_dir`` to the path where the predictions and logs will
be saved.

You can test a model with the following command:

.. code-block:: bash

    atommic run -c /projects/REC/SKMTEA/conf/test/{model}.yaml

**Note:** The default logger is tensorboard.
