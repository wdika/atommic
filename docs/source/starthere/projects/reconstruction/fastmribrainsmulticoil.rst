fastMRI Brains Multicoil
=========================


**Training/Testing**

.. important::
    The ``fastMRI`` datasets are natively supported in ``atommic``. Therefore, you do not need to create a custom
    dataset  class. You just need to set the ``dataset_format`` argument in the configuration file to ``fastMRI``.
    Also the FFT needs to be centered. For example:

    .. code-block:: bash

        model:
            fft_centered: true
            fft_normalization: ortho

        train_ds:
            dataset_format: fastMRI
            fft_centered: true
            fft_normalization: ortho

        validation_ds:
            dataset_format: fastMRI
            fft_centered: true
            fft_normalization: ortho

        test_ds:
            dataset_format: fastMRI
            fft_centered: true
            fft_normalization: ortho

For training a model, you just need to set up the data and export paths to the
`configuration <https://github.com/wdika/atommic/tree/main/projects/REC/fastMRIBrainsMulticoil/conf/train/>`_
file of the model you want to train. In ``train_ds`` and `validation_ds` please set the ``data_path`` to the generated
json files. In ``exp_manager`` please set the ``exp_dir`` to the path where you want to save the model checkpoints and
tensorboard or wandb logs.

You can train a model with the following command:

.. code-block:: bash

    atommic run -c /projects/REC/fastMRIBrainsMulticoil/conf/train/{model}.yaml

For testing a model, you just need to set up the data and export paths to the
`configuration <https://github.com/wdika/atommic/tree/main/projects/REC/fastMRIBrainsMulticoil/conf/test/>`_
file model you want to test. In ``checkpoint`` (line 2) set the path the trained model checkpoint and in ``test_ds``
please set the ``data_path``. In ``exp_manager`` please set the ``exp_dir`` to the path where the predictions and logs
will be saved.

You can test a model with the following command:

.. code-block:: bash

    atommic run -c /projects/REC/fastMRIBrainsMulticoil/conf/test/{model}.yaml

**Note:** The default logger is tensorboard.
