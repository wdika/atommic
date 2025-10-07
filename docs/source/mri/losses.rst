Losses
======

``ATOMMIC`` provides a number of loss functions for training models. These are all subclasses of ``torch.nn.Module``
and can be used in the same way as any other PyTorch loss function.

For ``reconstruction``,  ``qMRI`` and ``multitask`` tasks, the following losses are available:

* :class:`~MSELoss`:
    A loss function based on the Mean Squared Error (MSE). It can be used for any task and it calls
    ``torch.nn.MSELoss``.

* :class:`~L1Loss`:
    A loss function based on the Mean Absolute Error (MAE). It can be used for any task and it calls
    ``torch.nn.L1Loss``.

* :class:`~atommic.collections.reconstruction.losses.SSIMLoss`:
    A loss function based on the Structural Similarity Index (SSIM). It can be used for any task and it is based on
    [Wang2004]_.

    References
    ----------
    .. [Wang2004] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from
        error visibility to structural similarity. IEEE transactions on image processing, 13(4), 600-612.

* :class:`~atommic.collections.reconstruction.losses.HaarPSILoss`:
    A loss function based on the Haar Wavelet-Based Perceptual Similarity (HaarPSI). It can be used for any task and it is based on
    [Reisenhofer2018]_. Parameters are optimised for medical images based on [Karner2024]_.

    References
    ----------
    .. [Reisenhofer2018] Reisenhofer, R., Bosse, S., Kutyniok, G., & Wiegand, T. (2018). A Haar wavelet-based perceptual similarity index for image 
        quality assessment. Signal Processing: Image Communication, 61, 33-43.
        
    .. [Karner2024] Karner, C., Gröhl, J., Selby, I., Babar, J., Beckford, J., Else, T. R., ... & Breger, A. (2024). Parameter choices in HaarPSI 
        for IQA with medical images. arXiv preprint arXiv:2410.24098.


* :class:`~atommic.collections.reconstruction.losses.NoiseAwareLoss`:
    A custom loss function that is aware of the noise level in the data. It can be used for any task and it is based
    on [Oh2021]_.

    References
    ----------
    .. [Oh2021] Oh, Y., Kim, B., & Ham, B. (2021). Background-aware pooling and noise-aware loss for
        weakly-supervised semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and
        pattern recognition (pp. 6913-6922).

* :class:`~atommic.collections.common.losses.SinkhornDistance`:
    Resembles the Wasserstein distance, but is differentiable and can be used as a loss function. It can be used for
    any task and it is based on [Cuturi2013]_.

    References
    ----------
    .. [Cuturi2013] Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport, NIPS 2013

* :class:`~atommic.collections.segmentation.losses.CrossEntropyLoss`:
    A loss function based on the cross-entropy between the predicted and the ground truth segmentation. It can be used
    for segmentation tasks and it is a wrapper around ``torch.nn.CrossEntropyLoss``.

* :class:`~atommic.collections.segmentation.losses.Dice`:
    A loss function based on the Dice coefficient. It can be used for segmentation tasks and it is a wrapper for
    :py:class:`monai.losses.DiceLoss` to support multi-class and multi-label tasks. It is based on [Milletari2016]_.

    References
    ----------
    .. [Milletari2016] Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). V-net: Fully convolutional
        neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D
        vision (3DV) (pp. 565-571). IEEE.

:class:`~atommic.collections.common.losses.AggregatorLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``AggregatorLoss`` class is used to combine multiple losses into a single loss function.

.. note::
    The ``AggregatorLoss`` is not a loss function itself, but a wrapper around multiple loss functions. It is used to
    combine multiple losses into a single loss function. The ``AggregatorLoss`` is used by the ``ATOMMIC`` models to
    combine the losses by setting a weight for each loss function. The weights must sum to 1.0.

``AggregatorLoss`` is configurable via YAML with Hydra. For example:

.. code-block:: bash

    model:
        reconstruction_loss:
            mse: 0.2
            l1: 0.2
            ssim: 0.2
            noise_aware: 0.2
            wasserstein: 0.2

This will create a loss function for the ``reconstruction`` task that is a weighted sum of the MSE, MAE, SSIM,
NoiseAware and Wasserstein losses.

.. code-block:: bash

    model:
        segmentation_loss:
            cross_entropy: 0.5
            dice: 0.5

This will create a loss function for the ``segmentation`` task that is a weighted sum of the CrossEntropy and Dice
losses.

.. code-block:: bash

    model:
        reconstruction_loss:
            mse: 0.2
            l1: 0.2
            ssim: 0.2
            noise_aware: 0.2
            wasserstein: 0.2
        segmentation_loss:
            cross_entropy: 0.5
            dice: 0.5
        total_reconstruction_loss_weight: 0.5
        total_segmentation_loss_weight: 0.5

This will create a loss function for the ``multitask`` task that is a weighted sum of the reconstruction and the
segmentation losses. The weights for the reconstruction and segmentation losses are set by the
``total_reconstruction_loss_weight`` and ``total_segmentation_loss_weight`` parameters, respectively.

.. code-block:: bash

    model:
        quantitative_loss:
            mse: 0.2
            l1: 0.2
            ssim: 0.2
            noise_aware: 0.2
            wasserstein: 0.2

This will create a loss function for the ``qMRI`` task that is a weighted sum of the MSE, MAE, SSIM, NoiseAware and
Wasserstein losses.
