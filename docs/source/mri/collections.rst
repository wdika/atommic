Collections & Models
====================

``ATOMMIC`` is organized in collections, each of which implements a specific task. The following collections are
currently available, implementing various models as listed.


MultiTask Learning (MTL)
------------------------

End-to-End Recurrent Attention Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-End Recurrent Attention Network (:class:`~atommic.collections.multitask.rs.nn.seranet.SERANet`), as
presented in [Huang2019]_.

    References
    ----------
    .. [Huang2019] Huang, Q., Chen, X., Metaxas, D., Nadar, M.S. (2019). Brain Segmentation from k-Space with
       End-to-End Recurrent Attention Network. In: , et al. Medical Image Computing and Computer Assisted
       Intervention – MICCAI 2019. Lecture Notes in Computer Science(), vol 11766. Springer, Cham.
       https://doi.org/10.1007/978-3-030-32248-9_31


Example configuration:

.. code-block:: bash

    model:
        model_name: SERANET
        use_reconstruction_module: true
        input_channels: 2
        reconstruction_module: unet
        reconstruction_module_output_channels: 2
        reconstruction_module_channels: 32
        reconstruction_module_pooling_layers: 4
        reconstruction_module_dropout: 0.0
        # or
        #  reconstruction_module: cascadenet
        #  reconstruction_module_hidden_channels: 32
        #  reconstruction_module_n_convs: 2
        #  reconstruction_module_batchnorm: true
        #  reconstruction_module_num_cascades: 5
        reconstruction_module_num_blocks: 3
        segmentation_module_input_channels: 32
        segmentation_module_output_channels: 2
        segmentation_module_channels: 32
        segmentation_module_pooling_layers: 4
        segmentation_module_dropout: 0.0
        recurrent_module_iterations: 2
        recurrent_module_attention_channels: 32
        recurrent_module_attention_pooling_layers: 4
        recurrent_module_attention_dropout: 0.0
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        complex_data: true
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Image domain Deep Structured Low-Rank Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Image domain Deep Structured Low-Rank Network (:class:`~atommic.collections.multitask.rs.nn.idslr.IDSLR`), as
presented in [Pramanik2021]_.

    References
    ----------
    .. [Pramanik2021] Pramanik A, Wu X, Jacob M. Joint calibrationless reconstruction and segmentation of parallel
        MRI. arXiv preprint arXiv:2105.09220. 2021 May 19.


Example configuration:

.. code-block:: bash

    model:
        model_name: IDSLR
        use_reconstruction_module: true
        input_channels: 24  # coils * 2
        reconstruction_module_output_channels: 24  # coils * 2
        segmentation_module_output_channels: 2
        channels: 64
        num_pools: 2
        padding_size: 11
        drop_prob: 0.0
        normalize: true
        padding: true
        norm_groups: 2
        num_iters: 5
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        complex_data: true
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Image domain Deep Structured Low-Rank UNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Image domain Deep Structured Low-Rank network using a UNet (and not only the decoder part) as segmentation model
(:class:`~atommic.collections.multitask.rs.nn.idslr_unet.IDSLRUNet`), as presented in [Pramanik2021]_.

    References
    ----------
    .. [Pramanik2021] Pramanik A, Wu X, Jacob M. Joint calibrationless reconstruction and segmentation of parallel
        MRI. arXiv preprint arXiv:2105.09220. 2021 May 19.


Example configuration:

.. code-block:: bash

    model:
        model_name: IDSLRUNET
        use_reconstruction_module: true
        input_channels: 24  # coils * 2
        reconstruction_module_output_channels: 24  # coils * 2
        segmentation_module_output_channels: 2
        channels: 64
        num_pools: 2
        padding_size: 11
        drop_prob: 0.0
        normalize: true
        padding: true
        norm_groups: 2
        num_iters: 5
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        complex_data: true
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Multi-Task Learning for MRI Reconstruction and Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-Task Learning for MRI Reconstruction and Segmentation
(:class:`~atommic.collections.multitask.rs.nn.mtlrs.MTLRS`), as presented in [Karkalousos2023]_.

    References
    ----------
    .. [Karkalousos2023] Karkalousos, D., Išgum, I., Marquering, H., Caan, M. W. A., (2023). MultiTask Learning for
        accelerated-MRI Reconstruction and Segmentation of Brain Lesions in Multiple Sclerosis. In Proceedings of
        Machine Learning Research (Vol. 078).


Example configuration:

.. code-block:: bash

    model:
        model_name: MTLRS
        joint_reconstruction_segmentation_module_cascades: 5
        task_adaption_type: multi_task_learning
        use_reconstruction_module: true
        reconstruction_module_recurrent_layer: IndRNN
        reconstruction_module_conv_filters:
            - 64
            - 64
            - 2
        reconstruction_module_conv_kernels:
            - 5
            - 3
            - 3
        reconstruction_module_conv_dilations:
            - 1
            - 2
            - 1
        reconstruction_module_conv_bias:
            - true
            - true
            - false
        reconstruction_module_recurrent_filters:
            - 64
            - 64
            - 0
        reconstruction_module_recurrent_kernels:
            - 1
            - 1
            - 0
        reconstruction_module_recurrent_dilations:
            - 1
            - 1
            - 0
        reconstruction_module_recurrent_bias:
            - true
            - true
            - false
        reconstruction_module_depth: 2
        reconstruction_module_time_steps: 8
        reconstruction_module_conv_dim: 2
        reconstruction_module_num_cascades: 1
        reconstruction_module_dimensionality: 2
        reconstruction_module_no_dc: true
        reconstruction_module_keep_prediction: true
        reconstruction_module_accumulate_predictions: true
        segmentation_module: AttentionUNet
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 2
        segmentation_module_channels: 64
        segmentation_module_pooling_layers: 2
        segmentation_module_dropout: 0.0
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        complex_data: true
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false
    
Multi-Task Learning for MRI Reconstruction and Segmentation with attentionmodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-Task Learning for MRI Reconstruction and Segmentation with attentionmodule
(:class:`~atommic.collections.multitask.rs.nn.mtlrs.MTLRS`)

Example configuration:

.. code-block:: bash

    model:
        model_name: MTLRS
        joint_reconstruction_segmentation_module_cascades: 5
        task_adaption_type: multi_task_learning_softmax
        use_reconstruction_module: true
        reconstruction_module_recurrent_layer: IndRNN
        reconstruction_module_conv_filters:
            - 64
            - 64
            - 2
        reconstruction_module_conv_kernels:
            - 5
            - 3
            - 3
        reconstruction_module_conv_dilations:
            - 1
            - 2
            - 1
        reconstruction_module_conv_bias:
            - true
            - true
            - false
        reconstruction_module_recurrent_filters:
            - 64
            - 64
            - 0
        reconstruction_module_recurrent_kernels:
            - 1
            - 1
            - 0
        reconstruction_module_recurrent_dilations:
            - 1
            - 1
            - 0
        reconstruction_module_recurrent_bias:
            - true
            - true
            - false
        reconstruction_module_depth: 2
        reconstruction_module_time_steps: 8
        reconstruction_module_conv_dim: 2
        reconstruction_module_num_cascades: 1
        reconstruction_module_dimensionality: 2
        reconstruction_module_no_dc: true
        reconstruction_module_keep_prediction: true
        reconstruction_module_accumulate_predictions: true
        segmentation_module: AttentionUNet
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 2
        segmentation_module_channels: 64
        segmentation_module_pooling_layers: 2
        segmentation_module_dropout: 0.0
        attention_module: SemanticGuidanceModule,
        attention_module_kernel_size: 3,
        attention_module_padding: 1,
        attention_module_remove_background: true,
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        complex_data: true
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Reconstruction Segmentation method using UNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reconstruction Segmentation method using UNets for both the reconstruction and segmentation
(:class:`~atommic.collections.multitask.rs.nn.recseg_unet.RecSegUNet`), as presented in [Sui2021]_.

    References
    ----------
    .. [Sui2021] Sui, B, Lv, J, Tong, X, Li, Y, Wang, C. Simultaneous image reconstruction and lesion segmentation in
        accelerated MRI using multitasking learning. Med Phys. 2021; 48: 7189– 7198. https://doi.org/10.1002/mp.15213


Example configuration:

.. code-block:: bash

    model:
        model_name: RECSEGNET
        use_reconstruction_module: true
        input_channels: 1
        reconstruction_module_output_channels: 1
        reconstruction_module_channels: 64
        reconstruction_module_pooling_layers: 2
        reconstruction_module_dropout: 0.0
        segmentation_module_output_channels: 2
        segmentation_module_channels: 64
        segmentation_module_pooling_layers: 2
        segmentation_module_dropout: 0.0
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        complex_data: true
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Segmentation Network MRI
~~~~~~~~~~~~~~~~~~~~~~~~
Segmentation Network MRI (:class:`~atommic.collections.multitask.rs.nn.segnet.SegNet`), as presented in [Sun2019]_.

    References
    ----------
    .. [Sun2019] Sun, L., Fan, Z., Ding, X., Huang, Y., Paisley, J. (2019). Joint CS-MRI Reconstruction and
        Segmentation with a Unified Deep Network. In: Chung, A., Gee, J., Yushkevich, P., Bao, S. (eds) Information
        Processing in Medical Imaging. IPMI 2019. Lecture Notes in Computer Science(), vol 11492. Springer, Cham.
        https://doi.org/10.1007/978-3-030-20351-1_38


Example configuration:

.. code-block:: bash

    model:
        model_name: SEGNET
        use_reconstruction_module: true
        input_channels: 24  # coils * 2
        reconstruction_module_output_channels: 24  # coils * 2
        segmentation_module_output_channels: 2
        channels: 64
        num_pools: 2
        padding_size: 11
        drop_prob: 0.0
        normalize: true
        padding: true
        norm_groups: 2
        num_cascades: 5
        segmentation_final_layer_conv_dim: 2
        segmentation_final_layer_kernel_size: 3
        segmentation_final_layer_dilation: 1
        segmentation_final_layer_bias: False
        segmentation_final_layer_nonlinear: relu
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        complex_data: true
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false


Quantitative MR Imaging (qMRI)
------------------------------

quantitative Cascades of Independently Recurrent Inference Machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
quantitative Cascades of Independently Recurrent Inference Machines
(:class:`~atommic.collections.quantitative.nn.qcirim.qCIRIM`), as presented in [Zhang2022]_.

    References
    ----------
    .. [Zhang2022] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW.
        A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent
        inference machine. NeuroImage. 2022 Dec 1;264:119680.


Example configuration:

.. code-block:: bash

    model:
        model_name: qCIRIM
        use_reconstruction_module: true
        reconstruction_module_recurrent_layer: IndRNN
        reconstruction_module_conv_filters:
            - 64
            - 64
            - 2
        reconstruction_module_conv_kernels:
            - 5
            - 3
            - 3
        reconstruction_module_conv_dilations:
            - 1
            - 2
            - 1
        reconstruction_module_conv_bias:
            - true
            - true
            - false
        reconstruction_module_recurrent_filters:
            - 64
            - 64
            - 0
        reconstruction_module_recurrent_kernels:
            - 1
            - 1
            - 0
        reconstruction_module_recurrent_dilations:
            - 1
            - 1
            - 0
        reconstruction_module_recurrent_bias:
            - true
            - true
            - false
        reconstruction_module_depth: 2
        reconstruction_module_time_steps: 8
        reconstruction_module_conv_dim: 2
        reconstruction_module_num_cascades: 1
        reconstruction_module_dimensionality: 2
        reconstruction_module_no_dc: true
        reconstruction_module_keep_prediction: true
        reconstruction_module_accumulate_predictions: true
        quantitative_module_recurrent_layer: IndRNN
        quantitative_module_conv_filters:
            - 64
            - 64
            - 4
        quantitative_module_conv_kernels:
            - 5
            - 3
            - 3
        quantitative_module_conv_dilations:
            - 1
            - 2
            - 1
        quantitative_module_conv_bias:
            - true
            - true
            - false
        quantitative_module_recurrent_filters:
            - 64
            - 64
            - 0
        quantitative_module_recurrent_kernels:
            - 1
            - 1
            - 0
        quantitative_module_recurrent_dilations:
            - 1
            - 1
            - 0
        quantitative_module_recurrent_bias:
            - true
            - true
            - false
        quantitative_module_depth: 2
        quantitative_module_time_steps: 8
        quantitative_module_conv_dim: 2
        quantitative_module_num_cascades: 1
        quantitative_module_no_dc: true
        quantitative_module_keep_prediction: true
        quantitative_module_accumulate_predictions: true
        quantitative_module_signal_forward_model_sequence: MEGRE
        quantitative_module_dimensionality: 2
        quantitative_module_gamma_regularization_factors:
            - 150.0
            - 150.0
            - 1000.0
            - 150.0
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        shift_B0_input: false
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

quantitative Recurrent Inference Machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
quantitative Recurrent Inference Machines
(:class:`~atommic.collections.quantitative.nn.qrim_base.qrim_block.qRIMBlock`), as presented in [Zhang2022]_.

    References
    ----------
    .. [Zhang2022] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW.
        A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent
        inference machine. NeuroImage. 2022 Dec 1;264:119680.


Example configuration:

.. code-block:: bash

    model:
        model_name: qCIRIM
        use_reconstruction_module: true
        reconstruction_module_recurrent_layer: GRU
        reconstruction_module_conv_filters:
            - 64
            - 64
            - 2
        reconstruction_module_conv_kernels:
            - 5
            - 3
            - 3
        reconstruction_module_conv_dilations:
            - 1
            - 2
            - 1
        reconstruction_module_conv_bias:
            - true
            - true
            - false
        reconstruction_module_recurrent_filters:
            - 64
            - 64
            - 0
        reconstruction_module_recurrent_kernels:
            - 1
            - 1
            - 0
        reconstruction_module_recurrent_dilations:
            - 1
            - 1
            - 0
        reconstruction_module_recurrent_bias:
            - true
            - true
            - false
        reconstruction_module_depth: 2
        reconstruction_module_time_steps: 8
        reconstruction_module_conv_dim: 2
        reconstruction_module_num_cascades: 1
        reconstruction_module_dimensionality: 2
        reconstruction_module_no_dc: true
        reconstruction_module_keep_prediction: true
        reconstruction_module_accumulate_predictions: true
        quantitative_module_recurrent_layer: GRU
        quantitative_module_conv_filters:
            - 64
            - 64
            - 4
        quantitative_module_conv_kernels:
            - 5
            - 3
            - 3
        quantitative_module_conv_dilations:
            - 1
            - 2
            - 1
        quantitative_module_conv_bias:
            - true
            - true
            - false
        quantitative_module_recurrent_filters:
            - 64
            - 64
            - 0
        quantitative_module_recurrent_kernels:
            - 1
            - 1
            - 0
        quantitative_module_recurrent_dilations:
            - 1
            - 1
            - 0
        quantitative_module_recurrent_bias:
            - true
            - true
            - false
        quantitative_module_depth: 2
        quantitative_module_time_steps: 8
        quantitative_module_conv_dim: 2
        quantitative_module_num_cascades: 1
        quantitative_module_no_dc: true
        quantitative_module_keep_prediction: true
        quantitative_module_accumulate_predictions: true
        quantitative_module_signal_forward_model_sequence: MEGRE
        quantitative_module_dimensionality: 2
        quantitative_module_gamma_regularization_factors:
            - 150.0
            - 150.0
            - 1000.0
            - 150.0
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        shift_B0_input: false
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

quantitative End-to-End Variational Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
quantitative End-to-End Variational Network (:class:`~atommic.collections.quantitative.nn.qvarnet.qVarNet`), as
presented in [Zhang2022]_.

    References
    ----------
    .. [Zhang2022] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW.
        A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent
        inference machine. NeuroImage. 2022 Dec 1;264:119680.


Example configuration:

.. code-block:: bash

    model:
        model_name: qVN
        use_reconstruction_module: false
        reconstruction_module_num_cascades: 2
        reconstruction_module_channels: 8
        reconstruction_module_pooling_layers: 2
        reconstruction_module_in_channels: 2
        reconstruction_module_out_channels: 2
        reconstruction_module_padding_size: 11
        reconstruction_module_normalize: true
        reconstruction_module_no_dc: false
        reconstruction_module_accumulate_predictions: false
        quantitative_module_num_cascades: 1
        quantitative_module_channels: 4
        quantitative_module_pooling_layers: 2
        quantitative_module_in_channels: 8
        quantitative_module_out_channels: 8
        quantitative_module_padding_size: 11
        quantitative_module_normalize: true
        quantitative_module_no_dc: false
        quantitative_module_dimensionality: 2
        quantitative_module_accumulate_predictions: false
        quantitative_module_signal_forward_model_sequence: MEGRE
        quantitative_module_gamma_regularization_factors:
            - 150.0
            - 150.0
            - 1000.0
            - 150.0
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        shift_B0_input: false
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false


MRI Reconstruction (REC)
------------------------

Cascades of Independently Recurrent Inference Machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cascades of Independently Recurrent Inference Machines (:class:`~atommic.collections.reconstruction.nn.cirim.CIRIM`),
as presented in [Karkalousos2022]_.

    References
    ----------
    .. [Karkalousos2022] Karkalousos D, Noteboom S, Hulst HE, Vos FM, Caan MWA. Assessment of data consistency through
        cascades of independently recurrent inference machines for fast and robust accelerated MRI reconstruction.
        Phys Med Biol. 2022 Jun 8;67(12). doi: 10.1088/1361-6560/ac6cc2. PMID: 35508147.


Example configuration:

.. code-block:: bash

    model:
        model_name: CIRIM
        recurrent_layer: IndRNN
        conv_filters:
            - 64
            - 64
            - 2
        conv_kernels:
            - 5
            - 3
            - 3
        conv_dilations:
            - 1
            - 2
            - 1
        conv_bias:
            - true
            - true
            - false
        recurrent_filters:
            - 64
            - 64
            - 0
        recurrent_kernels:
            - 1
            - 1
            - 0
        recurrent_dilations:
            - 1
            - 1
            - 0
        recurrent_bias:
            - true
            - true
            - false
        depth: 2
        time_steps: 8
        conv_dim: 2
        num_cascades: 8
        no_dc: true
        keep_prediction: true
        accumulate_predictions: true
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Convolutional Recurrent Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convolutional Recurrent Neural Networks (:class:`~atommic.collections.reconstruction.nn.crnn.CRNNet`), as presented
in [Qin2019]_.

    References
    ----------
    .. [Qin2019] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional
        Recurrent Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol.
        38, no. 1, pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.


Example configuration:

.. code-block:: bash

    model:
        model_name: CRNNet
        num_iterations: 10
        hidden_channels: 64
        n_convs: 3
        batchnorm: false
        no_dc: false
        accumulate_predictions: true
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Deep Cascade of Convolutional Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Deep Cascade of Convolutional Neural Networks (:class:`~atommic.collections.reconstruction.nn.ccnn.CascadeNet`), as
presented in [Schlemper2017]_.

    References
    ----------
    .. [Schlemper2017] Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D., A Deep Cascade of
        Convolutional Neural Networks for MR Image Reconstruction. Information Processing in Medical Imaging (IPMI),
        2017.


Example configuration:

.. code-block:: bash

    model:
        model_name: CascadeNet
        num_cascades: 10
        hidden_channels: 64
        n_convs: 5
        batchnorm: false
        no_dc: false
        accumulate_predictions: false
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Down-Up Net
~~~~~~~~~~~
Down-Up NET (:class:`~atommic.collections.reconstruction.nn.dunet.DUNet`), inspired by [Hammernik2021]_.

    References
    ----------
    .. [Hammernik2021] Hammernik, K, Schlemper, J, Qin, C, et al. Systematic valuation of iterative deep neural
        networks for fast parallel MRI reconstruction with sensitivity-weighted coil combination. Magn Reson Med.
        2021; 86: 1859– 1872. https://doi.org/10.1002/mrm.28827


Example configuration:

.. code-block:: bash

    model:
        model_name: DUNet
        num_iter: 10
        reg_model_architecture: DIDN
        didn_hidden_channels: 64
        didn_num_dubs: 2
        didn_num_convs_recon: 1
        data_consistency_term: VS
        data_consistency_lambda_init: 0.1
        data_consistency_iterations: 10
        shared_params: false
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

End-to-End Variational Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-End Variational Network (:class:`~atommic.collections.reconstruction.nn.varnet.VarNet`), as presented in
[Sriram2020]_.

    References
    ----------
    .. [Sriram2020] Sriram A, Zbontar J, Murrell T, Defazio A, Zitnick CL, Yakubova N, Knoll F, Johnson P. End-to-end
        variational networks for accelerated MRI reconstruction. InInternational Conference on Medical Image Computing
        and Computer-Assisted Intervention 2020 Oct 4 (pp. 64-73). Springer, Cham.


Example configuration:

.. code-block:: bash

    model:
        model_name: VN
        num_cascades: 8
        channels: 18
        pooling_layers: 4
        padding_size: 11
        normalize: true
        no_dc: false
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Independently Recurrent Inference Machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Independently Recurrent Inference Machines
(:class:`~atommic.collections.reconstruction.nn.rim_base.rim_block.RIMBlock`), as presented in [Karkalousos2022]_.

    References
    ----------
    .. [Karkalousos2022] Karkalousos D, Noteboom S, Hulst HE, Vos FM, Caan MWA. Assessment of data consistency through
        cascades of independently recurrent inference machines for fast and robust accelerated MRI reconstruction.
        Phys Med Biol. 2022 Jun 8;67(12). doi: 10.1088/1361-6560/ac6cc2. PMID: 35508147.


Example configuration:

.. code-block:: bash

    model:
        model_name: CIRIM
        recurrent_layer: IndRNN
        conv_filters:
            - 64
            - 64
            - 2
        conv_kernels:
            - 5
            - 3
            - 3
        conv_dilations:
            - 1
            - 2
            - 1
        conv_bias:
            - true
            - true
            - false
        recurrent_filters:
            - 64
            - 64
            - 0
        recurrent_kernels:
            - 1
            - 1
            - 0
        recurrent_dilations:
            - 1
            - 1
            - 0
        recurrent_bias:
            - true
            - true
            - false
        depth: 2
        time_steps: 8
        conv_dim: 2
        num_cascades: 1
        no_dc: true
        keep_prediction: true
        accumulate_predictions: true
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network
(:class:`~atommic.collections.reconstruction.nn.jointicnet.JointICNet`), as presented in [Jun2021]_.

    References
    ----------
    .. [Jun2021] Jun, Yohan, et al. “Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network
        (Joint-ICNet) for Fast MRI.” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE,
        2021, pp. 5266–75. DOI.org (Crossref), https://doi.org/10.1109/CVPR46437.2021.00523.


Example configuration:

.. code-block:: bash

    model:
        model_name: JointICNet
        num_iter: 2
        kspace_unet_num_filters: 16
        kspace_unet_num_pool_layers: 2
        kspace_unet_dropout_probability: 0.0
        kspace_unet_padding_size: 11
        kspace_unet_normalize: true
        imspace_unet_num_filters: 16
        imspace_unet_num_pool_layers: 2
        imspace_unet_dropout_probability: 0.0
        imspace_unet_padding_size: 11
        imspace_unet_normalize: true
        sens_unet_num_filters: 16
        sens_unet_num_pool_layers: 2
        sens_unet_dropout_probability: 0.0
        sens_unet_padding_size: 11
        sens_unet_normalize: true
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

KIKINet
~~~~~~~
KIKINet (:class:`~atommic.collections.reconstruction.nn.kikinet.KIKINet`), modified to work with multi-coil k-space
data, as presented in [Taejoon2018]_.

    References
    ----------
    .. [Taejoon2018] Eo, Taejoon, et al. “KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing
        Undersampled Magnetic Resonance Images.” Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp.
        2188–201. PubMed, https://doi.org/10.1002/mrm.27201.


Example configuration:

.. code-block:: bash

    model:
        model_name: KIKINet
        num_iter: 2
        kspace_model_architecture: UNET
        kspace_in_channels: 2
        kspace_out_channels: 2
        kspace_unet_num_filters: 16
        kspace_unet_num_pool_layers: 2
        kspace_unet_dropout_probability: 0.0
        kspace_unet_padding_size: 11
        kspace_unet_normalize: true
        imspace_model_architecture: UNET
        imspace_in_channels: 2
        imspace_unet_num_filters: 16
        imspace_unet_num_pool_layers: 2
        imspace_unet_dropout_probability: 0.0
        imspace_unet_padding_size: 11
        imspace_unet_normalize: true
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Learned Primal-Dual Net
~~~~~~~~~~~~~~~~~~~~~~~
Learned Primal-Dual Net (:class:`~atommic.collections.reconstruction.nn.lpd.LPDNet`), as presented in [Adler2018]_.

    References
    ----------
    .. [Adler2018] Adler, Jonas, and Ozan Öktem. “Learned Primal-Dual Reconstruction.” IEEE Transactions on Medical
        Imaging, vol. 37, no. 6, June 2018, pp. 1322–32. arXiv.org, https://doi.org/10.1109/TMI.2018.2799231.


Example configuration:

.. code-block:: bash

    model:
        model_name: LPDNet
        num_primal: 5
        num_dual: 5
        num_iter: 5
        primal_model_architecture: UNET
        primal_in_channels: 2
        primal_out_channels: 2
        primal_unet_num_filters: 16
        primal_unet_num_pool_layers: 2
        primal_unet_dropout_probability: 0.0
        primal_unet_padding_size: 11
        primal_unet_normalize: true
        dual_model_architecture: UNET
        dual_in_channels: 2
        dual_out_channels: 2
        dual_unet_num_filters: 16
        dual_unet_num_pool_layers: 2
        dual_unet_dropout_probability: 0.0
        dual_unet_padding_size: 11
        dual_unet_normalize: true
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

MoDL: Model Based Deep Learning Architecture for Inverse Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MoDL: Model Based Deep Learning Architecture for Inverse Problems
(:class:`~atommic.collections.reconstruction.nn.modl.MoDL`).

Adjusted to optionally perform a data consistency step (Conjugate Gradient), as presented in [Aggarwal2018]_,
[Yaman2020]_. If dc is set to False, the network will perform a simple residual learning step.

    References
    ----------
    .. [Aggarwal2018] MoDL: Model Based Deep Learning Architecture for Inverse Problems by H.K. Aggarwal, M.P Mani, and
        Mathews Jacob in IEEE Transactions on Medical Imaging, 2018

    .. [Yaman2020] Yaman, B, Hosseini, SAH, Moeller, S, Ellermann, J, Uğurbil, K, Akçakaya, M. Self-supervised
        learning of physics-guided reconstruction neural networks without fully sampled reference data. Magn Reson
        Med. 2020; 84: 3172– 3191. https://doi.org/10.1002/mrm.28378


Example configuration:

.. code-block:: bash

    model:
        model_name: MoDL
        unrolled_iterations: 5
        residual_blocks: 5
        channels: 64
        regularization_factor: 0.1
        penalization_weight: 1.0
        conjugate_gradient_dc: false
        conjugate_gradient_iterations: 1
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

MultiDomainNet
~~~~~~~~~~~~~~
Feature-level multi-domain module. Inspired by AIRS Medical submission to the FastMRI 2020 challenge.


Example configuration:

.. code-block:: bash

    model:
        model_name: MultiDomainNet
        standardization: true
        num_filters: 64
        num_pool_layers: 2
        dropout_probability: 0.0
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

ProximalGradient
~~~~~~~~~~~~~~~~
Proximal/Conjugate Gradient (:class:`~atommic.collections.reconstruction.nn.proximal_gradient.ProximalGradient`),
according to [Aggarwal2018]_, [Yaman2020]_.

    References
    ----------
    .. [Aggarwal2018] MoDL: Model Based Deep Learning Architecture for Inverse Problems by H.K. Aggarwal, M.P Mani, and
        Mathews Jacob in IEEE Transactions on Medical Imaging, 2018

    .. [Yaman2020] Yaman, B, Hosseini, SAH, Moeller, S, Ellermann, J, Uğurbil, K, Akçakaya, M. Self-supervised
        learning of physics-guided reconstruction neural networks without fully sampled reference data. Magn Reson
        Med. 2020; 84: 3172– 3191. https://doi.org/10.1002/mrm.28378


Example configuration:

.. code-block:: bash

    model:
        model_name: PROXIMALGRADIENT
        conjugate_gradient_dc: true
        conjugate_gradient_iterations: 10
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Recurrent Variational Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Recurrent Variational Network (:class:`~atommic.collections.reconstruction.nn.recurrentvarnet.RecurrentVarNet`), as
presented in [Yiasemis2021]_.

    References
    ----------
    .. [Yiasemis2021] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver
        Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.


Example configuration:

.. code-block:: bash

    model:
        model_name: RVN
        in_channels: 2
        recurrent_hidden_channels: 64
        recurrent_num_layers: 4
        num_steps: 8
        no_parameter_sharing: true
        learned_initializer: true
        initializer_initialization: "sense"
        initializer_channels:
            - 32
            - 32
            - 64
            - 64
        initializer_dilations:
            - 1
            - 1
            - 2
            - 4
        initializer_multiscale: 1
        accumulate_predictions: false
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Recurrent Inference Machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Recurrent Inference Machines (:class:`~atommic.collections.reconstruction.nn.rim_base.rim_block.RIMBlock`), as
presented in [Lonning19]_.

    References
    ----------
    .. [Lonning19] Lønning K, Putzky P, Sonke JJ, Reneman L, Caan MW, Welling M. Recurrent inference machines for
        reconstructing heterogeneous MRI data. Medical image analysis. 2019 Apr 1;53:64-78.


Example configuration:

.. code-block:: bash

    model:
        model_name: CIRIM
        recurrent_layer: GRU
        conv_filters:
            - 64
            - 64
            - 2
        conv_kernels:
            - 5
            - 3
            - 3
        conv_dilations:
            - 1
            - 2
            - 1
        conv_bias:
            - true
            - true
            - false
        recurrent_filters:
            - 64
            - 64
            - 0
        recurrent_kernels:
            - 1
            - 1
            - 0
        recurrent_dilations:
            - 1
            - 1
            - 0
        recurrent_bias:
            - true
            - true
            - false
        depth: 2
        time_steps: 8
        conv_dim: 2
        num_cascades: 1
        no_dc: true
        keep_prediction: true
        accumulate_predictions: true
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

UNet
~~~~
UNet (:class:`~atommic.collections.reconstruction.nn.unet.UNet`), as presented in [Ronneberger2015]_.

    References
    ----------
    .. [Ronneberger2015] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
        image segmentation. In International Conference on Medical image computing and computer-assisted intervention,
        pages 234–241. Springer, 2015.


Example configuration:

.. code-block:: bash

    model:
        model_name: UNet
        channels: 64
        pooling_layers: 4
        in_channels: 2
        out_channels: 2
        padding_size: 11
        dropout: 0.0
        normalize: true
        norm_groups: 2
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Variable Splitting Network
~~~~~~~~~~~~~~~~~~~~~~~~~~
Variable Splitting Network (:class:`~atommic.collections.reconstruction.nn.vsnet.VSNet`), as presented in [Duan2019]_.

    References
    ----------
    .. [Duan2019] Duan, J. et al. (2019) Vs-net: Variable splitting network for accelerated parallel MRI
        reconstruction, Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial
        Intelligence and Lecture Notes in Bioinformatics), 11767 LNCS, pp. 713–722. doi: 10.1007/978-3-030-32251-9_78.


Example configuration:

.. code-block:: bash

    model:
        model_name: VSNet
        num_cascades: 10
        imspace_model_architecture: CONV
        imspace_in_channels: 2
        imspace_out_channels: 2
        imspace_conv_hidden_channels: 64
        imspace_conv_n_convs: 4
        imspace_conv_batchnorm: false
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

XPDNet
~~~~~~
XPDNet (:class:`~atommic.collections.reconstruction.nn.xpdnet.XPDNet`), as presented in [Ramzi2021]_.

    References
    ----------
    .. [Ramzi2021] Ramzi, Zaccharie, et al. “XPDNet for MRI Reconstruction: An Application to the 2020 FastMRI
        Challenge. ArXiv:2010.07290 [Physics, Stat], July 2021. arXiv.org, http://arxiv.org/abs/2010.07290.


Example configuration:

.. code-block:: bash

    model:
        model_name: XPDNet
        num_primal: 5
        num_dual: 1
        num_iter: 20
        use_primal_only: true
        kspace_model_architecture: CONV
        kspace_in_channels: 2
        kspace_out_channels: 2
        dual_conv_hidden_channels: 16
        dual_conv_num_dubs: 2
        dual_conv_batchnorm: false
        image_model_architecture: MWCNN
        imspace_in_channels: 2
        imspace_out_channels: 2
        mwcnn_hidden_channels: 16
        mwcnn_num_scales: 2
        mwcnn_bias: true
        mwcnn_batchnorm: false
        normalize_image: false
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Zero-Filled
~~~~~~~~~~~
Zero-Filled reconstruction using either root-sum-of-squares (RSS) or SENSE (SENSitivity Encoding, as presented in
[Pruessmann1999]_).

    References
    ----------
    .. [Pruessmann1999] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast MRI.
        Magn Reson Med 1999; 42:952-962.


Example configuration:

.. code-block:: bash

    model:
        model_name: ZF
        # task & dataset related parameters
        coil_combination_method: SENSE
        coil_dim: 1
        complex_valued_type: stacked  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false
        fft_normalization: backward
        spatial_dims:
            - -2
            - -1
        normalization_type: minmax
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false


MRI Segmentation (SEG)
----------------------

Attention UNet
~~~~~~~~~~~~~~
Attention UNet for MRI segmentation
(:class:`~atommic.collections.segmentation.nn.attentionunet.SegmentationAttentionUNet`), as presented in [Oktay2018]_.

    References
    ----------
    .. [Oktay2018] O. Oktay, J. Schlemper, L.L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N.Y.
        Hammerla, B. Kainz, B. Glocker, D. Rueckert. Attention U-Net: Learning Where to Look for the Pancreas. 2018.
        https://arxiv.org/abs/1804.03999


Example configuration:

.. code-block:: bash

    model:
        model_name: SEGMENTATIONATTENTIONUNET
        segmentation_module: AttentionUNet
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 4
        segmentation_module_channels: 32
        segmentation_module_pooling_layers: 5
        segmentation_module_dropout: 0.0
        # task & dataset related parameters
        coil_combination_method: SENSE  # if complex data
        coil_dim: 1  # if complex data
        complex_data: true  # or false if using magnitude data
        complex_valued_type: stacked (only for complex data)  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false  # if complex data
        fft_normalization: backward  # if complex data
        spatial_dims:
            - -2  # if complex data
            - -1  # if complex data
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Dynamic UNet
~~~~~~~~~~~~
Dynamic UNet for MRI segmentation (:class:`~atommic.collections.segmentation.nn.dynunet.SegmentationDYNUNet`), as
presented in [Isensee2018]_.

    References
    ----------
    .. [Isensee2018] Isensee F, Petersen J, Klein A, Zimmerer D, Jaeger PF, Kohl S, Wasserthal J, Koehler G,
        Norajitra T, Wirkert S, Maier-Hein KH. nnu-net: Self-adapting framework for u-net-based medical image
        segmentation. arXiv preprint arXiv:1809.10486. 2018 Sep 27.


Example configuration:

.. code-block:: bash

    model:
        model_name: SEGMENTATIONDYNUNET
        segmentation_module: DYNUNet
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 4
        segmentation_module_channels:
            - 64
            - 128
            - 256
            - 512
        segmentation_module_kernel_size:
            - 3
            - 3
            - 3
            - 1
        segmentation_module_strides:
            - 1
            - 1
            - 1
            - 1
        segmentation_module_dropout: 0.0
        segmentation_module_norm: instance
        segmentation_module_activation: leakyrelu
        segmentation_module_deep_supervision: true
        segmentation_module_deep_supervision_levels: 2
        # task & dataset related parameters
        coil_combination_method: SENSE  # if complex data
        coil_dim: 1  # if complex data
        complex_data: true  # or false if using magnitude data
        complex_valued_type: stacked (only for complex data)  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false  # if complex data
        fft_normalization: backward  # if complex data
        spatial_dims:
            - -2  # if complex data
            - -1  # if complex data
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

Lambda UNet
~~~~~~~~~~~
Lambda UNet for MRI segmentation (:class:`~atommic.collections.segmentation.nn.lambdaunet.SegmentationLambdaUNet`), as
presented in [Yanglan2021]_.

    References
    ----------
    .. [Yanglan2021] Yanglan Ou, Ye Yuan, Xiaolei Huang, Kelvin Wong, John Volpi, James Z. Wang, Stephen T.C. Wong.
        LambdaUNet: 2.5D Stroke Lesion Segmentation of Diffusion-weighted MR Images. 2021.
        https://arxiv.org/abs/2104.13917


Example configuration:

.. code-block:: bash

    model:
        model_name: SEGMENTATIONLAMBDAUNET
        segmentation_module: LambdaUNet
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 4
        segmentation_module_channels: 64
        segmentation_module_pooling_layers: 2
        segmentation_module_dropout: 0.0
        segmentation_module_query_depth: 16
        segmentation_module_intra_depth: 1
        segmentation_module_receptive_kernel: 1
        segmentation_module_temporal_kernel: 1
        # task & dataset related parameters
        coil_combination_method: SENSE  # if complex data
        coil_dim: 1  # if complex data
        complex_data: true  # or false if using magnitude data
        complex_valued_type: stacked (only for complex data)  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false  # if complex data
        fft_normalization: backward  # if complex data
        spatial_dims:
            - -2  # if complex data
            - -1  # if complex data
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

UNet
~~~~
2D UNet for MRI segmentation (:class:`~atommic.collections.segmentation.nn.unet.SegmentationUNet`), as
presented in [Ronneberger2015]_.

    References
    ----------
    .. [Ronneberger2015] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
        image segmentation. In International Conference on Medical image computing and computer-assisted intervention,
        pages 234–241. Springer, 2015.


Example configuration:

.. code-block:: bash

    model:
        model_name: SEGMENTATIONUNET
        segmentation_module: UNet
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 4
        segmentation_module_channels: 64
        segmentation_module_pooling_layers: 2
        segmentation_module_dropout: 0.0
        # task & dataset related parameters
        coil_combination_method: SENSE  # if complex data
        coil_dim: 1  # if complex data
        complex_data: true  # or false if using magnitude data
        complex_valued_type: stacked (only for complex data)  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false  # if complex data
        fft_normalization: backward  # if complex data
        spatial_dims:
            - -2  # if complex data
            - -1  # if complex data
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

UNet 3D
~~~~~~~
3D UNet for MRI segmentation (:class:`~atommic.collections.segmentation.nn.unet3d.Segmentation3DUNet`), as
presented in [Ronneberger2015]_.

    References
    ----------
    .. [Ronneberger2015] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
        image segmentation. In International Conference on Medical image computing and computer-assisted intervention,
        pages 234–241. Springer, 2015.


Example configuration:

.. code-block:: bash

    model:
        model_name: SEGMENTATION3DUNET
        segmentation_module: UNet
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 4
        segmentation_module_channels: 64
        segmentation_module_pooling_layers: 2
        segmentation_module_dropout: 0.0
        # task & dataset related parameters
        coil_combination_method: SENSE  # if complex data
        coil_dim: 1  # if complex data
        complex_data: true  # or false if using magnitude data
        complex_valued_type: stacked (only for complex data)  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false  # if complex data
        fft_normalization: backward  # if complex data
        spatial_dims:
            - -2  # if complex data
            - -1  # if complex data
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

UNETR
~~~~~
UNETR for MRI segmentation (:class:`~atommic.collections.segmentation.nn.unetr.SegmentationUNetR`), as
presented in [Hatamizadeh2022]_.

    References
    ----------
    .. [Hatamizadeh2022] Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr:
        Transformers for 3d medical image segmentation. InProceedings of the IEEE/CVF Winter Conference on
        Applications of Computer Vision 2022 (pp. 574-584).


Example configuration:

.. code-block:: bash

    model:
        model_name: SEGMENTATIONUNETR
        segmentation_module: UNETR
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 3
        segmentation_module_img_size: (256, 256)
        segmentation_module_channels: 64
        segmentation_module_hidden_size: 768
        segmentation_module_mlp_dim: 3072
        segmentation_module_num_heads: 12
        segmentation_module_pos_embed: conv
        segmentation_module_norm_name: instance
        segmentation_module_conv_block: true
        segmentation_module_res_block: true
        segmentation_module_dropout: 0.0
        segmentation_module_qkv_bias: false
        # task & dataset related parameters
        coil_combination_method: SENSE  # if complex data
        coil_dim: 1  # if complex data
        complex_data: true  # or false if using magnitude data
        complex_valued_type: stacked (only for complex data)  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false  # if complex data
        fft_normalization: backward  # if complex data
        spatial_dims:
            - -2  # if complex data
            - -1  # if complex data
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false

V-Net
~~~~~
V-Net for MRI segmentation (:class:`~atommic.collections.segmentation.nn.vnet.SegmentationVNet`), as
presented in [Milletari2016]_.

    References
    ----------
    .. [Milletari2016] Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks
        for Volumetric Medical Image Segmentation, 2016. https://arxiv.org/abs/1606.04797

Example configuration:

.. code-block:: bash

    model:
        use_reconstruction_module: false
        segmentation_module: VNet
        segmentation_module_input_channels: 1
        segmentation_module_output_channels: 4
        segmentation_module_activation: elu
        segmentation_module_dropout: 0.0
        segmentation_module_bias: False
        segmentation_module_padding_size: 15
        # task & dataset related parameters
        coil_combination_method: SENSE  # if complex data
        coil_dim: 1  # if complex data
        complex_data: true  # or false if using magnitude data
        complex_valued_type: stacked (only for complex data)  # stacked, complex_abs, complex_sqrt_abs
        consecutive_slices: 1
        dimensionality: 2
        estimate_coil_sensitivity_maps_with_nn: false
        fft_centered: false  # if complex data
        fft_normalization: backward  # if complex data
        spatial_dims:
            - -2  # if complex data
            - -1  # if complex data
        magnitude_input: true
        normalization_type: minmax
        normalize_segmentation_output: true
        unnormalize_loss_inputs: false
        unnormalize_log_outputs: false
