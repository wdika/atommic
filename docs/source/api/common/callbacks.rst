*********
Callbacks
*********

Exponential Moving Average (EMA)
================================

During training, EMA maintains a moving average of the trained parameters.
EMA parameters can produce significantly better results and faster convergence for a variety of different domains and
models.

EMA is a simple calculation. EMA Weights are pre-initialized with the model weights at the start of training.

Every training update, the EMA weights are updated based on the new model weights.

.. math::

    ema_w = ema_w * decay + model_w * (1-decay)

Enabling EMA is straightforward in your .yaml file. For example:

.. code-block:: bash

        exp_manager.ema.enable=True
        exp_manager.ema.decay=0.999

Also offers other helpful arguments.

.. list-table::
   :header-rows: 1

   * - Argument
     - Description
   * - `exp_manager.ema.validate_original_weights=True`
     - Validate the original weights instead of EMA weights.
   * - `exp_manager.ema.every_n_steps=2`
     - Apply EMA every N steps instead of every step.
   * - `exp_manager.ema.cpu_offload=True`
     - Offload EMA weights to CPU. May introduce significant slow-downs.
