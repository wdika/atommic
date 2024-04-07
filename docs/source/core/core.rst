Training & Testing
==================

Basics
------

ATOMMIC models contain everything needed to train and reproduce AI models for MR Imaging tasks.

ATOMMIC uses `Hydra <https://hydra.cc/>`_ for configuring both ATOMMIC models and the PyTorch Lightning Trainer.

.. note:: Every ATOMMIC model has an example configuration file and training script that can be found
    `here <https://github.com/wdika/atommic/tree/main/projects>`_.

The end result of using ATOMMIC, `Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_, and
Hydra is that ATOMMIC models all have the same look and feel and are also fully compatible with the PyTorch ecosystem.


Training
--------

ATOMMIC leverages `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ for model training. PyTorch Lightning lets
ATOMMIC decouple the AI MR Imaging code from the PyTorch training code. This means that ATOMMIC users can focus on
their domain (...) and build complex AI applications without having to rewrite boiler plate code for PyTorch training.

When using PyTorch Lightning, ATOMMIC users can automatically train with:

- multi-GPU/multi-node
- mixed precision (supported types are 16-mixed, bf16-mixed, 32-true, 64-true, 64, 32, and 16)
- model checkpointing
- logging
- early stopping
- and more

The two main aspects of the Lightning API are the
`LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#>`_
and the `Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_.

PyTorch Lightning ``LightningModule``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ATOMMIC model is a ``LightningModule`` which is an ``nn.module``. This means that ATOMMIC models are compatible
with the PyTorch ecosystem and can be plugged into existing PyTorch workflows.


PyTorch Lightning Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~

Since every ATOMMIC model is a ``LightningModule``, we can automatically take advantage of the PyTorch Lightning
``Trainer``.


Configuration
-------------

Hydra is an open-source Python framework that simplifies configuration for complex applications that must bring
together many different software libraries. To train an MRI AI model, we must be able to configure:

- neural network architectures
- training and optimization algorithms
- data pre/post processing
- data augmentation
- experiment logging/visualization
- model checkpointing

For an introduction to using Hydra, refer to the `Hydra Tutorials <https://hydra.cc/docs/tutorials/intro>`_.

With Hydra, we can configure everything needed for ATOMMIC through Configuration Files (YAML).


YAML
~~~~

ATOMMIC provides YAML configuration files for all of our tasks and publicly available datasets in
`projects <https://github.com/wdika/atommic/tree/main/projects>`_. YAML files make it easy to experiment with
different model and training configurations.

Every ATOMMIC example YAML has the same underlying configuration structure:

- trainer
- exp_manager
- model

Model configuration always contain ``train_ds``, ``validation_ds``, ``test_ds``, and ``optim``.  Model architectures
vary across domains, therefore, refer to the MTL, QI, REC, and SEG Collections documentation for more detailed
information on Model architecture configuration.

A ATOMMIC configuration file should look similar to the following:

.. code-block:: yaml

    # PyTorch Lightning Trainer configuration
    # any argument of the Trainer object can be set here
    trainer:
        devices: 1 # number of gpus per node
        accelerator: gpu
        num_nodes: 1 # number of nodes
        max_epochs: 10 # how many training epochs to run
        val_check_interval: 1.0 # run validation after every epoch

    # Experiment logging configuration
    exp_manager:
        exp_dir: /path/to/my/atommic/experiments
        name: name_of_my_experiment
        create_tensorboard_logger: True
        create_wandb_logger: True

    # Model configuration
    # model network architecture, train/val/test datasets, data augmentation, and optimization
    model:
        train_ds:
            data_path: /path/to/my/train_data/
            batch_size: 1
            shuffle: True
        validation_ds:
            data_path: /path/to/my/validation_data/
            batch_size: 1
            shuffle: False
        test_ds:
            data_path: /path/to/my/test_data/
            batch_size: 1
            shuffle: False
        optim:
            name: novograd
            lr: .01
            betas: [0.8, 0.5]
            weight_decay: 0.001
        # network architecture can vary greatly depending on the domain
        encoder:
            ...
        decoder:
            ...


.. _optimization-label:

Optimization
------------

Optimizers and learning rate schedules are configurable across all ATOMMIC models and have their own namespace. Here
is a sample YAML configuration for a Novograd optimizer with Cosine Annealing learning rate schedule.

.. code-block:: yaml

    optim:
        name: novograd
        lr: 0.01

        # optimizer arguments
        betas: [0.8, 0.25]
        weight_decay: 0.001

        # scheduler setup
        sched:
            name: CosineAnnealing

            # Optional arguments
            max_steps: -1 # computed at runtime or explicitly set here
            monitor: val_loss
            reduce_on_plateau: false

            # scheduler config override
            warmup_steps: 1000
            warmup_ratio: null
            min_lr: 1e-9:


.. _optimizers-label:

Optimizers
~~~~~~~~~~

``name`` corresponds to the lowercase name of the optimizer. To view a list of available optimizers, run:

.. code-block:: Python

    from atommic.core.optim.optimizers import AVAILABLE_OPTIMIZERS

    for name, opt in AVAILABLE_OPTIMIZERS.items():
        print(f'name: {name}, opt: {opt}')

.. code-block:: bash

    name: sgd opt: <class 'torch.optim.sgd.SGD'>
    name: adam opt: <class 'torch.optim.adam.Adam'>
    name: adamw opt: <class 'torch.optim.adamw.AdamW'>
    name: adadelta opt: <class 'torch.optim.adadelta.Adadelta'>
    name: adamax opt: <class 'torch.optim.adamax.Adamax'>
    name: adagrad opt: <class 'torch.optim.adagrad.Adagrad'>
    name: rmsprop opt: <class 'torch.optim.rmsprop.RMSprop'>
    name: rprop opt: <class 'torch.optim.rprop.Rprop'>
    name: novograd opt: <class 'atommic.core.optim.novograd.Novograd'>
    name: lion, opt: <class 'atommic.core.optim.lion.Lion'>


Optimizer Params
~~~~~~~~~~~~~~~~

Optimizer params can vary between optimizers but the ``lr`` param is required for all optimizers. To see the available
params for an optimizer, we can look at its corresponding dataclass.


Register Optimizer
~~~~~~~~~~~~~~~~~~

To register a new optimizer to be used with ATOMMIC, run:

.. autofunction:: atommic.core.optim.optimizers.register_optimizer

.. _learning-rate-schedulers-label:

Learning Rate Schedulers
~~~~~~~~~~~~~~~~~~~~~~~~

Learning rate schedulers can be optionally configured under the ``optim.sched`` namespace.

``name`` corresponds to the name of the learning rate schedule. To view a list of available schedulers, run:

.. code-block:: Python

    from atommic.core.optim.lr_scheduler import AVAILABLE_SCHEDULERS

    for name, opt in AVAILABLE_SCHEDULERS.items():
        print(f'name: {name}, schedule: {opt}')

.. code-block:: bash

    name: WarmupPolicy, schedule: <class 'atommic.core.optim.lr_scheduler.WarmupPolicy'>
    name: WarmupHoldPolicy, schedule: <class 'atommic.core.optim.lr_scheduler.WarmupHoldPolicy'>
    name: SquareAnnealing, schedule: <class 'atommic.core.optim.lr_scheduler.SquareAnnealing'>
    name: CosineAnnealing, schedule: <class 'atommic.core.optim.lr_scheduler.CosineAnnealing'>
    name: NoamAnnealing, schedule: <class 'atommic.core.optim.lr_scheduler.NoamAnnealing'>
    name: WarmupAnnealing, schedule: <class 'atommic.core.optim.lr_scheduler.WarmupAnnealing'>
    name: InverseSquareRootAnnealing, schedule: <class 'atommic.core.optim.lr_scheduler.InverseSquareRootAnnealing'>
    name: SquareRootAnnealing, schedule: <class 'atommic.core.optim.lr_scheduler.SquareRootAnnealing'>
    name: PolynomialDecayAnnealing, schedule: <class 'atommic.core.optim.lr_scheduler.PolynomialDecayAnnealing'>
    name: PolynomialHoldDecayAnnealing, schedule: <class 'atommic.core.optim.lr_scheduler.PolynomialHoldDecayAnnealing'>
    name: StepLR, schedule: <class 'torch.optim.lr_scheduler.StepLR'>
    name: ExponentialLR, schedule: <class 'torch.optim.lr_scheduler.ExponentialLR'>
    name: ReduceLROnPlateau, schedule: <class 'torch.optim.lr_scheduler.ReduceLROnPlateau'>
    name: CyclicLR, schedule: <class 'torch.optim.lr_scheduler.CyclicLR'>


Register scheduler
~~~~~~~~~~~~~~~~~~

To register a new scheduler to be used with ATOMMIC, run:

.. autofunction:: atommic.core.optim.lr_scheduler.register_scheduler

Save and Restore
----------------

ATOMMIC models all come with ``.save_to`` and ``.restore_from`` methods.

Save
~~~~

To save a ATOMMIC model, run:

.. code-block:: Python

    model.save_to('/path/to/model.atommic')

Everything needed to use the trained model is packaged and saved in the ``.atommic`` file.

.. note:: A ``.atommic`` file is simply an archive like any other ``.tar`` file.

Restore
~~~~~~~

To restore a ATOMMIC model, run:

.. code-block:: Python

    # Here, you should usually use the class of the model, or simply use ModelPT.restore_from() for simplicity.
    model.restore_from('/path/to/model.atommic')

When using the PyTorch Lightning Trainer, a PyTorch Lightning checkpoint is created. These are mainly used within
ATOMMIC to auto-resume training. Since ATOMMIC models are ``LightningModules``, the PyTorch Lightning method
``load_from_checkpoint`` is available. Note that ``load_from_checkpoint`` won't necessarily work out-of-the-box for
all models as some models require more artifacts than just the checkpoint to be restored. For these models, the user
will have to override ``load_from_checkpoint`` if they want to use it.

It's highly recommended to use ``restore_from`` to load ATOMMIC models.

Restore with Modified Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, there may be a need to modify the model (or it's sub-components) prior to restoring a model. A common case
is when the model's internal config must be updated due to various reasons (such as deprecation, newer versioning,
support a new feature). As long as the model has the same parameters as compared to the original config, the
parameters can once again be restored safely.

In ATOMMIC, as part of the .atommic file, the model's internal config will be preserved. This config is used during
restoration, and as shown below we can update this config prior to restoring the model.

.. code-block::

    # When restoring a model, you should generally use the class of the model
    # Obtain the config (as an OmegaConf object)
    config = model_class.restore_from('/path/to/model.atommic', return_config=True)
    # OR
    config = model_class.from_pretrained('name_of_the_model', return_config=True)

    # Modify the config as needed
    config.x.y = z

    # Restore the model from the updated config
    model = model_class.restore_from('/path/to/model.atommic', override_config_path=config)
    # OR
    model = model_class.from_pretrained('name_of_the_model', override_config_path=config)

Register Artifacts
------------------

ATOMMIC models can save additional artifacts in the .atommic file by calling ``.register_artifact``.
When restoring ATOMMIC models using ``.restore_from`` or ``.from_pretrained``, any artifacts that were registered will
be available automatically.

By default, ``.register_artifact`` will always return a path. If the model is being restored from a .atommic file,
then that path will be to the artifact in the .atommic file. Otherwise, ``.register_artifact`` will return the local
path specified by the user.

``config_path`` is the artifact key. It usually corresponds to a model configuration but does not have to.
The model config that is packaged with the .atommic file will be updated according to the ``config_path`` key.

``src`` is the path to the artifact and the base-name of the path will be used when packaging the artifact in the
.atommic file. Each artifact will have a hash prepended to the basename of ``src`` in the .atommic file. This is to
prevent collisions with basenames base-names that are identical (say when there are two or more tokenizers, both
called `tokenizer.model`).

If ``verify_src_exists`` is set to ``False``, then the artifact is optional. This means that ``.register_artifact``
will return ``None`` if the ``src`` cannot be found.

Nested ATOMMIC Models
---------------------

In some cases, it may be helpful to use ATOMMIC models inside other ATOMMIC models. For example, we can incorporate
reconstruction and segmentation models into MTL models to use in a multitask learning setting. In these cases, we can
use the ``register_atommic_submodule`` method to register the child model.

There are 3 ways to instantiate child models inside parent models:

- use subconfig directly
- use the ``.atommic`` checkpoint path to load the child model
- use a pretrained ATOMMIC model

To register a child model, use the ``register_atommic_submodule`` method of the parent model. This method will add the
child model to a provided model attribute and, in the serialization process, will handle child artifacts correctly and
store the child model config in the parent model config in ``config_field``.

.. code-block:: python

    from atommic.core.classes.modelPT import ModelPT

    class ChildModel(ModelPT):
        ...  # implement necessary methods

    class ParentModel(ModelPT):
        def __init__(self, cfg, trainer=None):
            super().__init__(cfg=cfg, trainer=trainer)

            # optionally annotate type for IDE autocompletion and type checking
            self.child_model: Optional[ChildModel]
            if cfg.get("child_model") is not None:
                # load directly from config
                # either if config provided initially, or automatically
                # after model restoration
                self.register_atommic_submodule(
                    name="child_model",
                    config_field="child_model",
                    model=ChildModel(self.cfg.child_model, trainer=trainer),
                )
            elif cfg.get('child_model_path') is not None:
                # load from .atommic model checkpoint
                # while saving, config will be automatically assigned/updated
                # in cfg.child_model
                self.register_atommic_submodule(
                    name="child_model",
                    config_field="child_model",
                    model=ChildModel.restore_from(self.cfg.child_model_path, trainer=trainer),
                )
            elif cfg.get('child_model_name') is not None:
                # load from pretrained model
                # while saving, config will be automatically assigned/updated
                # in cfg.child_model
                self.register_atommic_submodule(
                    name="child_model",
                    config_field="child_model",
                    model=ChildModel.from_pretrained(self.cfg.child_model_name, trainer=trainer),
                )
            else:
                self.child_model = None


Dynamic Layer Freezing
----------------------

You can selectively freeze any modules inside a ATOMMIC model by specifying a freezing schedule in the config yaml.
Freezing stops any gradient updates to that module, so that its weights are not changed for that step. This can be
useful for combatting catastrophic forgetting, for example when finetuning a large pretrained model on a small dataset.

The default approach is to freeze a module for the first N training steps, but you can also enable freezing for a
specific range of steps, for example, from step 20 - 100, or even activate freezing from some N until the end of
training. You can also freeze a module for the entire training run. Dynamic freezing is specified in training steps,
not epochs.

To enable freezing, add the following to your config:

.. code-block:: yaml

  model:
    ...
    freeze_updates:
      enabled: true  # set to false if you want to disable freezing

      modules:   # list all of the modules you want to have freezing logic for
        encoder: 200       # module will be frozen for the first 200 training steps
        decoder: [50, -1]  # module will be frozen at step 50 and will remain frozen until training ends
        joint: [10, 100]   # module will be frozen between step 10 and step 100 (step >= 10 and step <= 100)
        transcoder: -1     # module will be frozen for the entire training run
