Core
====

Base class for all ATOMMIC models
---------------------------------

.. autoclass:: atommic.core.classes.modelPT.ModelPT
    :show-inheritance:
    :members:
    :member-order: bysource
    :undoc-members: cfg, num_weights
    :exclude-members: set_eff_save, use_eff_save, teardown

Base Mixin classes
------------------

.. autoclass:: atommic.core.classes.common.Typing
    :show-inheritance:
    :members:
    :member-order: bysource
    :private-members:
    :exclude-members: _abc_impl
    :noindex:

-----

.. autoclass:: atommic.core.classes.common.Serialization
    :show-inheritance:
    :members:
    :member-order: bysource
    :noindex:

-----

.. autoclass:: atommic.core.classes.common.FileIO
    :show-inheritance:
    :members:
    :member-order: bysource
    :noindex:


Base Connector classes
----------------------

.. autoclass:: atommic.core.connectors.save_restore_connector.SaveRestoreConnector
    :show-inheritance:
    :members:
    :member-order: bysource

Neural Type checking
--------------------

.. autoclass:: atommic.core.classes.common.typecheck
    :show-inheritance:
    :members:
    :member-order: bysource

    .. automethod:: __call__

Neural Type classes
-------------------

.. autoclass:: atommic.core.neural_types.neural_type.NeuralType
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: atommic.core.neural_types.axes.AxisType
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: atommic.core.neural_types.elements.ElementType
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: atommic.core.neural_types.comparison.NeuralTypeComparisonResult
    :show-inheritance:
    :members:
    :member-order: bysource

Experiment manager
------------------

.. autoclass:: atommic.utils.exp_manager.exp_manager
    :show-inheritance:
    :members:
    :member-order: bysource

.. autoclass:: atommic.utils.exp_manager.ExpManagerConfig
    :show-inheritance:
    :members:
    :member-order: bysource


Exportable
----------

.. autoclass:: atommic.core.classes.export.Exportable
    :show-inheritance:
    :members:
    :member-order: bysource
