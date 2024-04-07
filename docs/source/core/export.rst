Exporting ATOMMIC Models
========================

Exporting Models
----------------

Most of the ATOMMIC models can be exported to ONNX or TorchScript to be deployed for inference in optimized execution
environments. Export interface is provided by the :class:`~atommic.core.classes.export.Exportable` mix-in class. If a
model extends :class:`~atommic.core.classes.exportable.Exportable`, it can be exported by:

.. code-block:: Python

   from atommic.core.classes import ModelPT, Exportable
   # deriving from Exportable
   class MyExportableModel(ModelPT, Exportable):
   ...

   mymodel = MyExportableModel.from_pretrained(model_name="MyModelName")
   model.eval()
   model.to('cuda')  # or to('cpu') if you don't have GPU

   # exporting pre-trained model to ONNX file for deployment.
   mymodel.export('mymodel.onnx', [options])


How to Use Model Export
-----------------------
The following arguments are for :meth:`~atommic.core.classes.export.Exportable.export`. In most cases, you should only
supply the name of the output file and use all defaults:

.. code-block:: Python

    def export(
        self,
        output: str,
        input_example=None,
        verbose=False,
        do_constant_folding=True,
        onnx_opset_version=None,
        check_trace: Union[bool, List[torch.Tensor]] = False,
        dynamic_axes=None,
        check_tolerance=0.01,
        export_modules_as_functions=False,
        keep_initializers_as_inputs=None,
    ):


The ``output``, ``input_example``, ``verbose``, ``do_constant_folding``, ``onnx_opset_version`` options have the same
semantics as in Pytorch ``onnx.export()`` and ``jit.trace()`` functions and are passed through. For more information
about Pytorch's``onnx.export()``, refer to the `torch.onnx functions documentation
<https://pytorch.org/docs/stable/onnx.html#functions>`_. Note that if ``input_example`` is None,
``Exportable.input_example()`` is called.

The file extension of the ``output`` parameter determines export format:

* ``.onnx->ONNX``
* ``.pt`` or ``.ts`` -> ``TorchScript``.

**TorchScript-specific**: By default, the module will undergo ``jit.trace()``. You may require to explicitly pass some
modules under ``jit.script()`` so that they are correctly traced.The ``check_trace`` arg is passed through to
``jit.trace()``.

**ONNX-specific**: If ``use_dynamic_axes`` is True, ``onnx.export()`` is called with dynamic axes. If ``dynamic_axes``
is ``None``, they are inferred from the model's ``input_types`` definition (batch dimension is dynamic, and so is
duration etc).

If ``check_trace`` is ``True``, the resulting ONNX also runs on ``input_example`` and the results compared to the
exported model's output, using the ``check_tolerance`` argument. Note the higher tolerance default.


How to Make Model Exportable
----------------------------

If you are simply using ATOMMIC models, the previous example is all you need to know.
If you write your own models, this section highlights the things you need to be aware of after extending ``Exportable``.

Exportable Hooks and Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should not normally need to override ``Exportable`` default methods. However, ``Exportable.export()`` relies on
the assumptions that certain methods are available in your class.

.. code-block:: Python

    @property
    def input_example(self) # => Tuple(input, [(input, ...], [Dict])
         """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
	 """

This function should return a tuple of (normally) Tensors - one per each of model inputs (args to ``forward()``). The
last element may be a ``Dict`` to specify non-positional arguments by name, as per Torch ``export()`` convention. For
more information, refer to the `Using dictionaries to handle Named Arguments as model inputs
<https://pytorch.org/docs/stable/onnx.html#using-dictionaries-to-handle-named-arguments-as-model-inputs>`_.

.. Note: ``Dict`` currently does not work with Torchscript ``trace()``.

.. code-block:: Python

    @property
    def input_types(self):
    @property
    def output_types(self):

Those are needed for inferring in/out names and dynamic axes. If your model derives from ``ModulePT``, those are
already there. Another common scenario is that your model contains one or more modules that processes input and
generates output. Then, you should override ``Exportable`` methods ``input_module()`` and ``output_module()`` to point
to them, like in this example:

.. code-block:: Python

    @property
    def input_module(self):
        return self.fastpitch

    @property
    def output_module(self):
        return self.fastpitch

Your model should also have an export-friendly ``forward()`` method - that can mean different things for ONNX ant
TorchScript. For ONNX, you can't have forced named parameters without default, like ``forward(self, *, text)``. For
TorchScript, you should avoid ``None`` and use ``Optional`` instead. The criteria are highly volatile and may change
with every PyTorch version, so it's a trial-and-error process. There is also the general issue that in many cases,
``forward()`` for inference can be simplified and even use less inputs/outputs. To address this, ``Exportable`` looks
for ``forward_for_export()`` method in your model and uses that instead of ``forward()`` to export.

To stay consistent with input_types()/output_types(), there are also those hooks in ``Exportable`` that let you
exclude particular inputs/outputs from the export process.

Another common requirement for models that are being exported is to run certain net modifications for inference
efficiency before exporting - like disabling masks in some convolutions or removing batch normalizations. A better
style is to make those happen on ``ModelPT.eval()`` (and reversed on ``.train()``), but it's not always feasible so
the following hook is provided in ``Exportable`` to run those:

.. code-block:: Python

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
    # do graph modifications specific for this model
        normalization_type = kwargs.get('normalization_type', 'minmax')
        replace_for_export(self, normalization_type)
    # call base method for common set of modifications
	Exportable._prepare_for_export(self, **kwargs)

Some models that require control flow, need to be exported in multiple parts. Typical examples are RNNT nets.
To facilitate that, the hooks below are provided. To export, for example, 'encoder' and 'decoder' subnets of the
model, overload list_export_subnets to return ['encoder', 'decoder'].

.. code-block:: Python

    def get_export_subnet(self, subnet=None):
        """
        Returns Exportable subnet model/module to export
        """


    def list_export_subnets(self):
        """
        Returns default set of subnet names exported for this model
        First goes the one receiving input (input_example)
        """

Some networks may be exported differently according to user-settable options. To facilitate that
- `set_export_config()` method is provided by Exportable to set key/value pairs to predefined model.export_config
dictionary, to be used during the export:

.. code-block:: Python

    def set_export_config(self, args):
        """
        Sets/updates export_config dictionary
        """

Also, if an action hook on setting config is desired, this method may be overloaded by `Exportable` descendants to
include one.


Exportable Model Code
~~~~~~~~~~~~~~~~~~~~~

Most importantly, the actual Torch code in your model should be ONNX or TorchScript - compatible (ideally, both).
#. Ensure the code is written in Torch - avoid bare `Numpy or Python operands <https://pytorch.org/docs/stable/onnx.html#write-pytorch-model-in-torch-way>`_.
#. Create your model ``Exportable`` and add an export unit test, to catch any operation/construct not supported in
ONNX/TorchScript, immediately.

For more information, refer to the PyTorch documentation:
       - `List of supported operators <https://pytorch.org/docs/stable/onnx.html#supported-operators>`_
       - `Tracing vs. scripting <https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting>`_
       - `AlexNet example <https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-onnx>`_
