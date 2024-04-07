# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/exportable.py

from abc import ABC
from typing import List, Union

import torch
from pytorch_lightning.core.module import _jit_is_scripting
from torch.onnx import TrainingMode

from atommic.core.classes.common import typecheck
from atommic.core.utils.neural_type_utils import get_dynamic_axes, get_io_names
from atommic.utils import logging
from atommic.utils.export_utils import (
    ExportFormat,
    augment_filename,
    get_export_format,
    parse_input_example,
    replace_for_export,
    verify_runtime,
    verify_torchscript,
    wrap_forward_method,
)

__all__ = ["ExportFormat", "Exportable"]


class Exportable(ABC):
    """This Interface should be implemented by particular classes derived from atommic.core.ModelPT. It gives these
    entities ability to be exported for deployment to formats such as ONNX.
    """

    @property
    def input_module(self):
        """Implement this method to return the input module"""
        return self

    @property
    def output_module(self):
        """Implement this method to return the output module."""
        return self

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
        export_modules_as_functions: bool = False,
        keep_initializers_as_inputs=None,
    ):
        """Export the module to a file.

        Parameters
        ----------
        output : str
            The output file path.
        input_example : dict
            A dictionary of input names and values.
        verbose : bool
            If True, print out the export process.
        do_constant_folding : bool
            If True, do constant folding.
        onnx_opset_version : int
            The ONNX opset version to use.
        check_trace : bool or list of torch.Tensor
            If True, check the trace of the exported model.
        dynamic_axes : dict
            A dictionary of input names and dynamic axes.
        check_tolerance : float
            The tolerance for the check_trace.
        export_modules_as_functions : bool
            If True, export modules as functions.
        keep_initializers_as_inputs : bool
            If True, keep initializers as inputs.
        """
        all_out = []
        all_descr = []
        for subnet_name in self.list_export_subnets():
            model = self.get_export_subnet(subnet_name)
            out_name = augment_filename(output, subnet_name)
            out, descr, out_example = model._export(  # pylint: disable=protected-access
                out_name,
                input_example=input_example,
                verbose=verbose,
                do_constant_folding=do_constant_folding,
                onnx_opset_version=onnx_opset_version,
                check_trace=check_trace,
                dynamic_axes=dynamic_axes,
                check_tolerance=check_tolerance,
                export_modules_as_functions=export_modules_as_functions,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
            )
            # Propagate input example (default scenario, may need to be overriden)
            if input_example is not None:
                input_example = out_example
            all_out.append(out)
            all_descr.append(descr)
            logging.info(f"Successfully exported {model.__class__.__name__} to {out_name}")
        return all_out, all_descr

    def _export(
        self,
        output: str,
        input_example=None,
        verbose=False,
        do_constant_folding=True,
        onnx_opset_version=None,
        training=TrainingMode.EVAL,  # pylint: disable=unused-argument
        check_trace: Union[bool, List[torch.Tensor]] = False,
        dynamic_axes=None,
        check_tolerance=0.01,
        export_modules_as_functions: bool = False,
        keep_initializers_as_inputs=None,
    ):
        """Helper to export the module to a file.

        Parameters
        ----------
        output : str
            The output file path.
        input_example : dict
            A dictionary of input names and values.
        verbose : bool
            If True, print out the export process.
        do_constant_folding : bool
            If True, do constant folding.
        onnx_opset_version : int
            The ONNX opset version to use.
        training : TrainingMode
            Training mode for the export.
        check_trace : bool or list of torch.Tensor
            If True, check the trace of the exported model.
        dynamic_axes : dict
            A dictionary of input names and dynamic axes.
        check_tolerance : float
            The tolerance for the check_trace.
        export_modules_as_functions : bool
            If True, export modules as functions.
        keep_initializers_as_inputs : bool
            If True, keep initializers as inputs.
        """
        my_args = locals().copy()
        my_args.pop("self")

        self.eval()  # type: ignore
        for param in self.parameters():  # type: ignore
            param.requires_grad = False

        exportables = [m for m in self.modules() if isinstance(m, Exportable)]  # type: ignore
        qual_name = f"{self.__module__}.{self.__class__.__qualname__}"
        exp_format = get_export_format(output)
        output_descr = f"{qual_name} exported to {exp_format}"

        # Pytorch's default opset version is too low, using reasonable latest one
        if onnx_opset_version is None:
            onnx_opset_version = 16

        try:
            # Disable typechecks
            typecheck.set_typecheck_enabled(enabled=False)

            # Allow user to completely override forward method to export
            forward_method, old_forward_method = wrap_forward_method(self)

            with torch.inference_mode(), torch.no_grad(), torch.jit.optimized_execution(True), _jit_is_scripting():
                if input_example is None:
                    input_example = self.input_module.input_example()

                # Remove i/o examples from args we propagate to enclosed Exportables
                my_args.pop("output")
                my_args.pop("input_example")

                # Run (possibly overridden) prepare methods before calling forward()
                for ex in exportables:
                    ex._prepare_for_export(**my_args, noreplace=True)  # pylint: disable=protected-access
                self._prepare_for_export(output=output, input_example=input_example, **my_args)

                input_list, input_dict = parse_input_example(input_example)
                input_names = self.input_names
                output_names = self.output_names
                output_example = tuple(self.forward(*input_list, **input_dict))  # type: ignore

                if check_trace:
                    if isinstance(check_trace, bool):
                        check_trace_input = [input_example]
                    else:
                        check_trace_input = check_trace
                jitted_model = self
                if exp_format == ExportFormat.TORCHSCRIPT:
                    jitted_model = torch.jit.trace_module(
                        self,
                        {"forward": tuple(input_list) + tuple(input_dict.values())},
                        strict=True,
                        check_trace=check_trace,
                        check_tolerance=check_tolerance,
                    )
                    jitted_model = torch.jit.freeze(jitted_model)
                    if verbose:
                        logging.info(f"JIT code:\n{jitted_model.code}")  # type: ignore
                    jitted_model.save(output)  # type: ignore
                    jitted_model = torch.jit.load(output)

                    if check_trace:
                        verify_torchscript(jitted_model, output, check_trace_input, check_tolerance)
                elif exp_format == ExportFormat.ONNX:
                    # dynamic axis is a mapping from input/output_name => list of "dynamic" indices
                    if dynamic_axes is None:
                        dynamic_axes = get_dynamic_axes(self.input_module.input_types_for_export, input_names)
                        dynamic_axes.update(get_dynamic_axes(self.output_module.output_types_for_export, output_names))
                    torch.onnx.export(
                        jitted_model,
                        input_example,
                        output,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=verbose,
                        do_constant_folding=do_constant_folding,
                        dynamic_axes=dynamic_axes,
                        opset_version=onnx_opset_version,
                        keep_initializers_as_inputs=keep_initializers_as_inputs,
                        export_modules_as_functions=export_modules_as_functions,
                    )

                    if check_trace:
                        verify_runtime(self, output, check_trace_input, input_names, check_tolerance=check_tolerance)
                else:
                    raise ValueError(f"Encountered unknown export format {exp_format}.")
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
            if forward_method:  # pylint: disable=used-before-assignment
                type(self).forward = old_forward_method  # type: ignore  # pylint: disable=used-before-assignment
            self._export_teardown()
        return (output, output_descr, output_example)

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set()

    @property
    def disabled_deployment_output_names(self):
        """Implement this method to return a set of output names disabled for export"""
        return set()

    @property
    def supported_export_formats(self):
        """Implement this method to return a set of export formats supported. Default is all types."""
        return {ExportFormat.ONNX, ExportFormat.TORCHSCRIPT}

    def _prepare_for_export(self, **kwargs):
        """Override this method to prepare module for export. This is in-place operation. Base version does common
        necessary module replacements (Apex etc)
        """
        if "noreplace" not in kwargs:
            replace_for_export(self)

    def _export_teardown(self):
        """Override this method for any teardown code after export."""

    @property
    def input_names(self):
        """Implement this method to return a list of input names"""
        return get_io_names(self.input_module.input_types_for_export, self.disabled_deployment_input_names)

    @property
    def output_names(self):
        """Override this method to return a set of output names disabled for export"""
        return get_io_names(self.output_module.output_types_for_export, self.disabled_deployment_output_names)

    @property
    def input_types_for_export(self):
        """Implement this method to return a list of input types"""
        return self.input_types

    @property
    def output_types_for_export(self):
        """Implement this method to return a list of output types"""
        return self.output_types

    def get_export_subnet(self, subnet=None):
        """Returns Exportable subnet model/module to export"""
        return self if subnet is None or subnet == "self" else getattr(self, subnet)

    @staticmethod
    def list_export_subnets():
        """Returns default set of subnet names exported for this model. First goes the one receiving input
        (input_example).
        """
        return ["self"]

    def get_export_config(self):
        """Returns export_config dictionary."""
        return getattr(self, 'export_config', {})

    def set_export_config(self, args):
        """Sets/updates export_config dictionary."""
        ex_config = self.get_export_config()
        ex_config.update(args)
        self.export_config = ex_config
