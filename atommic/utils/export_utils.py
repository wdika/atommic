# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/export_utils.py
import os
from enum import Enum
from typing import Callable, Dict, Optional

import onnx
import torch
from torch import nn

from atommic.utils import logging

try:
    import onnxruntime

    ort_available = True
except ImportError:
    ort_available = False


class ExportFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""

    ONNX = (1,)
    TORCHSCRIPT = (2,)


_EXT_DICT = {".pt": ExportFormat.TORCHSCRIPT, ".ts": ExportFormat.TORCHSCRIPT, ".onnx": ExportFormat.ONNX}


def get_export_format(filename: str):
    """Get export format from filename."""
    _, ext = os.path.splitext(filename)
    try:
        return _EXT_DICT[ext.lower()]
    except KeyError as e:
        raise ValueError(f"Export file {filename} extension does not correspond to any export format!") from e


def augment_filename(output: str, prepend: str):
    """Augment output filename with prepend"""
    if prepend == "self":
        return output
    path, filename = os.path.split(output)
    filename = f"{prepend}-{filename}"
    return os.path.join(path, filename)


def forward_method(self):
    """Forward method for export"""
    if hasattr(self, "forward_for_export"):
        return self.forward_for_export
    return self.forward


def wrap_forward_method(self):
    """Wraps the forward method of the module with a function that returns the output of the forward method"""
    tp = type(self)
    old_forward_method = None
    if hasattr(tp, "forward_for_export"):
        forward_method = tp.forward_for_export
        old_forward_method = tp.forward
        tp.forward = forward_method
    else:
        forward_method = None
    return forward_method, old_forward_method


def parse_input_example(input_example):
    """Parse input example to onnxrt input format"""
    input_list = list(input_example)
    input_dict = {}
    # process possible kwargs
    if isinstance(input_list[-1], dict):
        input_dict = input_list[-1]
        input_list = input_list[:-1]
    return input_list, input_dict


def to_onnxrt_input(ort_input_names, input_names, input_dict, input_list):
    """Convert input to onnxrt input"""
    odict = {}
    for k in reversed(input_names):
        val = None
        if k in input_dict:
            val = input_dict[k].cpu().numpy()
        elif len(input_list) > 0:
            val = input_list.pop().cpu().numpy()
        if k in ort_input_names and val is not None:
            odict[k] = val
    return odict


def verify_torchscript(model, output, input_examples, check_tolerance=0.01):
    """Verify torchscript output with torchscript forward.

    Parameters
    ----------
    model : torch.nn.Module
        Model to verify.
    output : str
        Output filename.
    input_examples : list
        List of input examples.
    check_tolerance : float
        Tolerance for checking.

    Returns
    -------
    bool
        Whether the verification was successful.
    """
    all_good = True
    for input_example in input_examples:
        input_list, input_dict = parse_input_example(input_example)
        # We disable autocast here to make sure exported TS will run under Triton or other C++ env
        with torch.cuda.amp.autocast(enabled=False):
            output_example = model.forward(*input_list, **input_dict)
            ts_model = torch.jit.load(output)
            all_good = all_good and run_ts_and_compare(
                ts_model, input_list, input_dict, output_example, check_tolerance
            )
    status = "SUCCESS" if all_good else "FAIL"
    logging.info(f"Torchscript generated at {output} verified with torchscript forward : " + status)
    return all_good


# pylint: disable=inconsistent-return-statements
def verify_runtime(
    model,
    output,
    input_examples,
    input_names,
    check_tolerance=0.01,
):
    """Verify runtime output with onnxrt."""
    onnx_model = onnx.load(output)
    ort_input_names = [node.name for node in onnx_model.graph.input]

    global ort_available  # pylint: disable=global-variable-not-assigned
    if not ort_available:
        logging.warning(f"ONNX generated at {output}, not verified - please install onnxruntime_gpu package.\n")
        onnx.checker.check_model(onnx_model, full_check=True)
        return

    onnx_session_opt = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), sess_options=onnx_session_opt, providers=["CUDAExecutionProvider"]
    )

    all_good = True

    for input_example in input_examples:
        input_list, input_dict = parse_input_example(input_example)
        output_example = model.forward(*input_list, **input_dict)
        ort_input = to_onnxrt_input(ort_input_names, input_names, input_dict, input_list)
        all_good = all_good and run_ort_and_compare(sess, ort_input, output_example, check_tolerance)
    status = "SUCCESS" if all_good else "FAIL"
    logging.info(f"ONNX generated at {output} verified with onnxruntime : {status}")
    return all_good


def run_ts_and_compare(ts_model, ts_input_list, ts_input_dict, output_example, check_tolerance=0.01):
    """Run torchscript model and compare with pytorch output."""
    # Verify the model can be read, and is valid
    ts_out = ts_model(*ts_input_list, **ts_input_dict)
    all_good = True
    for i, out in enumerate(ts_out):
        expected = output_example[i]
        if torch.is_tensor(expected):
            tout = out.to("cpu")
            logging.debug(f"Checking output {i}, shape: {expected.shape}:\n")
            this_good = True
            try:
                if not torch.allclose(tout, expected.cpu(), rtol=check_tolerance, atol=check_tolerance):
                    this_good = False
            except Exception:  # there maybe size mismatch and it may be OK
                this_good = False
            if not this_good:
                logging.info(f"Results mismatch! PyTorch(expected):\n{expected}\nTorchScript:\n{tout}")
                all_good = False
    return all_good


def run_ort_and_compare(sess, ort_input, output_example, check_tolerance=0.01):
    """Run onnxrt and compare with output example"""
    ort_out = sess.run(None, ort_input)
    all_good = True
    for i, out in enumerate(ort_out):
        expected = output_example[i]
        if torch.is_tensor(expected):
            tout = torch.from_numpy(out)
            logging.debug(f"Checking output {i}, shape: {expected.shape}:\n")
            this_good = True
            try:
                if not torch.allclose(tout, expected.cpu(), rtol=check_tolerance, atol=100 * check_tolerance):
                    this_good = False
            except Exception:  # there maybe size mismatch and it may be OK
                this_good = False
            if not this_good:
                logging.info(f"onnxruntime results mismatch! PyTorch(expected):\n{expected}\nONNXruntime:\n{tout}")
                all_good = False
    return all_good


def swap_modules(model: nn.Module, mapping: Dict[str, nn.Module]):
    """This function swaps nested modules as specified by "dot paths" in mod with a desired replacement. This allows
    for swapping nested modules through arbitrary levels if children.

    note::
        This occurs in place, if you want to preserve model then make sure to copy it first.
    """
    for path, new_mod in mapping.items():
        expanded_path = path.split(".")
        parent_mod = model
        for sub_path in expanded_path[:-1]:
            parent_mod = parent_mod._modules[sub_path]  # pylint: disable=protected-access
        parent_mod._modules[expanded_path[-1]] = new_mod  # pylint: disable=protected-access

    return model


def replace_modules(
    model: nn.Module, expansions: Dict[str, Callable[[nn.Module], Optional[nn.Module]]] = None
) -> nn.Module:
    """Top-level function to replace modules in model, specified by class name with a desired replacement.

    note::
        This occurs in place, if you want to preserve model then make sure to copy it first.

    Parameters
    ----------
    model : nn.Module
        Top-level model to replace modules in.
    expansions : Dict[str, Callable[[nn.Module], Optional[nn.Module]]]
        A dictionary of module class names to functions to replace them with.

    Returns
    -------
    nn.Module
        The model with replaced modules.
    """
    mapping: Dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        m_type = type(m).__name__
        if m_type in expansions:  # type: ignore
            if swapped := expansions[m_type](m):  # type: ignore
                mapping[name] = swapped
    logging.warning(f"Swapped {len(mapping)} modules")
    swap_modules(model, mapping)
    return model


script_replacements: Dict = {}


def replace_for_export(model: nn.Module) -> nn.Module:
    """Top-level function to replace default set of modules in model

    note::
        This occurs in place, if you want to preserve model then make sure to copy it first.

    Parameters
    ----------
    model : nn.Module
        Top-level model to replace modules in.

    Returns
    -------
    nn.Module
        The model with replaced modules.
    """
    replace_modules(model, script_replacements)
