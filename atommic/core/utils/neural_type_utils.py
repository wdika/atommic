# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/utils/neural_type_utils.py

from collections import defaultdict


def get_io_names(types, disabled_names):
    """This method will return a list of input and output names for a given NeuralType.

    Parameters
    ----------
    types : type
        The NeuralType of the module or model to be inspected.
    disabled_names : list
        A list of names that should be excluded from the result.

    Returns
    -------
    list
        A list of input and output names.
    """
    names = list(types.keys())
    for name in disabled_names:
        if name in names:
            names.remove(name)
    return names


def get_dynamic_axes(types, names):
    """This method will return a dictionary with input/output names as keys and a list of dynamic axes as values.

    Parameters
    ----------
    types : NeuralType
        The NeuralType of the module or model to be inspected.
    names : list
        A list of names that should be inspected.

    Returns
    -------
    dict
        A dictionary with input/output names as keys and a list of dynamic axes as values.
    """
    dynamic_axes = defaultdict(list)
    if names is not None:
        for name in names:
            if name in types:
                dynamic_axes.update(extract_dynamic_axes(name, types[name]))  # noqa: F821
    return dynamic_axes
