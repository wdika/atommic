# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/tree/main/nemo/utils/cast_utils.py

import torch


def cast_tensor(x, from_dtype=torch.float16, to_dtype=torch.float32):
    """Cast a tensor from one dtype to another if it is of the specified dtype."""
    return x.to(dtype=to_dtype) if x.dtype == from_dtype else x


# pylint: disable=inconsistent-return-statements
def cast_all(x, from_dtype=torch.float16, to_dtype=torch.float32):
    """Cast all tensors in a dict or tuple from one dtype to another if they are of the specified dtype."""
    if isinstance(x, torch.Tensor):
        return cast_tensor(x, from_dtype=from_dtype, to_dtype=to_dtype)
    if isinstance(x, dict):
        new_dict = {}
        for k in x.keys():
            new_dict[k] = cast_all(x[k], from_dtype=from_dtype, to_dtype=to_dtype)
        return new_dict
    if isinstance(x, tuple):
        return tuple(cast_all(y, from_dtype=from_dtype, to_dtype=to_dtype) for y in x)
