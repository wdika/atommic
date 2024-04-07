# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/loss.py

import torch

from atommic.core.classes.common import Serialization, Typing

__all__ = ["Loss"]


class Loss(torch.nn.modules.loss._Loss, Typing, Serialization):  # pylint: disable=protected-access
    """Inherit this class to implement custom loss."""
