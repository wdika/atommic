# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/neural_types/elements.py

from abc import ABC, ABCMeta
from typing import Dict, Optional, Tuple

from atommic.core.neural_types.comparison import NeuralTypeComparisonResult

__all__ = ["ElementType", "LossType", "VoidType"]


class ElementType(ABC):
    """Abstract class defining semantics of the tensor elements. We are relying on Python for inheritance checking"""

    def __str__(self):
        """Override this method to provide a human readable representation of the type"""
        return self.__doc__

    def __repr__(self):
        """Override this method to provide a human readable representation of the type"""
        return self.__class__.__name__

    @property
    def type_parameters(self) -> Dict:
        """Override this property to parametrize your type. For example, you can specify 'storage' type such as float,
        int, bool with 'dtype' keyword. Another example, is if you want to represent a signal with a particular
        property (say, sample frequency), then you can put sample_freq->value in there. When two types are compared
        their type_parameters must match."
        """
        return {}

    @property
    def fields(self) -> Optional[Tuple]:
        """This should be used to logically represent tuples/structures. For example, if you want to represent a \
        bounding box (x, y, width, height) you can put a tuple with names ('x', y', 'w', 'h') in here. Under the \
        hood this should be converted to the last tensor dimension of fixed size = len(fields). When two types are \
        compared their fields must match."""
        return None

    def compare(self, second) -> NeuralTypeComparisonResult:
        """Override this method to provide a comparison between two types."""
        # First, check general compatibility
        first_t = type(self)
        second_t = type(second)

        if first_t == second_t:
            result = NeuralTypeComparisonResult.SAME
        elif issubclass(first_t, second_t):
            result = NeuralTypeComparisonResult.LESS
        elif issubclass(second_t, first_t):
            result = NeuralTypeComparisonResult.GREATER
        else:
            result = NeuralTypeComparisonResult.INCOMPATIBLE

        if result != NeuralTypeComparisonResult.SAME:
            return result
        # now check that all parameters match
        check_params = set(self.type_parameters.keys()) == set(second.type_parameters.keys())
        if not check_params:
            return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
        for k1, v1 in self.type_parameters.items():
            if v1 is None or second.type_parameters[k1] is None:
                # Treat None as Void
                continue
            if v1 != second.type_parameters[k1]:
                return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
                # check that all fields match
        if self.fields == second.fields:
            return NeuralTypeComparisonResult.SAME
        return NeuralTypeComparisonResult.INCOMPATIBLE


class VoidType(ElementType):
    """
    Void-like type which is compatible with everything. It is a good practice to use this type only as necessary.
    For example, when you need template-like functionality.
    """

    def compare(cls, second: ABCMeta) -> NeuralTypeComparisonResult:  # pylint: disable=arguments-renamed
        """Void type is compatible with everything."""
        return NeuralTypeComparisonResult.SAME


class LossType(ElementType):
    """Element type to represent outputs of Loss modules"""
