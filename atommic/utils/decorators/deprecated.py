# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/decorators/deprecated.py

__all__ = ["deprecated"]

import functools
import inspect
from typing import Dict

import wrapt

# Remember which deprecation warnings have been printed already.
from atommic.utils import logging

_PRINTED_WARNING: Dict = {}


def deprecated(wrapped=None, version=None, explanation=None):
    """This is a decorator which can be used to mark functions as deprecated. It will result in a warning being emitted
    when the function is used.

    Parameters
    ----------
    wrapped : function
        The function to be decorated.
    version : str
        The version of the package where the function was deprecated.
    explanation : str
        The explanation of the deprecation.

    Returns
    -------
    function
        The decorated function.
    """
    if wrapped is None:
        return functools.partial(deprecated, version=version, explanation=explanation)

    @wrapt.decorator
    def wrapper(_wrapped, args, kwargs):
        """Prints the adequate warning (only once per function) when required and calls the function func, passing the
        original arguments, i.e. version and explanation.

        Parameters
        ----------
        _wrapped : function
            The function to be decorated.
        args : tuple
            The arguments passed to the function to be decorated.
        kwargs : dict
            The keyword arguments passed to the function to be decorated.

        Returns
        -------
        function
            The decorated function.
        """
        # Check if we already warned about that function.
        if _wrapped.__name__ not in _PRINTED_WARNING:
            # Add to list, so we won't print it again.
            _PRINTED_WARNING[_wrapped.__name__] = True

            # Prepare the warning message.
            entity_name = "Class" if inspect.isclass(wrapped) else "Function"
            msg = f"{entity_name} '{_wrapped.__name__}' is deprecated."

            # Optionally, add version and explanation.
            if version is not None:
                msg = f"{msg} It is going to be removed in the {version} version."

            if explanation is not None:
                msg = f"{msg} {explanation}"

            # Display the deprecated warning.
            logging.warning(msg)

        # Call the function.
        return _wrapped(*args, **kwargs)

    return wrapper(wrapped)  # pylint: disable=no-value-for-parameter
