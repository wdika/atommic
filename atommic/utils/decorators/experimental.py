# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/decorators/experimental.py

import wrapt

from atommic.utils import logging

__all__ = ["experimental"]


@wrapt.decorator
def experimental(wrapped, instance, args, kwargs):  # pylint: disable=unused-argument
    """Decorator to mark a class as experimental.

    Parameters
    ----------
    wrapped : function
        The function to be decorated.
    instance : object
        The instance of the class to be decorated.
    args : tuple
        The arguments passed to the function to be decorated.
    kwargs : dict
        The keyword arguments passed to the function to be decorated.
    """
    logging.warning(f"`{wrapped}` is experimental and not ready for production yet. Use at your own risk.")
    return wrapped(*args, **kwargs)
