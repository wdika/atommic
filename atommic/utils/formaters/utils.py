# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/formatters/utils.py

import sys

from atommic.constants import ATOMMIC_ENV_VARNAME_ENABLE_COLORING
from atommic.utils.env_var_parsing import get_envbool

__all__ = ["check_color_support", "to_unicode"]


def check_color_support():
    """Checks if the terminal supports color.

    Returns
    -------
    bool
        True if the terminal supports color, False otherwise.
    """
    # Colors can be forced with an env variable
    return bool(not sys.platform.lower().startswith("win") and get_envbool(ATOMMIC_ENV_VARNAME_ENABLE_COLORING, False))


def to_unicode(value):
    """Converts a string to unicode. If the string is already unicode, it is returned as is. If it is a byte string, it
     is decoded using utf-8.

    Parameters
    ----------
    value : str
        The string to convert.

    Returns
    -------
    str
        The converted string.
    """
    try:
        if isinstance(value, (str, type(None))):
            return value

        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, unicode, or None; got %{type(value)}")

        return value.decode("utf-8")

    except UnicodeDecodeError:
        return repr(value)
