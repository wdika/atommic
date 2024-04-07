# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/formatters/colors.py

CSI = "\033["


def code_to_chars(code):
    """Convert ANSI color code to string of characters.

    Parameters
    ----------
    code : int
        ANSI color code.

    Returns
    -------
    str
        String of characters.
    """
    return CSI + str(code) + "m"


class AnsiCodes:
    """ANSI color codes."""

    def __init__(self):
        """Inits :class:`AnsiCodes`."""
        # The subclasses declare class attributes which are numbers. Upon instantiation, we define instance attributes,
        # which are the same as the class attributes but wrapped with the ANSI escape sequence
        for name in dir(self):
            if not name.startswith("_"):
                value = getattr(self, name)
                setattr(self, name, code_to_chars(value))


class AnsiFore(AnsiCodes):
    """ANSI color codes for foreground text."""

    RED = 31
    GREEN = 32
    YELLOW = 33
    MAGENTA = 35
    CYAN = 36
    RESET = 39


Fore = AnsiFore()
