# coding=utf-8
__author__ = "Dimitris Karkalousos"

from atommic.utils.atommic_logging import Logger as _Logger

logging = _Logger()
try:
    from atommic.utils.lightning_logger_patch import add_memory_handlers_to_pl_logger

    add_memory_handlers_to_pl_logger()
except ModuleNotFoundError:
    pass
