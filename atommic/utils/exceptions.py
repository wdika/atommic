# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/exceptions.py


class ATOMMICBaseException(Exception):
    """ATOMMIC Base Exception. All exceptions created in atommic should inherit from this class"""
