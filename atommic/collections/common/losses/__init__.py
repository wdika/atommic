# coding=utf-8
__author__ = "Dimitris Karkalousos"

from atommic.collections.common.losses.aggregator import AggregatorLoss  # noqa: F401
from atommic.collections.common.losses.wasserstein import SinkhornDistance  # noqa: F401

VALID_RECONSTRUCTION_LOSSES = ["l1", "mse", "ssim", "noise_aware", "wasserstein"]
VALID_SEGMENTATION_LOSSES = [
    "categorical_cross_entropy",
    "dice",
    "binary_cross_entropy",
    "generalized_dice",
    "focal_loss",
]
