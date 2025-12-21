# coding=utf-8
__author__ = "Tim Paquaij, Dimitris Karkalousos"
import pytest
import torch

from atommic.collections.segmentation.losses.cross_entropy import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
from tests.collections.reconstruction.mri_data.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg",
    [
        (
            [2, 10, 10],
            {
                "include_background": True,
                "reduction": 'mean',
                "weight": None,
                "to_onehot_y": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [4, 10, 10],
            {
                "include_background": True,
                "reduction": 'mean',
                "weight": [1.0, 1.0, 1.0, 1.0],
                "to_onehot_y": False,
                "num_segmentation_classes": 4,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": True,
                "reduction": 'sum',
                "weight": None,
                "to_onehot_y": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "reduction": 'sum',
                "weight": [1.0, 1.0, 1.0, 1.0],
                "to_onehot_y": False,
                "num_segmentation_classes": None,
            },
        ),
    ],
)
def test_CCE_loss(shape, cfg):
    """
    Test Categorical Cross-Entropy Loss

    Parameters
    ----------
    shape : list of int
        Shape of the input data
    cfg : dict
        Dictionary with the parameters of the loss function
    """
    x = create_input(shape).requires_grad_()
    y = create_input(shape)
    ce_loss = CategoricalCrossEntropyLoss(
        include_background=cfg.get('include_background'),
        reduction=cfg.get('reduction'),
        weight=cfg.get('weight'),
        to_onehot_y=cfg.get('to_onehot_y'),
        num_segmentation_classes=cfg.get('num_segmentation_classes'),
    )
    loss = ce_loss(y, x)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() <= 1


@pytest.mark.parametrize(
    "shape, cfg",
    [
        (
            [2, 10, 10],
            {
                "include_background": True,
                "reduction": 'mean',
                "weight": None,
                "to_onehot_y": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [4, 10, 10],
            {
                "include_background": True,
                "reduction": 'mean',
                "weight": [1.0, 1.0, 1.0, 1.0],
                "to_onehot_y": False,
                "num_segmentation_classes": 4,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": True,
                "reduction": 'sum',
                "weight": None,
                "to_onehot_y": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "reduction": 'sum',
                "weight": [1.0, 1.0, 1.0, 1.0],
                "to_onehot_y": False,
                "num_segmentation_classes": None,
            },
        ),
    ],
)
def test_BCE_loss(shape, cfg):
    """
    Test Binary Cross-Entropy Loss

    Parameters
    ----------
    shape : list of int
        Shape of the input data
    cfg : dict
        Dictionary with the parameters of the loss function
    """
    x = create_input(shape).requires_grad_()
    y = create_input(shape)
    ce_loss = BinaryCrossEntropyLoss(
        include_background=cfg.get('include_background'),
        reduction=cfg.get('reduction'),
        weight=cfg.get('weight'),
        to_onehot_y=cfg.get('to_onehot_y'),
        num_segmentation_classes=cfg.get('num_segmentation_classes'),
    )
    loss = ce_loss(y.float(), x)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() <= 1
