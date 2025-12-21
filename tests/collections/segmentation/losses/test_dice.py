# coding=utf-8
__author__ = "Tim Paquaij, Dimitris Karkalousos"

import pytest
import torch
from atommic.collections.segmentation.losses.dice import Dice, GeneralisedDice
from tests.collections.reconstruction.mri_data.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg",
    [
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "to_onehot_y": False,
                "sigmoid": True,
                "softmax": False,
                "flatten": False,
                "reduction": 'none',
                "batch": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": False,
                "to_onehot_y": True,
                "sigmoid": False,
                "softmax": True,
                "flatten": True,
                "reduction": 'none',
                "batch": False,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "to_onehot_y": False,
                "sigmoid": True,
                "softmax": False,
                "flatten": False,
                "reduction": 'mean',
                "batch": True,
                "num_segmentation_classes": 4,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": False,
                "to_onehot_y": True,
                "sigmoid": False,
                "softmax": True,
                "flatten": True,
                "reduction": 'mean',
                "batch": False,
                "num_segmentation_classes": 2,
            },
        ),
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "to_onehot_y": False,
                "sigmoid": True,
                "softmax": False,
                "flatten": False,
                "reduction": 'sum',
                "batch": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": False,
                "to_onehot_y": True,
                "sigmoid": False,
                "softmax": True,
                "flatten": True,
                "reduction": 'sum',
                "batch": False,
                "num_segmentation_classes": None,
            },
        ),
    ],
)
def test_dice_loss(shape, cfg):
    """
    Test Dice Loss

    Parameters
    ----------
    shape : list of int
        Shape of the input data
    cfg : dict
        Dictionary with the parameters of the loss function
    """
    x = create_input(shape)
    y = create_input(shape)
    dice_loss = Dice(
        include_background=cfg.get('include_background'),
        to_onehot_y=cfg.get('to_onehot_y'),
        sigmoid=cfg.get('sigmoid'),
        softmax=cfg.get('softmax'),
        flatten=cfg.get('flatten'),
        reduction=cfg.get('reduction'),
        batch=cfg.get('batch'),
        num_segmentation_classes=cfg.get('num_segmentation_classes'),
    )
    loss = dice_loss(y, x)[1]
    assert isinstance(loss, torch.Tensor)
    if not cfg.get('batch') and cfg.get('reduction') == 'none':
        assert loss.dim() == 2
    else:
        assert loss.dim() <= 1


@pytest.mark.parametrize(
    "shape, cfg",
    [
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "to_onehot_y": False,
                "sigmoid": True,
                "softmax": False,
                "flatten": False,
                "reduction": 'none',
                "batch": True,
                "num_segmentation_classes": 4,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": False,
                "to_onehot_y": True,
                "sigmoid": False,
                "softmax": False,
                "flatten": True,
                "reduction": 'none',
                "batch": False,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "to_onehot_y": False,
                "sigmoid": True,
                "softmax": False,
                "flatten": False,
                "reduction": 'mean',
                "batch": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": False,
                "to_onehot_y": False,
                "sigmoid": False,
                "softmax": True,
                "flatten": True,
                "reduction": 'mean',
                "batch": False,
                "num_segmentation_classes": 2,
            },
        ),
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "to_onehot_y": False,
                "sigmoid": True,
                "softmax": False,
                "flatten": False,
                "reduction": 'sum',
                "batch": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": False,
                "to_onehot_y": True,
                "sigmoid": False,
                "softmax": True,
                "flatten": True,
                "reduction": 'sum',
                "batch": False,
                "num_segmentation_classes": None,
            },
        ),
    ],
)
def test_gendice_loss(shape, cfg):
    """
    Test generalised Dice loss

    Parameters
    ----------
    shape : list of int
        Shape of the input data
    cfg : dict
        Dictionary with the parameters of the loss function
    """
    x = create_input(shape)
    y = create_input(shape)
    gendice_loss = GeneralisedDice(
        include_background=cfg.get('include_background'),
        to_onehot_y=cfg.get('to_onehot_y'),
        sigmoid=cfg.get('sigmoid'),
        softmax=cfg.get('softmax'),
        reduction=cfg.get('reduction'),
        batch=cfg.get('batch'),
        num_segmentation_classes=cfg.get('num_segmentation_classes'),
    )
    loss = gendice_loss(y, x)[1]
    assert isinstance(loss, torch.Tensor)
    if not cfg.get('batch') and cfg.get('reduction') == 'none':
        assert loss.dim() == 2
    else:
        assert loss.dim() <= 1
