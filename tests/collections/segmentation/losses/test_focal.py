# coding=utf-8
__author__ = "Tim Paquaij, Dimitris Karkalousos"
import pytest
import torch

from atommic.collections.segmentation.losses.focal import FocalLoss
from tests.collections.reconstruction.mri_data.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg",
    [
        (
            [2, 10, 10],
            {
                "include_background": True,
                "reduction": 'none',
                "weight": None,
                "to_onehot_y": True,
                "use_softmax": True,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "reduction": 'none',
                "weight": [1.0, 1.0, 1.0, 1.0],
                "to_onehot_y": False,
                "use_softmax": False,
                "num_segmentation_classes": 4,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": True,
                "reduction": 'mean',
                "weight": None,
                "to_onehot_y": False,
                "use_softmax": True,
                "num_segmentation_classes": 2,
            },
        ),
        (
            [2, 4, 10, 10],
            {
                "include_background": True,
                "reduction": 'mean',
                "weight": [1.0, 1.0, 1.0, 1.0],
                "to_onehot_y": False,
                "use_softmax": False,
                "num_segmentation_classes": None,
            },
        ),
        (
            [2, 10, 10],
            {
                "include_background": True,
                "reduction": 'sum',
                "weight": None,
                "to_onehot_y": True,
                "use_softmax": True,
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
                "use_softmax": False,
                "num_segmentation_classes": 4,
            },
        ),
    ],
)
def test_focal_loss(shape, cfg):
    """
    Test focal loss

    Parameters
    ----------
    shape : list of int
        Shape of the input data
    cfg : dict
        Dictionary with the parameters of the loss function
    """
    x = create_input(shape).requires_grad_()
    y = create_input(shape)
    focal_loss = FocalLoss(
        include_background=cfg.get('include_background'),
        reduction=cfg.get('reduction'),
        weight=cfg.get('weight'),
        to_onehot_y=cfg.get('to_onehot_y'),
        num_segmentation_classes=cfg.get('num_segmentation_classes'),
    )
    loss = focal_loss(y, x)[1]
    assert isinstance(loss, torch.Tensor)
    if cfg.get('reduction') == 'none':
        assert loss.dim() == y.dim()
    else:
        assert loss.dim() <= 1
