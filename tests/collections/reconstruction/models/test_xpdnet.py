# coding=utf-8
__author__ = "Dimitris Karkalousos"

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from atommic.collections.common.data.subsample import Random1DMaskFunc
from atommic.collections.common.parts import utils
from atommic.collections.reconstruction.nn.xpdnet import XPDNet


def create_input(shape):
    """Create a random input tensor."""
    return torch.rand(shape).float()


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, trainer",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "num_primal": 5,
                "num_dual": 5,
                "num_iter": 20,
                "use_primal_only": True,
                "kspace_model_architecture": "CONV",
                "dual_conv_hidden_channels": 16,
                "dual_conv_num_dubs": 2,
                "dual_conv_batchnorm": False,
                "image_model_architecture": "MWCNN",
                "imspace_in_channels": 2,
                "imspace_out_channels": 2,
                "mwcnn_hidden_channels": 16,
                "mwcnn_num_scales": 2,
                "mwcnn_bias": True,
                "mwcnn_batchnorm": False,
                "normalize_image": False,
                "use_sens_net": False,
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "coil_combination_method": "SENSE",
                "reconstruction_loss": {"l1": 1.0},
            },
            [0.08],
            [4],
            2,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
        (
            [1, 3, 32, 16, 2],
            {
                "num_primal": 5,
                "num_dual": 5,
                "num_iter": 20,
                "use_primal_only": True,
                "kspace_model_architecture": "CONV",
                "dual_conv_hidden_channels": 16,
                "dual_conv_num_dubs": 2,
                "dual_conv_batchnorm": False,
                "image_model_architecture": "MWCNN",
                "imspace_in_channels": 2,
                "imspace_out_channels": 2,
                "mwcnn_hidden_channels": 16,
                "mwcnn_num_scales": 2,
                "mwcnn_bias": True,
                "mwcnn_batchnorm": False,
                "normalize_image": False,
                "use_sens_net": False,
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "coil_combination_method": "SENSE",
                "train_reconstruction_loss": "mse",
                "val_reconstruction_loss": "mse",
            },
            [0.08],
            [4],
            2,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
    ],
)
def test_xpdnet(shape, cfg, center_fractions, accelerations, dimensionality, trainer):
    """
    Test the XPDNet model.

    Args:
        shape: shape of the input
        cfg: configuration of the model
        center_fractions: center fractions
        accelerations: accelerations
        dimensionality: 2D or 3D inputs
        trainer: trainer configuration

    Returns:
        None.
    """
    mask_func = Random1DMaskFunc(center_fractions, accelerations)
    x = create_input(shape)

    outputs, masks = [], []
    for i in range(x.shape[0]):
        output, mask, _ = utils.apply_mask(x[i : i + 1], mask_func, seed=123)
        outputs.append(output)
        masks.append(mask)

    output = torch.cat(outputs)
    mask = torch.cat(masks)

    if dimensionality == 3 and shape[1] > 1:
        mask = torch.cat([mask, mask], 1)

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    trainer = OmegaConf.create(trainer)
    trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
    trainer = pl.Trainer(**trainer)

    xpdnet = XPDNet(cfg, trainer=trainer)

    coil_dim = cfg.coil_dim if dimensionality == 2 else cfg.coil_dim + 1

    with torch.no_grad():
        y = xpdnet.forward(output, output, mask, output.sum(coil_dim))

        while isinstance(y, list):
            y = y[-1]

    x = x.sum(coil_dim)

    if y.shape != x.shape:
        raise AssertionError
