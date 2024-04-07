# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Parts of the code have been taken from: https://github.com/facebookresearch/fastMRI

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from atommic.collections.common.data.subsample import Random1DMaskFunc
from atommic.collections.common.parts import utils
from atommic.collections.reconstruction.nn.recurrentvarnet import RecurrentVarNet
from tests.collections.reconstruction.mri_data.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, trainer",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 8,
                "no_parameter_sharing": True,
                "learned_initializer": True,
                "initializer_initialization": "sense",
                "initializer_channels": [32, 32, 64, 64],
                "initializer_dilations": [1, 1, 2, 4],
                "initializer_multiscale": 1,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
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
            [1, 5, 15, 12, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 8,
                "no_parameter_sharing": False,
                "learned_initializer": True,
                "initializer_initialization": "sense",
                "initializer_channels": [32, 32, 64, 64],
                "initializer_dilations": [1, 1, 2, 4],
                "initializer_multiscale": 1,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
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
        (
            [1, 8, 13, 18, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 8,
                "no_parameter_sharing": False,
                "learned_initializer": False,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "train_reconstruction_loss": "ssim",
                "val_reconstruction_loss": "ssim",
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
            [1, 2, 17, 19, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 8,
                "no_parameter_sharing": True,
                "learned_initializer": False,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
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
            [1, 2, 17, 19, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 18,
                "no_parameter_sharing": True,
                "learned_initializer": False,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
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
    ],
)
def test_recurrentvarnet(shape, cfg, center_fractions, accelerations, dimensionality, trainer):
    """
    Test RecurrentVarNet with different parameters

    Args:
        shape: shape of the input
        cfg: configuration of the model
        center_fractions: center fractions
        accelerations: accelerations
        dimensionality: 2D or 3D inputs
        trainer: trainer configuration

    Returns:
        None
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

    rvn = RecurrentVarNet(cfg, trainer=trainer)

    coil_dim = cfg.coil_dim if dimensionality == 2 else cfg.coil_dim + 1

    with torch.no_grad():
        y = rvn.forward(output, output, mask, output.sum(coil_dim))

        while isinstance(y, list):
            y = y[-1]

    x = torch.view_as_complex(x.sum(coil_dim))

    if y.shape != x.shape:
        raise AssertionError
