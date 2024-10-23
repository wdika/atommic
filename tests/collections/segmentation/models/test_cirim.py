# coding=utf-8
__author__ = "Dimitris Karkalousos"

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from atommic.collections.segmentation.nn.cirim import SegmentationCIRIM
from tests.collections.reconstruction.mri_data.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, segmentation_classes, trainer",
    [
        (
            [1, 32, 16],
            {
                "use_reconstruction_module": False,
                "modality": "MRI",
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 4],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 1,
                "accumulate_predictions": True,
                "no_dc": True,
                "keep_prediction": True,
                "segmentation_module_input_channels": 1,
                "segmentation_loss": {"dice": 1.0},
                "dice_loss_include_background": False,
                "dice_loss_to_onehot_y": False,
                "dice_loss_sigmoid": True,
                "dice_loss_softmax": False,
                "dice_loss_other_act": None,
                "dice_loss_squared_pred": False,
                "dice_loss_jaccard": False,
                "dice_loss_reduction": "mean",
                "dice_loss_smooth_nr": 1,
                "dice_loss_smooth_dr": 1,
                "dice_loss_batch": True,
                "consecutive_slices": 1,
                "magnitude_input": True,
            },
            [0.08],
            [4],
            2,
            4,
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
            [1, 45, 45],
            {
                "use_reconstruction_module": False,
                "modality": "MRI",
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 4],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 4,
                "num_cascades": 5,
                "accumulate_predictions": True,
                "no_dc": True,
                "keep_prediction": True,
                "segmentation_module_input_channels": 1,
                "segmentation_loss": {"dice": 1.0},
                "dice_loss_include_background": False,
                "dice_loss_to_onehot_y": False,
                "dice_loss_sigmoid": True,
                "dice_loss_softmax": False,
                "dice_loss_other_act": None,
                "dice_loss_squared_pred": False,
                "dice_loss_jaccard": False,
                "dice_loss_reduction": "mean",
                "dice_loss_smooth_nr": 1,
                "dice_loss_smooth_dr": 1,
                "dice_loss_batch": True,
                "consecutive_slices": 5,
                "magnitude_input": True,
            },
            [0.08],
            [4],
            2,
            4,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 16,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
        (
            [1, 32, 16],
            {
                "use_reconstruction_module": False,
                "modality": "CT",
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 4],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 1,
                "accumulate_predictions": True,
                "no_dc": True,
                "keep_prediction": True,
                "segmentation_module_input_channels": 1,
                "segmentation_loss": {"dice": 1.0},
                "dice_loss_include_background": False,
                "dice_loss_to_onehot_y": False,
                "dice_loss_sigmoid": True,
                "dice_loss_softmax": False,
                "dice_loss_other_act": None,
                "dice_loss_squared_pred": False,
                "dice_loss_jaccard": False,
                "dice_loss_reduction": "mean",
                "dice_loss_smooth_nr": 1,
                "dice_loss_smooth_dr": 1,
                "dice_loss_batch": True,
                "consecutive_slices": 1,
                "magnitude_input": True,
            },
            [0.08],
            [4],
            2,
            4,
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
            [1, 45, 45],
            {
                "use_reconstruction_module": False,
                "modality": "CT",
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 4],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 4,
                "num_cascades": 5,
                "accumulate_predictions": True,
                "no_dc": True,
                "keep_prediction": True,
                "segmentation_module_input_channels": 1,
                "segmentation_loss": {"dice": 1.0},
                "dice_loss_include_background": False,
                "dice_loss_to_onehot_y": False,
                "dice_loss_sigmoid": True,
                "dice_loss_softmax": False,
                "dice_loss_other_act": None,
                "dice_loss_squared_pred": False,
                "dice_loss_jaccard": False,
                "dice_loss_reduction": "mean",
                "dice_loss_smooth_nr": 1,
                "dice_loss_smooth_dr": 1,
                "dice_loss_batch": True,
                "consecutive_slices": 5,
                "magnitude_input": True,
            },
            [0.08],
            [4],
            2,
            4,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 16,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
    ],
)
def test_cirim(shape, cfg, center_fractions, accelerations, dimensionality, segmentation_classes, trainer):
    """
    Test the Segmentation CIRIM with different parameters.

    Parameters
    ----------
    shape : list of int
        Shape of the input data
    cfg : dict
        Dictionary with the parameters of the qRIM model
    center_fractions : list of float
        List of center fractions to test
    accelerations : list of float
        List of acceleration factors to test
    dimensionality : int
        Dimensionality of the data
    segmentation_classes : int
        Number of segmentation classes
    trainer : dict
        Dictionary with the parameters of the trainer
    """
    output = create_input(shape)

    classes_dim = 1

    consecutive_slices = cfg.get("consecutive_slices")
    if consecutive_slices > 1:
        output = torch.stack([output for _ in range(consecutive_slices)], 1)
        classes_dim += 1

    if output.shape[-1] == 2:
        output = torch.abs(torch.view_as_complex(output))

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    trainer = OmegaConf.create(trainer)
    trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
    trainer = pl.Trainer(**trainer)

    segmentationnet = SegmentationCIRIM.get_model(cfg, trainer=trainer)

    with torch.no_grad():
        pred_segmentation = segmentationnet.forward(output)

    output = torch.stack([output for _ in range(segmentation_classes)], classes_dim)

    if pred_segmentation.shape != output.shape:
        raise AssertionError