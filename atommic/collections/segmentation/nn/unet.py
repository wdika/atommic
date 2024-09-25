# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from atommic.collections.reconstruction.nn.unet_base.unet_block import Unet
from atommic.collections.segmentation.nn.segmentationnet import BaseCTSegmentationNet, BaseMRISegmentationNet

__all__ = ["SegmentationUNet"]


class MRISegmentationUNet(BaseMRISegmentationNet):
    """Implementation of the (2D) UNet for MRI segmentation, as presented in [Ronneberger2015]_.

    References
    ----------
    .. [Ronneberger2015] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
        image segmentation. In International Conference on Medical image computing and computer-assisted intervention,
        pages 234–241. Springer, 2015.

    """

    def build_segmentation_module(self, cfg: DictConfig) -> torch.nn.Module:
        """Build the segmentation module.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object specifying the model's hyperparameters.

        Returns
        -------
        torch.nn.Module
            The segmentation module.
        """
        return Unet(
            in_chans=self.input_channels,
            out_chans=cfg.get("segmentation_module_output_channels", 2),
            chans=cfg.get("segmentation_module_channels", 64),
            num_pool_layers=cfg.get("segmentation_module_pooling_layers", 2),
            drop_prob=cfg.get("segmentation_module_dropout", 0.0),
        )


class CTSegmentationUNet(BaseCTSegmentationNet):
    """Implementation of the (2D) UNet for CT segmentation, as presented in [Ronneberger2015]_.

    References
    ----------
    .. [Ronneberger2015] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
        image segmentation. In International Conference on Medical image computing and computer-assisted intervention,
        pages 234–241. Springer, 2015.

    """

    def build_segmentation_module(self, cfg: DictConfig) -> torch.nn.Module:
        """Build the segmentation module.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object specifying the model's hyperparameters.

        Returns
        -------
        torch.nn.Module
            The segmentation module.
        """
        return Unet(
            in_chans=self.input_channels,
            out_chans=cfg.get("segmentation_module_output_channels", 2),
            chans=cfg.get("segmentation_module_channels", 64),
            num_pool_layers=cfg.get("segmentation_module_pooling_layers", 2),
            drop_prob=cfg.get("segmentation_module_dropout", 0.0),
        )


class SegmentationUNet:
    """Factory class for the UNet segmentation network, as presented in [Ronneberger2015]_.

    References
    ----------
    .. [Ronneberger2015] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
        image segmentation. In International Conference on Medical image computing and computer-assisted intervention,
        pages 234–241. Springer, 2015.
    """

    @staticmethod
    def get_model(cfg: DictConfig, trainer: Trainer = None):  # pylint: disable=unused-argument
        """Get the segmentation network.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object specifying the network's hyperparameters.
        trainer : Trainer, optional
            PyTorch Lightning trainer object, by default None.

        Returns
        -------
        torch.nn.Module
            The segmentation network.
        """
        modality = cfg.get("modality", "MRI").lower()
        if modality == "mri":
            return MRISegmentationUNet(cfg)
        if modality == "ct":
            return CTSegmentationUNet(cfg)
        raise ValueError(f"Unknown modality: {modality}")
