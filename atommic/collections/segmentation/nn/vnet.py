# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from atommic.collections.segmentation.nn.segmentationnet import BaseCTSegmentationNet, BaseMRISegmentationNet
from atommic.collections.segmentation.nn.vnet_base.vnet_block import VNet
from atommic.core.classes.common import typecheck

__all__ = ["SegmentationVNet"]


class MRISegmentationVNet(BaseMRISegmentationNet):
    """Implementation of the V-Net for MRI segmentation, as presented in [Milletari2016]_.

    References
    ----------
    .. [Milletari2016] Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks
        for Volumetric Medical Image Segmentation, 2016. https://arxiv.org/abs/1606.04797

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
        return VNet(
            in_chans=self.input_channels,
            out_chans=cfg.get("segmentation_module_output_channels", 2),
            act=cfg.get("segmentation_module_activation", "elu"),
            drop_prob=cfg.get("segmentation_module_dropout", 0.0),
            bias=cfg.get("segmentation_module_bias", False),
        )

    @typecheck()
    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of :class:`BaseSegmentationNet`.

        Parameters
        ----------
        image : torch.Tensor
            Input image. Shape [batch_size, n_x, n_y] or [batch_size, n_x, n_y, 2]
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Predicted segmentation. Shape [batch_size, n_classes, n_x, n_y]
        """
        if self.consecutive_slices > 1:
            batch, slices = image.shape[:2]
            image = image.reshape(batch * slices, *image.shape[2:])

        if image.shape[-1] == 2:
            if self.input_channels == 1:
                image = torch.view_as_complex(image).unsqueeze(1)
                if self.magnitude_input:
                    image = torch.abs(image)
            elif self.input_channels == 2 and not self.magnitude_input:
                image = image.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"The input channels must be either 1 or 2. Found: {self.input_channels}")
        elif self.magnitude_input:
            image = torch.abs(image)

        if image.dim() == 3:
            image = image.unsqueeze(1)

        # if dim 1 is even, add a row of zeros to make it odd
        if image.shape[1] % 2 != 0 and image.shape[1] != 1:
            image = torch.cat((image, torch.zeros_like(image[:, 0:1, :, :]).to(image.device)), dim=1)

        mean = 1.0
        std = 1.0
        if self.normalize:
            image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)
        segmentation = self.segmentation_module(image)
        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)


class CTSegmentationVNet(BaseCTSegmentationNet):
    """Implementation of the V-Net for CT segmentation, as presented in [Milletari2016]_.

    References
    ----------
    .. [Milletari2016] Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks
        for Volumetric Medical Image Segmentation, 2016. https://arxiv.org/abs/1606.04797

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
        return VNet(
            in_chans=self.input_channels,
            out_chans=cfg.get("segmentation_module_output_channels", 2),
            act=cfg.get("segmentation_module_activation", "elu"),
            drop_prob=cfg.get("segmentation_module_dropout", 0.0),
            bias=cfg.get("segmentation_module_bias", False),
        )

    @typecheck()
    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of :class:`BaseSegmentationNet`.

        Parameters
        ----------
        image : torch.Tensor
            Input image. Shape [batch_size, n_x, n_y] or [batch_size, n_x, n_y, 2]
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Predicted segmentation. Shape [batch_size, n_classes, n_x, n_y]
        """
        if self.consecutive_slices > 1:
            batch, slices = image.shape[:2]
            image = image.reshape(batch * slices, *image.shape[2:])

        if image.shape[-1] == 2:
            if self.input_channels == 1:
                image = torch.view_as_complex(image).unsqueeze(1)
                if self.magnitude_input:
                    image = torch.abs(image)
            elif self.input_channels == 2 and not self.magnitude_input:
                image = image.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"The input channels must be either 1 or 2. Found: {self.input_channels}")
        elif self.magnitude_input:
            image = torch.abs(image)

        if image.dim() == 3:
            image = image.unsqueeze(1)

        # if dim 1 is even, add a row of zeros to make it odd
        if image.shape[1] % 2 != 0 and image.shape[1] != 1:
            image = torch.cat((image, torch.zeros_like(image[:, 0:1, :, :]).to(image.device)), dim=1)

        mean = 1.0
        std = 1.0
        if self.normalize:
            image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)
        segmentation = self.segmentation_module(image)
        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)


class SegmentationVNet:
    """Factory class for the V-Net segmentation network, as presented in [Milletari2016]_.

    References
    ----------
    .. [Milletari2016] Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks
        for Volumetric Medical Image Segmentation, 2016. https://arxiv.org/abs/1606.04797
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
            return MRISegmentationVNet(cfg)
        if modality == "ct":
            return CTSegmentationVNet(cfg)
        raise ValueError(f"Unknown modality: {modality}")
