# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from omegaconf import DictConfig

from atommic.collections.segmentation.nn.segmentationnet import BaseSegmentationNet
from atommic.collections.segmentation.nn.unet3d_base.unet3d_block import UNet3D
from atommic.core.classes.common import typecheck

__all__ = ["Segmentation3DUNet"]


class Segmentation3DUNet(BaseSegmentationNet):
    """Implementation of the (3D) UNet for MRI segmentation, as presented in [Ronneberger2015]_.

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
        return UNet3D(
            in_chans=self.input_channels,
            out_chans=cfg.get("segmentation_module_output_channels", 2),
            chans=cfg.get("segmentation_module_channels", 64),
            num_pool_layers=cfg.get("segmentation_module_pooling_layers", 2),
            drop_prob=cfg.get("segmentation_module_dropout", 0.0),
        )

    @typecheck()
    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        image : torch.Tensor
            Input image. Shape [batch_size, slices, classes, n_x, n_y] or [batch_size, slices, classes, n_x, n_y, 2]
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Predicted segmentation. Shape [batch_size, n_classes, n_x, n_y]
        """
        # Adjust the dimensions of the input image
        if image.shape[-1] == 2:
            if self.input_channels == 1:
                image = torch.view_as_complex(image).unsqueeze(1)
                if self.magnitude_input:
                    image = torch.abs(image)
            elif self.input_channels == 2 and not self.magnitude_input:
                image = image.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"The input channels must be either 1 or 2. Found: {self.input_channels}")

        if image.dim() == 4:
            # we are missing the classes dimension
            image = image.unsqueeze(2)

        mean = 1.0
        std = 1.0
        if self.normalize:
            image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)
        segmentation = self.segmentation_module(image.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        return torch.abs(segmentation)
