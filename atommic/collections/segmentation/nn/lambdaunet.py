# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from omegaconf import DictConfig

from atommic.collections.segmentation.nn.lambdaunet_base.lambdaunet_block import LambdaUNet
from atommic.collections.segmentation.nn.segmentationnet import BaseSegmentationNet
from atommic.core.classes.common import typecheck

__all__ = ["SegmentationLambdaUNet"]


class SegmentationLambdaUNet(BaseSegmentationNet):
    """Implementation of the Lambda UNet for MRI segmentation, as presented in [Yanglan2021]_.

    References
    ----------
    .. [Yanglan2021] Yanglan Ou, Ye Yuan, Xiaolei Huang, Kelvin Wong, John Volpi, James Z. Wang, Stephen T.C. Wong.
        LambdaUNet: 2.5D Stroke Lesion Segmentation of Diffusion-weighted MR Images. 2021.
        https://arxiv.org/abs/2104.13917

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
        return LambdaUNet(
            in_chans=self.input_channels,
            out_chans=cfg.get("segmentation_module_output_channels", 2),
            chans=cfg.get("segmentation_module_channels", 32),
            num_pool_layers=cfg.get("segmentation_module_pooling_layers", 4),
            drop_prob=cfg.get("segmentation_module_dropout", 0.0),
            query_depth=cfg.get("segmentation_module_query_depth", 16),
            intra_depth=cfg.get("segmentation_module_intra_depth", 4),
            receptive_kernel=cfg.get("segmentation_module_receptive_kernel", 3),
            temporal_kernel=cfg.get("segmentation_module_temporal_kernel", 1),
            num_slices=self.consecutive_slices,
        )

    @typecheck()
    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the network.

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

        mean = 1.0
        std = 1.0
        if self.normalize:
            image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)
        if self.consecutive_slices > 1:
            batch, slices = image.shape[:2]
            image = image.reshape(batch * slices, *image.shape[2:])
        segmentation = self.segmentation_module(image)
        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)
