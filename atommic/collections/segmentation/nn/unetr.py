# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from omegaconf import DictConfig

from atommic.collections.segmentation.nn.segmentationnet import BaseSegmentationNet
from atommic.collections.segmentation.nn.unetr_base.unetr_block import UNETR

__all__ = ["SegmentationUNetR"]

from atommic.core.classes.common import typecheck


class SegmentationUNetR(BaseSegmentationNet):
    """Implementation of the UNETR for MRI segmentation, as presented in [Hatamizadeh2022]_.

    References
    ----------
    .. [Hatamizadeh2022] Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr:
        Transformers for 3d medical image segmentation. InProceedings of the IEEE/CVF Winter Conference on
        Applications of Computer Vision 2022 (pp. 574-584).

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
        return UNETR(
            in_channels=self.input_channels,
            out_channels=cfg.get("segmentation_module_output_channels", 2),
            img_size=cfg.get("segmentation_module_img_size", (256, 256)),
            feature_size=cfg.get("segmentation_module_channels", 64),
            hidden_size=cfg.get("segmentation_module_hidden_size", 768),
            mlp_dim=cfg.get("segmentation_module_mlp_dim", 3072),
            num_heads=cfg.get("segmentation_module_num_heads", 12),
            pos_embed=cfg.get("segmentation_module_pos_embed", "conv"),
            norm_name=cfg.get("segmentation_module_norm_name", "instance"),
            conv_block=cfg.get("segmentation_module_conv_block", True),
            res_block=cfg.get("segmentation_module_res_block", True),
            dropout_rate=cfg.get("segmentation_module_dropout", 0.0),
            spatial_dims=cfg.get("dimensionality", 2),
            qkv_bias=cfg.get("segmentation_module_qkv_bias", False),
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
