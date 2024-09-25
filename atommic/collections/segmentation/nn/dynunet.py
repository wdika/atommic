# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from atommic.collections.segmentation.nn.dynunet_base.dynunet_block import DynUNet
from atommic.collections.segmentation.nn.segmentationnet import BaseCTSegmentationNet, BaseMRISegmentationNet
from atommic.core.classes.common import typecheck

__all__ = ["SegmentationDYNUNet"]


class MRISegmentationDYNUNet(BaseMRISegmentationNet):
    """Implementation of a Dynamic UNet (DynUNet) for MRI segmentation, based on [Isensee2018]_.

    References
    ----------
    .. [Isensee2018] Isensee F, Petersen J, Klein A, Zimmerer D, Jaeger PF, Kohl S, Wasserthal J, Koehler G,
        Norajitra T, Wirkert S, Maier-Hein KH. nnu-net: Self-adapting framework for u-net-based medical image
        segmentation. arXiv preprint arXiv:1809.10486. 2018 Sep 27.

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
        strides = cfg.get("segmentation_module_strides", (1, 1, 1, 1))
        self.deep_supervision = cfg.get("segmentation_module_deep_supervision", False)
        return DynUNet(
            spatial_dims=cfg.get("dimensionality", 2),
            in_channels=self.input_channels,
            out_channels=cfg.get("segmentation_module_output_channels", 2),
            kernel_size=cfg.get("segmentation_module_kernel_size", 3),
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=cfg.get("segmentation_module_channels", 64),
            dropout=cfg.get("segmentation_module_dropout", 0.0),
            norm_name=cfg.get("segmentation_module_norm", "instance"),
            act_name=cfg.get("segmentation_module_activation", "leakyrelu"),
            deep_supervision=self.deep_supervision,
            deep_supr_num=cfg.get("segmentation_module_deep_supervision_levels", 1),
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
        elif image.dim() == 3:
            image = image.unsqueeze(1)

        mean = 1.0
        std = 1.0
        if self.normalize:
            image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)
        segmentation = self.segmentation_module(image)
        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.deep_supervision and segmentation.dim() == 5:
            # TODO: check if this is correct. They do unbind, but they don't show how they handle the tuples.
            segmentation = torch.sum(segmentation, dim=1)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)


class CTSegmentationDYNUNet(BaseCTSegmentationNet):
    """Implementation of a Dynamic UNet (DynUNet) for CT segmentation, based on [Isensee2018]_.

    References
    ----------
    .. [Isensee2018] Isensee F, Petersen J, Klein A, Zimmerer D, Jaeger PF, Kohl S, Wasserthal J, Koehler G,
        Norajitra T, Wirkert S, Maier-Hein KH. nnu-net: Self-adapting framework for u-net-based medical image
        segmentation. arXiv preprint arXiv:1809.10486. 2018 Sep 27.

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
        strides = cfg.get("segmentation_module_strides", (1, 1, 1, 1))
        self.deep_supervision = cfg.get("segmentation_module_deep_supervision", False)
        return DynUNet(
            spatial_dims=cfg.get("dimensionality", 2),
            in_channels=self.input_channels,
            out_channels=cfg.get("segmentation_module_output_channels", 2),
            kernel_size=cfg.get("segmentation_module_kernel_size", 3),
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=cfg.get("segmentation_module_channels", 64),
            dropout=cfg.get("segmentation_module_dropout", 0.0),
            norm_name=cfg.get("segmentation_module_norm", "instance"),
            act_name=cfg.get("segmentation_module_activation", "leakyrelu"),
            deep_supervision=self.deep_supervision,
            deep_supr_num=cfg.get("segmentation_module_deep_supervision_levels", 1),
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
        elif image.dim() == 3:
            image = image.unsqueeze(1)

        mean = 1.0
        std = 1.0
        if self.normalize:
            image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)
        segmentation = self.segmentation_module(image)
        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.deep_supervision and segmentation.dim() == 5:
            # TODO: check if this is correct. They do unbind, but they don't show how they handle the tuples.
            segmentation = torch.sum(segmentation, dim=1)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)


class SegmentationDYNUNet:
    """Factory class for the DynUNet segmentation network, as presented in [Isensee2018]_.

    References
    ----------
    .. [Isensee2018] Isensee F, Petersen J, Klein A, Zimmerer D, Jaeger PF, Kohl S, Wasserthal J, Koehler G,
        Norajitra T, Wirkert S, Maier-Hein KH. nnu-net: Self-adapting framework for u-net-based medical image
        segmentation. arXiv preprint arXiv:1809.10486. 2018 Sep 27.

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
            return MRISegmentationDYNUNet(cfg)
        if modality == "ct":
            return CTSegmentationDYNUNet(cfg)
        raise ValueError(f"Unknown modality: {modality}")
