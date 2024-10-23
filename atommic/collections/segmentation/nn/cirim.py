# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.segmentation.nn.rim_base.rim_block import SegmentationRIMBlock
from atommic.collections.segmentation.nn.segmentationnet import BaseCTSegmentationNet, BaseMRISegmentationNet
from atommic.core.classes.common import typecheck

__all__ = ["SegmentationCIRIM"]


class MRISegmentationCIRIM(BaseMRISegmentationNet):
    """Implementation of the Cascades of Independently Recurrent Inference Machines, as presented in
    [Karkalousos2022]_.

    References
    ----------
    .. [Karkalousos2022] Karkalousos D, Noteboom S, Hulst HE, Vos FM, Caan MWA. Assessment of data consistency through
        cascades of independently recurrent inference machines for fast and robust accelerated MRI reconstruction.
        Phys Med Biol. 2022 Jun 8;67(12). doi: 10.1088/1361-6560/ac6cc2. PMID: 35508147.

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
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        conv_filters = cfg_dict.get("conv_filters")
        self.output_classes = conv_filters[-1]

        reconstruction_module = torch.nn.ModuleList(
            [
                SegmentationRIMBlock(
                    recurrent_layer=cfg_dict.get("recurrent_layer"),
                    conv_filters=conv_filters,
                    conv_kernels=cfg_dict.get("conv_kernels"),
                    conv_dilations=cfg_dict.get("conv_dilations"),
                    conv_bias=cfg_dict.get("conv_bias"),
                    recurrent_filters=cfg_dict.get("recurrent_filters"),
                    recurrent_kernels=cfg_dict.get("recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("recurrent_bias"),
                    depth=cfg_dict.get("depth"),
                    time_steps=8 * math.ceil(cfg_dict.get("time_steps") / 8),
                    conv_dim=cfg_dict.get("conv_dim"),
                )
                for _ in range(cfg_dict.get("num_cascades"))
            ]
        )

        # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
        self.keep_prediction = cfg_dict.get("keep_prediction")

        return reconstruction_module

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

        mean = 1.0
        std = 1.0
        if self.normalize:
            image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)

        image = torch.cat([image for _ in range(self.output_classes)], dim=1)

        noise_mean = 0.0
        noise_std = 0.1
        noise = torch.randn_like(image) * noise_std + noise_mean
        prediction = image + noise

        hx = None
        cascades_predictions = []
        for _, cascade in enumerate(self.segmentation_module):
            # Forward pass through the cascades
            prediction, hx = cascade(prediction, image, hx)
            cascades_predictions.append(prediction)
            prediction = prediction[-1]

        segmentation = cascades_predictions[-1][-1]

        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)


class CTSegmentationCIRIM(BaseCTSegmentationNet):
    """Implementation of the Cascades of Independently Recurrent Inference Machines, as presented in
    [Karkalousos2022]_.

    References
    ----------
    .. [Karkalousos2022] Karkalousos D, Noteboom S, Hulst HE, Vos FM, Caan MWA. Assessment of data consistency through
        cascades of independently recurrent inference machines for fast and robust accelerated MRI reconstruction.
        Phys Med Biol. 2022 Jun 8;67(12). doi: 10.1088/1361-6560/ac6cc2. PMID: 35508147.

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
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        conv_filters = cfg_dict.get("conv_filters")
        self.output_classes = conv_filters[-1]

        reconstruction_module = torch.nn.ModuleList(
            [
                SegmentationRIMBlock(
                    recurrent_layer=cfg_dict.get("recurrent_layer"),
                    conv_filters=conv_filters,
                    conv_kernels=cfg_dict.get("conv_kernels"),
                    conv_dilations=cfg_dict.get("conv_dilations"),
                    conv_bias=cfg_dict.get("conv_bias"),
                    recurrent_filters=cfg_dict.get("recurrent_filters"),
                    recurrent_kernels=cfg_dict.get("recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("recurrent_bias"),
                    depth=cfg_dict.get("depth"),
                    time_steps=8 * math.ceil(cfg_dict.get("time_steps") / 8),
                    conv_dim=cfg_dict.get("conv_dim"),
                )
                for _ in range(cfg_dict.get("num_cascades"))
            ]
        )

        # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
        self.keep_prediction = cfg_dict.get("keep_prediction")

        return reconstruction_module

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

        if image.dim() == 3:
            image = image.unsqueeze(1)

        mean = 1.0
        std = 1.0
        if self.normalize:
            image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)

        image = torch.cat([image for _ in range(self.output_classes)], dim=1)

        noise_mean = 0.0
        noise_std = 0.1
        noise = torch.randn_like(image) * noise_std + noise_mean
        prediction = image + noise

        hx = None
        cascades_predictions = []
        for _, cascade in enumerate(self.segmentation_module):
            # Forward pass through the cascades
            prediction, hx = cascade(prediction, image, hx)
            cascades_predictions.append(prediction)
            prediction = prediction[-1]

        segmentation = cascades_predictions[-1][-1]

        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)


class SegmentationCIRIM:
    """Factory class for the Cascades of Independently Recurrent Inference Machines, as presented in
    [Karkalousos2022]_. Here it is used for segmentation.

    References
    ----------
    .. [Karkalousos2022] Karkalousos D, Noteboom S, Hulst HE, Vos FM, Caan MWA. Assessment of data consistency through
        cascades of independently recurrent inference machines for fast and robust accelerated MRI reconstruction.
        Phys Med Biol. 2022 Jun 8;67(12). doi: 10.1088/1361-6560/ac6cc2. PMID: 35508147.

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
            return MRISegmentationCIRIM(cfg)
        if modality == "ct":
            return CTSegmentationCIRIM(cfg)
        raise ValueError(f"Unknown modality: {modality}")
