# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
from abc import ABC
from typing import List, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import atommic.collections.segmentation.nn.base as base_segmentation_models
from atommic.core.classes.common import typecheck

__all__ = ["BaseMRISegmentationNet", "BaseCTSegmentationNet"]


class BaseMRISegmentationNet(base_segmentation_models.BaseMRISegmentationModel, ABC):
    """Abstract class for all MRI segmentation models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """inits :class:`BaseMRISegmentationNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object specifying the model's hyperparameters.
        trainer : Trainer, optional
            PyTorch Lightning trainer object, by default None.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.padding_size = cfg_dict.get("segmentation_module_padding_size", 11)
        self.normalize = cfg_dict.get("segmentation_module_normalize", False)
        self.norm_groups = cfg_dict.get("segmentation_module_norm_groups", 2)
        self.segmentation_module = self.build_segmentation_module(cfg)

    def build_segmentation_module(self, cfg: DictConfig) -> torch.nn.Module:
        """Build the MRI segmentation module.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object specifying the model's hyperparameters.
        """
        raise NotImplementedError

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the input."""
        # group norm
        b, c, s, h, w = x.shape

        x = x.reshape(b, self.norm_groups, -1)

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        x = (x - mean) / std

        x = x.reshape(b, c, s, h, w)

        return x, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Unnormalize the input."""
        b, c, h, w = x.shape
        input_data = x.reshape(b, self.norm_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """Pad the input with zeros to make it square.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]
            Padded tensor and padding sizes.
        """
        if x.dim() == 4:
            _, _, h, w = x.shape
        elif x.dim() == 5:
            _, _, _, h, w = x.shape
        w_mult = ((w - 1) | self.padding_size) + 1
        h_mult = ((h - 1) | self.padding_size) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = torch.nn.functional.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        """Unpad the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        h_pad : List[int]
            Height padding sizes.
        w_pad : List[int]
            Width padding sizes.
        h_mult : int
            Height multiplier.
        w_mult : int
            Width multiplier.

        Returns
        -------
        torch.Tensor
            Unpadded tensor.
        """
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    @typecheck()
    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:  # pylint: disable=arguments-differ
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
        segmentation = self.segmentation_module(image)
        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)


class BaseCTSegmentationNet(base_segmentation_models.BaseCTSegmentationModel, ABC):
    """Abstract class for all CT segmentation models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """inits :class:`BaseCTSegmentationNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object specifying the model's hyperparameters.
        trainer : Trainer, optional
            PyTorch Lightning trainer object, by default None.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.padding_size = cfg_dict.get("segmentation_module_padding_size", 11)
        self.normalize = cfg_dict.get("segmentation_module_normalize", False)
        self.norm_groups = cfg_dict.get("segmentation_module_norm_groups", 2)
        self.segmentation_module = self.build_segmentation_module(cfg)

    def build_segmentation_module(self, cfg: DictConfig) -> torch.nn.Module:
        """Build the CT segmentation module.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object specifying the model's hyperparameters.
        """
        raise NotImplementedError

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the input."""
        # group norm
        b, c, s, h, w = x.shape

        x = x.reshape(b, self.norm_groups, -1)

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        x = (x - mean) / std

        x = x.reshape(b, c, s, h, w)

        return x, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Unnormalize the input."""
        b, c, h, w = x.shape
        input_data = x.reshape(b, self.norm_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """Pad the input with zeros to make it square.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]
            Padded tensor and padding sizes.
        """
        if x.dim() == 4:
            _, _, h, w = x.shape
        elif x.dim() == 5:
            _, _, _, h, w = x.shape
        w_mult = ((w - 1) | self.padding_size) + 1
        h_mult = ((h - 1) | self.padding_size) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = torch.nn.functional.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        """Unpad the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        h_pad : List[int]
            Height padding sizes.
        w_pad : List[int]
            Width padding sizes.
        h_mult : int
            Height multiplier.
        w_mult : int
            Width multiplier.

        Returns
        -------
        torch.Tensor
            Unpadded tensor.
        """
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    @typecheck()
    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:  # pylint: disable=arguments-differ
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
        segmentation = self.segmentation_module(image)
        segmentation = self.unpad(segmentation, *pad_sizes)
        if self.normalize:
            segmentation = self.unnorm(segmentation, mean, std)

        if self.normalize_segmentation_output:
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())

        if self.consecutive_slices > 1:
            segmentation = segmentation.reshape(batch, slices, *segmentation.shape[1:])

        return torch.abs(segmentation)
