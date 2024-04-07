# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import check_stacked_complex, coil_combination_method, rnn_weights_init
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.recurrentvarnet_base.recurrentvarnet_block import (
    RecurrentInit,
    RecurrentVarNetBlock,
)
from atommic.core.classes.common import typecheck

__all__ = ["RecurrentVarNet"]


class RecurrentVarNet(BaseMRIReconstructionModel):
    """Implementation of the Recurrent Variational Network implementation, as presented in [Yiasemis2021]_.

    References
    ----------
    .. [Yiasemis2021] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver
        Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`RecurrentVarNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.in_channels = cfg_dict.get("in_channels")
        self.recurrent_hidden_channels = cfg_dict.get("recurrent_hidden_channels")
        self.recurrent_num_layers = cfg_dict.get("recurrent_num_layers")
        self.no_parameter_sharing = cfg_dict.get("no_parameter_sharing")

        # make time-steps size divisible by 8 for fast fp16 training
        self.num_steps = 8 * math.ceil(cfg_dict.get("num_steps") / 8)

        self.learned_initializer = cfg_dict.get("learned_initializer")
        self.initializer_initialization = cfg_dict.get("initializer_initialization")
        self.initializer_channels = cfg_dict.get("initializer_channels")
        self.initializer_dilations = cfg_dict.get("initializer_dilations")

        if (
            self.learned_initializer
            and self.initializer_initialization is not None
            and self.initializer_channels is not None
            and self.initializer_dilations is not None
        ):
            if self.initializer_initialization not in [
                "sense",
                "input_image",
                "zero_filled",
            ]:
                raise ValueError(
                    "Unknown initializer_initialization. Expected `sense`, `'input_image` or `zero_filled`."
                    f"Got {self.initializer_initialization}."
                )
            self.initializer = RecurrentInit(
                self.in_channels,
                self.recurrent_hidden_channels,
                channels=self.initializer_channels,
                dilations=self.initializer_dilations,
                depth=self.recurrent_num_layers,
                multiscale_depth=cfg_dict.get("initializer_multiscale"),
            )
        else:
            self.initializer = None  # type: ignore

        self.block_list: torch.nn.Module = torch.nn.ModuleList()
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.block_list.append(
                RecurrentVarNetBlock(
                    in_channels=self.in_channels,
                    hidden_channels=self.recurrent_hidden_channels,
                    num_layers=self.recurrent_num_layers,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                )
            )

        std_init_range = 1 / self.recurrent_hidden_channels**0.5

        # initialize weights if not using pretrained cirim
        if not cfg_dict.get("pretrained", False):
            self.block_list.apply(lambda module: rnn_weights_init(module, std_init_range))

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        initial_prediction: torch.Tensor,
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass of :class:`RecurrentVarNet`.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        initial_prediction : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2]
        sigma : float
            Noise level. Default is ``1.0``.

        Returns
        -------
        torch.Tensor
            Prediction of the final cascade. Shape [batch_size, n_x, n_y]
        """
        previous_state: Optional[torch.Tensor] = None

        if self.initializer is not None:
            if self.initializer_initialization in ("sense", "input_image"):
                initializer_input_image = initial_prediction.unsqueeze(self.coil_dim)
            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = ifft2(
                    y,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

            previous_state = self.initializer(
                fft2(
                    initializer_input_image,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                .sum(1)
                .permute(0, 3, 1, 2)
            )

        kspace_prediction = y.clone()
        for step in range(self.num_steps):
            block = self.block_list[step] if self.no_parameter_sharing else self.block_list[0]
            kspace_prediction, previous_state = block(
                kspace_prediction,
                y,
                mask,
                sensitivity_maps,
                previous_state,
            )

        return check_stacked_complex(
            coil_combination_method(
                ifft2(kspace_prediction, self.fft_centered, self.fft_normalization, self.spatial_dims),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )
        )
