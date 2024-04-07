# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import List, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import Tensor

from atommic.collections.common.parts.fft import ifft2
from atommic.collections.common.parts.utils import coil_combination_method
from atommic.collections.quantitative.nn.base import BaseqMRIReconstructionModel
from atommic.core.classes.common import typecheck

__all__ = ["qZF"]


class qZF(BaseqMRIReconstructionModel):
    """Abstract class for returning the initial estimates of the quantitative maps."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        quantitative_module_dimensionality = cfg_dict.get("quantitative_module_dimensionality")
        if quantitative_module_dimensionality != 2:
            raise ValueError(
                f"Only 2D is currently supported for qMRI models.Found {quantitative_module_dimensionality}"
            )

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module")

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        R2star_map_init: torch.Tensor,
        S0_map_init: torch.Tensor,
        B0_map_init: torch.Tensor,
        phi_map_init: torch.Tensor,
        TEs: List,  # pylint: disable=unused-argument
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        initial_prediction: torch.Tensor,
        anatomy_mask: torch.Tensor,  # pylint: disable=unused-argument
        sampling_mask: torch.Tensor,  # pylint: disable=unused-argument
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> Union[List[List[Tensor]], List[Tensor]]:
        """
        Forward pass of the network.

        Parameters
        ----------
        R2star_map_init : torch.Tensor
            Initial R2* map of shape [batch_size, n_x, n_y].
        S0_map_init : torch.Tensor
            Initial S0 map of shape [batch_size, n_x, n_y].
        B0_map_init : torch.Tensor
            Initial B0 map of shape [batch_size, n_x, n_y].
        phi_map_init : torch.Tensor
            Initial phase map of shape [batch_size, n_x, n_y].
        TEs : List
            List of echo times.
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2].
        initial_prediction : torch.Tensor
            Initial prediction of shape [batch_size, n_x, n_y, 2].
        anatomy_mask : torch.Tensor
            Brain mask of shape [batch_size, 1, n_x, n_y, 1].
        sampling_mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        sigma : float
            Standard deviation of the noise. Default is ``1.0``.

        Returns
        -------
        List of list of torch.Tensor or torch.Tensor
             If self.accumulate_loss is True, returns a list of all intermediate predictions.
             If False, returns the final estimate.
        """
        if self.use_reconstruction_module:
            reconstruction_prediction = torch.stack(
                [
                    torch.view_as_complex(
                        coil_combination_method(
                            ifft2(
                                y[:, echo, ...].clone(), self.fft_centered, self.fft_normalization, self.spatial_dims
                            ),
                            sensitivity_maps,
                            method=self.coil_combination_method,
                            dim=self.coil_dim - 1,
                        )
                    )
                    for echo in range(y.shape[1])
                ],
                dim=1,
            )
        else:
            reconstruction_prediction = initial_prediction

        return [
            reconstruction_prediction,
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
        ]
