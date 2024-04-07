# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import List, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import Tensor

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import coil_combination_method
from atommic.collections.quantitative.nn.base import BaseqMRIReconstructionModel, SignalForwardModel
from atommic.collections.quantitative.nn.qvarnet_base.qvarnet_block import qVarNetBlock
from atommic.collections.quantitative.parts.transforms import R2star_B0_S0_phi_mapping
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.collections.reconstruction.nn.varnet_base.varnet_block import VarNetBlock
from atommic.core.classes.common import typecheck

__all__ = ["qVarNet"]


class qVarNet(BaseqMRIReconstructionModel):
    """Implementation of the quantitative End-to-end Variational Network (qVN), as presented in [Zhang2022]_.

    References
    ----------
    .. [Zhang2022] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW.
        A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent
        inference machine. NeuroImage. 2022 Dec 1;264:119680.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        quantitative_module_dimensionality = cfg_dict.get("quantitative_module_dimensionality")
        if quantitative_module_dimensionality != 2:
            raise ValueError(
                f"Only 2D is currently supported for qMRI models.Found {quantitative_module_dimensionality}"
            )

        self.reconstruction_module = torch.nn.ModuleList([])

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module")
        if self.use_reconstruction_module:
            self.reconstruction_module_num_cascades = cfg_dict.get("reconstruction_module_num_cascades")
            self.reconstruction_module_no_dc = cfg_dict.get("reconstruction_module_no_dc")

            for _ in range(self.reconstruction_module_num_cascades):
                self.reconstruction_module.append(
                    VarNetBlock(
                        NormUnet(
                            chans=cfg_dict.get("reconstruction_module_channels"),
                            num_pools=cfg_dict.get("reconstruction_module_pooling_layers"),
                            in_chans=cfg_dict.get("reconstruction_module_in_channels"),
                            out_chans=cfg_dict.get("reconstruction_module_out_channels"),
                            padding_size=cfg_dict.get("reconstruction_module_padding_size"),
                            normalize=cfg_dict.get("reconstruction_module_normalize"),
                        ),
                        fft_centered=self.fft_centered,
                        fft_normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                        coil_dim=self.coil_dim - 1,
                        no_dc=self.reconstruction_module_no_dc,
                    )
                )

            self.dc_weight = torch.nn.Parameter(torch.ones(1))
            self.reconstruction_module_accumulate_predictions = cfg_dict.get(
                "reconstruction_module_accumulate_predictions"
            )

        quantitative_module_num_cascades = cfg_dict.get("quantitative_module_num_cascades")
        self.quantitative_module = torch.nn.ModuleList(
            [
                qVarNetBlock(
                    NormUnet(
                        chans=cfg_dict.get("quantitative_module_channels"),
                        num_pools=cfg_dict.get("quantitative_module_pooling_layers"),
                        in_chans=cfg_dict.get("quantitative_module_in_channels"),
                        out_chans=cfg_dict.get("quantitative_module_out_channels"),
                        padding_size=cfg_dict.get("quantitative_module_padding_size"),
                        normalize=cfg_dict.get("quantitative_module_normalize"),
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    no_dc=cfg_dict.get("quantitative_module_no_dc"),
                    linear_forward_model=SignalForwardModel(
                        sequence=cfg_dict.get("quantitative_module_signal_forward_model_sequence")
                    ),
                )
                for _ in range(quantitative_module_num_cascades)
            ]
        )

        self.quantitative_maps_regularization_factors = cfg_dict.get(
            "quantitative_maps_regularization_factors", [150.0, 150.0, 1000.0, 150.0]
        )

        self.accumulate_predictions = cfg_dict.get("quantitative_module_accumulate_predictions")

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        R2star_map_init: torch.Tensor,
        S0_map_init: torch.Tensor,
        B0_map_init: torch.Tensor,
        phi_map_init: torch.Tensor,
        TEs: List,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        initial_prediction: torch.Tensor,
        anatomy_mask: torch.Tensor,
        sampling_mask: torch.Tensor,
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
            cascades_echoes_predictions = []
            for echo in range(y.shape[1]):
                prediction = y[:, echo, ...].clone()
                for cascade in self.reconstruction_module:
                    # Forward pass through the cascades
                    prediction = cascade(prediction, y[:, echo, ...], sensitivity_maps, sampling_mask.squeeze(1))
                reconstruction_prediction = ifft2(
                    prediction,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                reconstruction_prediction = coil_combination_method(
                    reconstruction_prediction,
                    sensitivity_maps,
                    method=self.coil_combination_method,
                    dim=self.coil_dim - 1,
                )
                cascades_echoes_predictions.append(torch.view_as_complex(reconstruction_prediction))

            reconstruction_prediction = torch.stack(cascades_echoes_predictions, dim=1)
            if reconstruction_prediction.shape[-1] != 2:
                reconstruction_prediction = torch.view_as_real(reconstruction_prediction)

            y = fft2(
                coil_combination_method(
                    reconstruction_prediction.unsqueeze(self.coil_dim),
                    sensitivity_maps.unsqueeze(self.coil_dim - 1),
                    method=self.coil_combination_method,
                    dim=self.coil_dim - 1,
                ),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )

            R2star_maps_init = []
            S0_maps_init = []
            B0_maps_init = []
            phi_maps_init = []
            for batch_idx in range(reconstruction_prediction.shape[0]):
                R2star_map_init, S0_map_init, B0_map_init, phi_map_init = R2star_B0_S0_phi_mapping(
                    reconstruction_prediction[batch_idx],
                    TEs,
                    anatomy_mask,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                R2star_maps_init.append(R2star_map_init.squeeze(0))
                S0_maps_init.append(S0_map_init.squeeze(0))
                B0_maps_init.append(B0_map_init.squeeze(0))
                phi_maps_init.append(phi_map_init.squeeze(0))
            R2star_map_init = torch.stack(R2star_maps_init, dim=0).to(y)
            S0_map_init = torch.stack(S0_maps_init, dim=0).to(y)
            B0_map_init = torch.stack(B0_maps_init, dim=0).to(y)
            phi_map_init = torch.stack(phi_maps_init, dim=0).to(y)
        else:
            reconstruction_prediction = initial_prediction.clone()

        R2star_map_pred = R2star_map_init / self.quantitative_maps_regularization_factors[0]
        S0_map_pred = S0_map_init / self.quantitative_maps_regularization_factors[1]
        B0_map_pred = B0_map_init / self.quantitative_maps_regularization_factors[2]
        phi_map_pred = phi_map_init / self.quantitative_maps_regularization_factors[3]

        prediction = torch.stack([R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred], dim=1)
        for cascade in self.quantitative_module:
            # Forward pass through the cascades
            prediction = cascade(
                prediction,
                y,
                sensitivity_maps,
                sampling_mask,
                TEs,
            )

        R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
            prediction[:, 0, ...] * self.quantitative_maps_regularization_factors[0],
            prediction[:, 1, ...] * self.quantitative_maps_regularization_factors[1],
            prediction[:, 2, ...] * self.quantitative_maps_regularization_factors[2],
            prediction[:, 3, ...] * self.quantitative_maps_regularization_factors[3],
        )

        return [
            reconstruction_prediction if self.use_reconstruction_module else torch.empty([]),
            R2star_map_pred,
            S0_map_pred,
            B0_map_pred,
            phi_map_pred,
        ]
