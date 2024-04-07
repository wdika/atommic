# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.utils import check_stacked_complex, coil_combination_method
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.didn_base.didn_block import DIDN
from atommic.collections.reconstruction.nn.sigmanet_base.dc_layers import (
    DataGDLayer,
    DataIDLayer,
    DataProxCGLayer,
    DataVSLayer,
)
from atommic.collections.reconstruction.nn.sigmanet_base.sensitivity_net import SensitivityNetwork
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.core.classes.common import typecheck

__all__ = ["DUNet"]


class DUNet(BaseMRIReconstructionModel):
    """Implementation of the Down-Up NET, inspired by [Hammernik2021]_.

    References
    ----------
    .. [Hammernik2021] Hammernik, K, Schlemper, J, Qin, C, et al. Systematic valuation of iterative deep neural
        networks for fast parallel MRI reconstruction with sensitivity-weighted coil combination. Magn Reson Med.
        2021; 86: 1859– 1872. https://doi.org/10.1002/mrm.28827

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`DUNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        reg_model_architecture = cfg_dict.get("reg_model_architecture")
        if reg_model_architecture == "DIDN":
            reg_model = DIDN(
                in_channels=cfg_dict.get("in_channels", 2),
                out_channels=cfg_dict.get("out_channels", 2),
                hidden_channels=cfg_dict.get("didn_hidden_channels"),
                num_dubs=cfg_dict.get("didn_num_dubs"),
                num_convs_recon=cfg_dict.get("didn_num_convs_recon"),
            )
        elif reg_model_architecture in ["UNET", "NORMUNET"]:
            reg_model = NormUnet(
                cfg_dict.get("unet_num_filters"),
                cfg_dict.get("unet_num_pool_layers"),
                in_chans=cfg_dict.get("in_channels", 2),
                out_chans=cfg_dict.get("out_channels", 2),
                drop_prob=cfg_dict.get("unet_dropout_probability"),
                padding_size=cfg_dict.get("unet_padding_size"),
                normalize=cfg_dict.get("unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "DUNET is currently implemented for reg_model_architecture == 'DIDN' or 'UNet'."
                f"Got reg_model_architecture == {reg_model_architecture}."
            )

        data_consistency_term = cfg_dict.get("data_consistency_term")

        if data_consistency_term == "GD":
            dc_layer = DataGDLayer(
                lambda_init=cfg_dict.get("data_consistency_lambda_init"),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif data_consistency_term == "PROX":
            dc_layer = DataProxCGLayer(
                lambda_init=cfg_dict.get("data_consistency_lambda_init"),
                iterations=cfg_dict.get("data_consistency_iterations", 10),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif data_consistency_term == "VS":
            dc_layer = DataVSLayer(
                alpha_init=cfg_dict.get("data_consistency_alpha_init", 1.0),
                beta_init=cfg_dict.get("data_consistency_beta_init", 1.0),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        else:
            dc_layer = DataIDLayer()

        self.reconstruction_module = SensitivityNetwork(
            cfg_dict.get("num_iter"),
            reg_model,
            dc_layer,
            shared_params=cfg_dict.get("shared_params"),
            save_space=False,
            reset_cache=False,
        )

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
        """Forward pass of :class:`DUNet`.

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
        return check_stacked_complex(
            coil_combination_method(
                self.reconstruction_module(initial_prediction, y, sensitivity_maps, mask),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )
        )
