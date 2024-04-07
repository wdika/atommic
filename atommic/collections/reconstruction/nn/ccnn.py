# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import ifft2
from atommic.collections.common.parts.utils import check_stacked_complex, coil_combination_method
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.ccnn_base.ccnn_block import CascadeNetBlock, Conv2d
from atommic.core.classes.common import typecheck

__all__ = ["CascadeNet"]


class CascadeNet(BaseMRIReconstructionModel):
    """Implementation of the Deep Cascade of Convolutional Neural Networks, as presented in [Schlemper2017]_.

    References
    ----------
    .. [Schlemper2017] Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D., A Deep Cascade of
        Convolutional Neural Networks for MR Image Reconstruction. Information Processing in Medical Imaging (IPMI),
        2017.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`CascadeNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Cascades of CascadeCNN blocks
        self.reconstruction_module = torch.nn.ModuleList(
            [
                CascadeNetBlock(
                    Conv2d(
                        in_channels=2,
                        out_channels=2,
                        hidden_channels=cfg_dict.get("hidden_channels"),
                        n_convs=cfg_dict.get("n_convs"),
                        batchnorm=cfg_dict.get("batchnorm"),
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    no_dc=cfg_dict.get("no_dc"),
                )
                for _ in range(cfg_dict.get("num_cascades"))
            ]
        )

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        initial_prediction: torch.Tensor,  # pylint: disable=unused-argument
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass of :class:`CascadeNet`.

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
        prediction = y.clone()
        for cascade in self.reconstruction_module:
            prediction = cascade(prediction, y, sensitivity_maps, mask)
        return check_stacked_complex(
            coil_combination_method(
                ifft2(prediction, self.fft_centered, self.fft_normalization, self.spatial_dims),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )
        )
