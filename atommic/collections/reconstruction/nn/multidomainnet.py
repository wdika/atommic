# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import ifft2
from atommic.collections.common.parts.utils import check_stacked_complex, coil_combination_method
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.multidomainnet_base.multidomainnet_block import (
    MultiDomainUnet2d,
    StandardizationLayer,
)
from atommic.core.classes.common import typecheck

__all__ = ["MultiDomainNet"]


class MultiDomainNet(BaseMRIReconstructionModel):
    """Feature-level multi-domain module. Inspired by AIRS Medical submission to the FastMRI 2020 challenge."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`MultiDomainNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_cascades = cfg_dict.get("num_cascades")

        standardization = cfg_dict["standardization"]
        if standardization:
            self.standardization = StandardizationLayer(self.coil_dim, -1)

        self.reconstruction_module = MultiDomainUnet2d(
            # if standardization, in_channels is 4 due to standardized input
            in_channels=4 if standardization else 2,
            out_channels=2,
            num_filters=cfg_dict["num_filters"],
            num_pool_layers=cfg_dict["num_pool_layers"],
            dropout_probability=cfg_dict["dropout_probability"],
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
        )

    def _compute_model_per_coil(self, model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Computes the model per coil.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be computed.
        data : torch.Tensor
            The data to be computed. Shape [batch_size, n_coils, n_x, n_y, 2].

        Returns
        -------
        torch.Tensor
            The computed output. Shape [batch_size, n_coils, n_x, n_y, 2].
        """
        output = []
        for idx in range(data.size(self.coil_dim)):
            subselected_data = data.select(self.coil_dim, idx)
            output.append(model(subselected_data))
        output = torch.stack(output, dim=self.coil_dim)
        return output

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,  # pylint: disable=unused-argument
        initial_prediction: torch.Tensor,  # pylint: disable=unused-argument
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass of :class:`MultiDomainNet`.

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
        image = ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)
        if hasattr(self, "standardization"):
            image = self.standardization(image, sensitivity_maps)
        prediction = self._compute_model_per_coil(self.reconstruction_module, image.permute(0, 1, 4, 2, 3)).permute(
            0, 1, 3, 4, 2
        )
        return check_stacked_complex(
            coil_combination_method(prediction, sensitivity_maps, self.coil_combination_method, self.coil_dim)
        )
