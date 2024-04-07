# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import ifft2
from atommic.collections.common.parts.utils import check_stacked_complex, coil_combination_method
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.core.classes.common import typecheck

__all__ = ["ZF"]


class ZF(BaseMRIReconstructionModel):
    """Zero-Filled reconstruction using either root-sum-of-squares (RSS) or SENSE (SENSitivity Encoding, as presented
    in [Pruessmann1999]_).

    References
    ----------
    .. [Pruessmann1999] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast MRI.
        Magn Reson Med 1999; 42:952-962.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`ZF`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

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
        """Forward pass of :class:`ZF`.

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
                ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                sensitivity_maps,
                self.coil_combination_method.upper(),
                self.coil_dim,
            )
        )
