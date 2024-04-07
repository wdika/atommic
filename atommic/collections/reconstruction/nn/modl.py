# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

from atommic.collections.common.parts.utils import check_stacked_complex
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.modl_base.modl_block import ConjugateGradient, ResidualNetwork
from atommic.core.classes.common import typecheck

__all__ = ["MoDL"]


class MoDL(BaseMRIReconstructionModel):
    """Implementation of the MoDL: Model Based Deep Learning Architecture for Inverse Problems.

    Adjusted to optionally perform a data consistency step (Conjugate Gradient), as presented in [Aggarwal2018]_,
    [Yaman2020]_. If dc is set to False, the network will perform a simple residual learning step.

    References
    ----------
    .. [Aggarwal2018] MoDL: Model Based Deep Learning Architecture for Inverse Problems by H.K. Aggarwal, M.P Mani, and
        Mathews Jacob in IEEE Transactions on Medical Imaging, 2018

    .. [Yaman2020] Yaman, B, Hosseini, SAH, Moeller, S, Ellermann, J, Uğurbil, K, Akçakaya, M. Self-supervised
        learning of physics-guided reconstruction neural networks without fully sampled reference data. Magn Reson
        Med. 2020; 84: 3172– 3191. https://doi.org/10.1002/mrm.28378

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`MoDL`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.unrolled_iterations = cfg_dict.get("unrolled_iterations", 10)
        self.reconstruction_module = ResidualNetwork(
            nb_res_blocks=cfg_dict.get("residual_blocks", 15),
            channels=cfg_dict.get("channels", 64),
            regularization_factor=cfg_dict.get("regularization_factor", 0.1),
        )
        self.dc = cfg_dict.get("conjugate_gradient_dc", False)
        if self.dc:
            self.mu = nn.Parameter(torch.Tensor([cfg_dict.get("penalization_weight")]), requires_grad=True)
            self.dc_block = ConjugateGradient(
                cfg_dict.get("conjugate_gradient_iterations", 10),
                self.mu,
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
                self.coil_dim,
                self.coil_combination_method,
            )

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,  # pylint: disable=unused-argument
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        initial_prediction: torch.Tensor,
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass of :class:`MoDL`.

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
        x = initial_prediction.clone()
        for _ in range(self.unrolled_iterations):
            x = self.reconstruction_module(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            if self.dc:
                x = self.dc_block(initial_prediction + self.mu * x, sensitivity_maps, mask)
        return check_stacked_complex(x)
