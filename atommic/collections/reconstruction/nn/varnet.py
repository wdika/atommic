# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import ifft2
from atommic.collections.common.parts.utils import check_stacked_complex, coil_combination_method
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.collections.reconstruction.nn.varnet_base.varnet_block import VarNetBlock
from atommic.core.classes.common import typecheck

__all__ = ["VarNet"]


class VarNet(BaseMRIReconstructionModel):
    """Implementation of the End-to-end Variational Network (VN), as presented in [Sriram2020]_.

    References
    ----------
    .. [Sriram2020] Sriram A, Zbontar J, Murrell T, Defazio A, Zitnick CL, Yakubova N, Knoll F, Johnson P. End-to-end
        variational networks for accelerated MRI reconstruction. InInternational Conference on Medical Image Computing
        and Computer-Assisted Intervention 2020 Oct 4 (pp. 64-73). Springer, Cham.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`VarNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = cfg_dict.get("no_dc")
        self.num_cascades = cfg_dict.get("num_cascades")

        # Cascades of VN blocks
        self.cascades = torch.nn.ModuleList(
            [
                VarNetBlock(
                    NormUnet(
                        chans=cfg_dict.get("channels", 18),
                        num_pools=cfg_dict.get("pooling_layers", 4),
                        in_chans=cfg_dict.get("in_chans", 2),
                        out_chans=cfg_dict.get("out_chans", 2),
                        drop_prob=cfg_dict.get("dropout", 0.0),
                        padding_size=cfg_dict.get("padding_size", 11),
                        normalize=cfg_dict.get("normalize", True),
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    no_dc=self.no_dc,
                )
                for _ in range(self.num_cascades)
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
        """Forward pass of :class:`VarNet`.

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
        for cascade in self.cascades:
            prediction = cascade(prediction, y, sensitivity_maps, mask)
        return check_stacked_complex(
            coil_combination_method(
                ifft2(prediction, self.fft_centered, self.fft_normalization, self.spatial_dims),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )
        )
