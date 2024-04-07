# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.core.classes.common import typecheck

__all__ = ["UNet"]


class UNet(BaseMRIReconstructionModel):
    """Implementation of the UNet, as presented in [Ronneberger2015]_.

    References
    ----------
    .. [Ronneberger2015] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
        image segmentation. In International Conference on Medical image computing and computer-assisted intervention,
        pages 234–241. Springer, 2015.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`UNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.in_channels = cfg_dict.get("in_channels", 2)
        self.reconstruction_module = NormUnet(
            chans=cfg_dict.get("channels"),
            num_pools=cfg_dict.get("pooling_layers"),
            in_chans=self.in_channels,
            out_chans=cfg_dict.get("out_channels", 2),
            padding_size=cfg_dict.get("padding_size", 11),
            drop_prob=cfg_dict.get("dropout", 0.0),
            normalize=cfg_dict.get("normalize", True),
            norm_groups=cfg_dict.get("norm_groups", 2),
        )

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,  # pylint: disable=unused-argument
        sensitivity_maps: torch.Tensor,  # pylint: disable=unused-argument
        mask: torch.Tensor,  # pylint: disable=unused-argument
        initial_prediction: torch.Tensor,
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass of :class:`UNet`.

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
        if self.in_channels == 1 and initial_prediction.shape[-1] == 2:
            initial_prediction = torch.abs(torch.view_as_complex(initial_prediction))
        prediction = self.reconstruction_module(initial_prediction.unsqueeze(self.coil_dim)).squeeze(self.coil_dim)
        if self.in_channels == 2:
            prediction = torch.view_as_complex(prediction)
        return prediction
