# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import complex_conj, complex_mul
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.ccnn_base.ccnn_block import Conv2d
from atommic.collections.reconstruction.nn.didn_base.didn_block import DIDN
from atommic.collections.reconstruction.nn.mwcnn_base.mwcnn_block import MWCNN
from atommic.collections.reconstruction.nn.primaldualnet_base.primaldualnet_block import DualNet, PrimalNet
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.core.classes.common import typecheck

__all__ = ["LPDNet"]


class LPDNet(BaseMRIReconstructionModel):
    """Implementation of the Learned Primal Dual network, inspired by [Adler2018]_.

    References
    ----------
    .. [Adler2018] Adler, Jonas, and Ozan Öktem. “Learned Primal-Dual Reconstruction.” IEEE Transactions on Medical
        Imaging, vol. 37, no. 6, June 2018, pp. 1322–32. arXiv.org, https://doi.org/10.1109/TMI.2018.2799231.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`LPDNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.dimensionality = cfg_dict.get("dimensionality", 2)
        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

        self.num_iter = cfg_dict.get("num_iter")
        self.num_primal = cfg_dict.get("num_primal")
        self.num_dual = cfg_dict.get("num_dual")

        primal_model_architecture = cfg_dict.get("primal_model_architecture")

        if primal_model_architecture == "MWCNN":
            primal_model = torch.nn.Sequential(
                *[
                    MWCNN(
                        input_channels=cfg_dict.get("primal_in_channels") * (self.num_primal + 1),
                        first_conv_hidden_channels=cfg_dict.get("primal_mwcnn_hidden_channels"),
                        num_scales=cfg_dict.get("primal_mwcnn_num_scales"),
                        bias=cfg_dict.get("primal_mwcnn_bias"),
                        batchnorm=cfg_dict.get("primal_mwcnn_batchnorm"),
                    ),
                    torch.nn.Conv2d(
                        cfg_dict.get("primal_out_channels") * (self.num_primal + 1),
                        cfg_dict.get("primal_out_channels") * self.num_primal,
                        kernel_size=1,
                    ),
                ]
            )
        elif primal_model_architecture in ["UNET", "NORMUNET"]:
            primal_model = NormUnet(
                cfg_dict.get("primal_unet_num_filters"),
                cfg_dict.get("primal_unet_num_pool_layers"),
                in_chans=cfg_dict.get("primal_in_channels") * (self.num_primal + 1),
                out_chans=cfg_dict.get("primal_out_channels") * self.num_primal,
                drop_prob=cfg_dict.get("primal_unet_dropout_probability"),
                padding_size=cfg_dict.get("primal_unet_padding_size"),
                normalize=cfg_dict.get("primal_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "LPDNet is currently implemented for primal_model_architecture == 'CONV' or 'UNet'."
                f"Got primal_model_architecture == {primal_model_architecture}."
            )

        dual_model_architecture = cfg_dict.get("dual_model_architecture")

        if dual_model_architecture == "CONV":
            dual_model = Conv2d(
                in_channels=cfg_dict.get("dual_in_channels") * (self.num_dual + 2),
                out_channels=cfg_dict.get("dual_out_channels") * self.num_dual,
                hidden_channels=cfg_dict.get("kspace_conv_hidden_channels"),
                n_convs=cfg_dict.get("kspace_conv_n_convs"),
                batchnorm=cfg_dict.get("kspace_conv_batchnorm"),
            )
        elif dual_model_architecture == "DIDN":
            dual_model = DIDN(
                in_channels=cfg_dict.get("dual_in_channels") * (self.num_dual + 2),
                out_channels=cfg_dict.get("dual_out_channels") * self.num_dual,
                hidden_channels=cfg_dict.get("kspace_didn_hidden_channels"),
                num_dubs=cfg_dict.get("kspace_didn_num_dubs"),
                num_convs_recon=cfg_dict.get("kspace_didn_num_convs_recon"),
            )
        elif dual_model_architecture in ["UNET", "NORMUNET"]:
            dual_model = NormUnet(
                cfg_dict.get("dual_unet_num_filters"),
                cfg_dict.get("dual_unet_num_pool_layers"),
                in_chans=cfg_dict.get("dual_in_channels") * (self.num_dual + 2),
                out_chans=cfg_dict.get("dual_out_channels") * self.num_dual,
                drop_prob=cfg_dict.get("dual_unet_dropout_probability"),
                padding_size=cfg_dict.get("dual_unet_padding_size"),
                normalize=cfg_dict.get("dual_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "LPDNet is currently implemented for dual_model_architecture == 'CONV' or 'DIDN' or 'UNet'."
                f"Got dual_model_architecture == {dual_model_architecture}."
            )

        self.primal_net = torch.nn.ModuleList(
            [PrimalNet(self.num_primal, primal_architecture=primal_model) for _ in range(self.num_iter)]
        )
        self.dual_net = torch.nn.ModuleList(
            [DualNet(self.num_dual, dual_architecture=dual_model) for _ in range(self.num_iter)]
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
        """Forward pass of :class:`LPDNet`.

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
        dual_buffer = torch.cat([y] * self.num_dual, -1).to(y.device)
        primal_buffer = torch.cat([initial_prediction] * self.num_primal, -1).to(y.device)

        for idx in range(self.num_iter):
            # Dual
            f_2 = primal_buffer[..., 2:4].clone()
            f_2 = torch.where(
                mask == 0,
                torch.tensor([0.0], dtype=f_2.dtype).to(f_2.device),
                fft2(
                    complex_mul(f_2.unsqueeze(self.coil_dim), sensitivity_maps),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ).type(f_2.type()),
            )
            dual_buffer = self.dual_net[idx](dual_buffer, f_2, y)

            # Primal
            h_1 = dual_buffer[..., 0:2].clone()
            # needed for python3.9
            h_1 = torch.view_as_real(h_1[..., 0] + 1j * h_1[..., 1])
            h_1 = complex_mul(
                ifft2(
                    torch.where(mask == 0, torch.tensor([0.0], dtype=h_1.dtype).to(h_1.device), h_1),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                complex_conj(sensitivity_maps),
            ).sum(self.coil_dim)
            primal_buffer = self.primal_net[idx](primal_buffer, h_1)

        primal_buffer = primal_buffer[..., 0:2]

        return torch.view_as_real(primal_buffer[..., 0] + 1j * primal_buffer[..., 1])
