# coding=utf-8
__author__ = "Dimitris Karkalousos"

import warnings

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.ccnn_base.ccnn_block import Conv2d
from atommic.collections.reconstruction.nn.crossdomain_base.crossdomain_block import CrossDomainNetwork, MultiCoil
from atommic.collections.reconstruction.nn.didn_base.didn_block import DIDN
from atommic.collections.reconstruction.nn.mwcnn_base.mwcnn_block import MWCNN
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.core.classes.common import typecheck

__all__ = ["XPDNet"]


class XPDNet(BaseMRIReconstructionModel):
    """Implementation of the XPDNet, as presented in [Ramzi2021]_.

    References
    ----------
    .. [Ramzi2021] Ramzi, Zaccharie, et al. “XPDNet for MRI Reconstruction: An Application to the 2020 FastMRI
        Challenge. ArXiv:2010.07290 [Physics, Stat], July 2021. arXiv.org, http://arxiv.org/abs/2010.07290.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`XPDNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        num_primal = cfg_dict.get("num_primal")
        num_dual = cfg_dict.get("num_dual")
        num_iter = cfg_dict.get("num_iter")

        kspace_model_architecture = cfg_dict.get("kspace_model_architecture")
        dual_conv_hidden_channels = cfg_dict.get("dual_conv_hidden_channels", 64)
        dual_conv_num_dubs = cfg_dict.get("dual_conv_num_dubs", 2)
        dual_conv_batchnorm = cfg_dict.get("dual_conv_batchnorm", True)
        dual_didn_hidden_channels = cfg_dict.get("dual_didn_hidden_channels", 64)
        dual_didn_num_dubs = cfg_dict.get("dual_didn_num_dubs", 2)
        dual_didn_num_convs_recon = cfg_dict.get("dual_didn_num_convs_recon", True)

        if cfg_dict.get("use_primal_only"):
            kspace_model_list = None
            num_dual = 1
        elif kspace_model_architecture == "CONV":
            kspace_model_list = torch.nn.ModuleList(
                [
                    MultiCoil(
                        Conv2d(
                            cfg_dict.get("kspace_in_channels") * (num_dual + num_primal + 1),
                            cfg_dict.get("kspace_out_channels") * num_dual,
                            dual_conv_hidden_channels,
                            dual_conv_num_dubs,
                            batchnorm=dual_conv_batchnorm,
                        )
                    )
                    for _ in range(num_iter)
                ]
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model_list = torch.nn.ModuleList(
                [
                    MultiCoil(
                        DIDN(
                            in_channels=cfg_dict.get("kspace_in_channels") * (num_dual + num_primal + 1),
                            out_channels=cfg_dict.get("kspace_out_channels") * num_dual,
                            hidden_channels=dual_didn_hidden_channels,
                            num_dubs=dual_didn_num_dubs,
                            num_convs_recon=dual_didn_num_convs_recon,
                        )
                    )
                    for _ in range(num_iter)
                ]
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            kspace_model_list = torch.nn.ModuleList(
                [
                    MultiCoil(
                        NormUnet(
                            cfg_dict.get("kspace_unet_num_filters"),
                            cfg_dict.get("kspace_unet_num_pool_layers"),
                            in_chans=cfg_dict.get("kspace_in_channels") * (num_dual + num_primal + 1),
                            out_chans=cfg_dict.get("kspace_out_channels") * num_dual,
                            drop_prob=cfg_dict.get("kspace_unet_dropout_probability"),
                            padding_size=cfg_dict.get("kspace_unet_padding_size"),
                            normalize=cfg_dict.get("kspace_unet_normalize"),
                        ),
                        coil_to_batch=True,
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(
                "XPDNet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN'."
                f"Got kspace_model_architecture == {kspace_model_architecture}."
            )

        image_model_architecture = cfg_dict.get("image_model_architecture")
        mwcnn_hidden_channels = cfg_dict.get("mwcnn_hidden_channels", 16)
        mwcnn_num_scales = cfg_dict.get("mwcnn_num_scales", 2)
        mwcnn_bias = cfg_dict.get("mwcnn_bias", True)
        mwcnn_batchnorm = cfg_dict.get("mwcnn_batchnorm", True)

        if image_model_architecture == "MWCNN":
            image_model_list = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        MWCNN(
                            input_channels=cfg_dict.get("imspace_in_channels") * (num_primal + num_dual),
                            first_conv_hidden_channels=mwcnn_hidden_channels,
                            num_scales=mwcnn_num_scales,
                            bias=mwcnn_bias,
                            batchnorm=mwcnn_batchnorm,
                        ),
                        torch.nn.Conv2d(2 * (num_primal + num_dual), 2 * num_primal, kernel_size=3, padding=1),
                    )
                    for _ in range(num_iter)
                ]
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            image_model_list = torch.nn.ModuleList(
                [
                    NormUnet(
                        cfg_dict.get("imspace_unet_num_filters"),
                        cfg_dict.get("imspace_unet_num_pool_layers"),
                        in_chans=cfg_dict.get("imspace_in_channels") * (num_primal + num_dual),
                        out_chans=cfg_dict.get("imspace_out_channels") * num_primal,
                        drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
                        padding_size=cfg_dict.get("imspace_unet_padding_size"),
                        normalize=cfg_dict.get("imspace_unet_normalize"),
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(f"Image model architecture {image_model_architecture} not found for XPDNet.")

        self.num_cascades = cfg_dict.get("num_cascades")

        self.reconstruction_module = CrossDomainNetwork(
            image_model_list=image_model_list,
            kspace_model_list=kspace_model_list,
            domain_sequence="KI" * num_iter,
            image_buffer_size=num_primal,
            kspace_buffer_size=num_dual,
            normalize_image=cfg_dict.get("normalize_image"),
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
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
        """Forward pass of :class:`XPDNet`.

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
        prediction = self.reconstruction_module(y, sensitivity_maps, mask)
        # filter UserWarning: ComplexHalf support is experimental and many operators don't support it yet.
        # TODO: remove this when PyTorch fixes the issue.
        warnings.filterwarnings("ignore", category=UserWarning)
        return torch.view_as_real(prediction[..., 0] + 1j * prediction[..., 1])
