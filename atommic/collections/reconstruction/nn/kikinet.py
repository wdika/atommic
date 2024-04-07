# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import check_stacked_complex, complex_conj, complex_mul
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.ccnn_base.ccnn_block import Conv2d
from atommic.collections.reconstruction.nn.crossdomain_base.crossdomain_block import MultiCoil
from atommic.collections.reconstruction.nn.didn_base.didn_block import DIDN
from atommic.collections.reconstruction.nn.mwcnn_base.mwcnn_block import MWCNN
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.core.classes.common import typecheck

__all__ = ["KIKINet"]


class KIKINet(BaseMRIReconstructionModel):
    """Based on KIKINet implementation. Modified to work with multi-coil k-space data, as presented in [Taejoon2018]_.

    References
    ----------
    .. [Taejoon2018] Eo, Taejoon, et al. “KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing
        Undersampled Magnetic Resonance Images.” Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp.
        2188–201. PubMed, https://doi.org/10.1002/mrm.27201.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`KIKINet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = cfg_dict.get("num_iter")
        self.no_dc = cfg_dict.get("no_dc")

        kspace_model_architecture = cfg_dict.get("kspace_model_architecture")

        if kspace_model_architecture == "CONV":
            kspace_model = Conv2d(
                in_channels=cfg_dict.get("kspace_in_channels", 2),
                out_channels=cfg_dict.get("kspace_out_channels", 2),
                hidden_channels=cfg_dict.get("kspace_conv_hidden_channels"),
                n_convs=cfg_dict.get("kspace_conv_n_convs"),
                batchnorm=cfg_dict.get("kspace_conv_batchnorm"),
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model = DIDN(
                in_channels=cfg_dict.get("kspace_in_channels", 2),
                out_channels=cfg_dict.get("kspace_out_channels", 2),
                hidden_channels=cfg_dict.get("kspace_didn_hidden_channels"),
                num_dubs=cfg_dict.get("kspace_didn_num_dubs"),
                num_convs_recon=cfg_dict.get("kspace_didn_num_convs_recon"),
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            kspace_model = NormUnet(
                cfg_dict.get("kspace_unet_num_filters"),
                cfg_dict.get("kspace_unet_num_pool_layers"),
                in_chans=cfg_dict.get("kspace_in_channels", 2),
                out_chans=cfg_dict.get("kspace_out_channels", 2),
                drop_prob=cfg_dict.get("kspace_unet_dropout_probability"),
                padding_size=cfg_dict.get("kspace_unet_padding_size"),
                normalize=cfg_dict.get("kspace_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "KIKINet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN' or 'UNet'."
                f"Got kspace_model_architecture == {kspace_model_architecture}."
            )

        image_model_architecture = cfg_dict.get("imspace_model_architecture")

        if image_model_architecture == "MWCNN":
            image_model = MWCNN(
                input_channels=cfg_dict.get("imspace_in_channels", 2),
                first_conv_hidden_channels=cfg_dict.get("image_mwcnn_hidden_channels"),
                num_scales=cfg_dict.get("image_mwcnn_num_scales"),
                bias=cfg_dict.get("image_mwcnn_bias"),
                batchnorm=cfg_dict.get("image_mwcnn_batchnorm"),
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            image_model = NormUnet(
                cfg_dict.get("imspace_unet_num_filters"),
                cfg_dict.get("imspace_unet_num_pool_layers"),
                in_chans=cfg_dict.get("imspace_in_channels", 2),
                out_chans=cfg_dict.get("imspace_out_channels", 2),
                drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
                padding_size=cfg_dict.get("imspace_unet_padding_size"),
                normalize=cfg_dict.get("imspace_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "KIKINet is currently implemented only with image_model_architecture == 'MWCNN' or 'UNet'."
                f"Got {image_model_architecture}."
            )

        self.image_model_list = torch.nn.ModuleList([image_model] * self.num_iter)
        self.kspace_model_list = torch.nn.ModuleList([MultiCoil(kspace_model, coil_dim=1)] * self.num_iter)

        self.dc_weight = torch.nn.Parameter(torch.ones(1))

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
        """Forward pass of :class:`KIKINet`.

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
        kspace = y.clone()
        zero = torch.zeros(1, 1, 1, 1, 1).to(kspace)

        for idx in range(self.num_iter):
            soft_dc = torch.where(mask.bool(), kspace - y, zero) * self.dc_weight

            kspace = self.kspace_model_list[idx](kspace)
            if kspace.shape[-1] != 2:
                kspace = kspace.permute(0, 1, 3, 4, 2).to(y)
                # this is necessary, but why?
                kspace = torch.view_as_real(kspace[..., 0] + 1j * kspace[..., 1])

            image = complex_mul(
                ifft2(
                    kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                complex_conj(sensitivity_maps),
            ).sum(self.coil_dim)
            image = self.image_model_list[idx](image.unsqueeze(self.coil_dim)).squeeze(self.coil_dim)

            if not self.no_dc:
                image = fft2(
                    complex_mul(image.unsqueeze(self.coil_dim), sensitivity_maps),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ).type(image.type())
                image = kspace - soft_dc - image
                image = complex_mul(
                    ifft2(
                        image,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    complex_conj(sensitivity_maps),
                ).sum(self.coil_dim)

            if idx < self.num_iter - 1:
                kspace = fft2(
                    complex_mul(image.unsqueeze(self.coil_dim), sensitivity_maps),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ).type(image.type())

        return check_stacked_complex(image)
