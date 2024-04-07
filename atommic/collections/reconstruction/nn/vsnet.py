# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import ifft2
from atommic.collections.common.parts.utils import check_stacked_complex, coil_combination_method
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.ccnn_base.ccnn_block import Conv2d
from atommic.collections.reconstruction.nn.mwcnn_base.mwcnn_block import MWCNN
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.collections.reconstruction.nn.vsnet_base.vsnet_block import (
    DataConsistencyLayer,
    VSNetBlock,
    WeightedAverageTerm,
)
from atommic.core.classes.common import typecheck

__all__ = ["VSNet"]


class VSNet(BaseMRIReconstructionModel):
    """Implementation of the Variable-Splitting Net, as presented in [Duan2019]_.

    References
    ----------
    .. [Duan2019] Duan, J. et al. (2019) Vs-net: Variable splitting network for accelerated parallel MRI
        reconstruction, Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial
        Intelligence and Lecture Notes in Bioinformatics), 11767 LNCS, pp. 713–722. doi: 10.1007/978-3-030-32251-9_78.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`VSNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        num_cascades = cfg_dict.get("num_cascades")
        self.num_cascades = cfg_dict.get("num_cascades")

        image_model_architecture = cfg_dict.get("imspace_model_architecture")
        if image_model_architecture == "CONV":
            image_model = Conv2d(
                in_channels=cfg_dict.get("imspace_in_channels", 2),
                out_channels=cfg_dict.get("imspace_out_channels", 2),
                hidden_channels=cfg_dict.get("imspace_conv_hidden_channels"),
                n_convs=cfg_dict.get("imspace_conv_n_convs"),
                batchnorm=cfg_dict.get("imspace_conv_batchnorm"),
            )
        elif image_model_architecture == "MWCNN":
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
                "VSNet is currently implemented only with image_model_architecture == 'MWCNN' or 'UNet'."
                f"Got {image_model_architecture}."
            )

        image_model = torch.nn.ModuleList([image_model] * num_cascades)
        data_consistency_model = torch.nn.ModuleList([DataConsistencyLayer()] * num_cascades)
        weighted_average_model = torch.nn.ModuleList([WeightedAverageTerm()] * num_cascades)

        self.reconstruction_module = VSNetBlock(
            denoiser_block=image_model,
            data_consistency_block=data_consistency_model,
            weighted_average_block=weighted_average_model,
            num_cascades=num_cascades,
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
        """Forward pass of :class:`VSNet`.

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
                ifft2(
                    self.reconstruction_module(y, sensitivity_maps, mask),
                    self.fft_centered,
                    self.fft_normalization,
                    self.spatial_dims,
                ),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )
        )
