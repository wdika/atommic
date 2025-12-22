# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Any, Dict, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import coil_combination_method
from atommic.collections.multitask.rs.nn.base import BaseMRIReconstructionSegmentationModel
from atommic.collections.multitask.rs.nn.idslr_base.idslr_block import DC, UnetDecoder, UnetEncoder
from atommic.collections.reconstruction.nn.rim_base.conv_layers import ConvNonlinear
from atommic.core.classes.common import typecheck

__all__ = ["SegNet"]


class SegNet(BaseMRIReconstructionSegmentationModel):
    """Implementation of the Segmentation Network MRI, as described in, as presented in [Sun2019]_.

    References
    ----------
    .. [Sun2019] Sun, L., Fan, Z., Ding, X., Huang, Y., Paisley, J. (2019). Joint CS-MRI Reconstruction and
        Segmentation with a Unified Deep Network. In: Chung, A., Gee, J., Yushkevich, P., Bao, S. (eds) Information
        Processing in Medical Imaging. IPMI 2019. Lecture Notes in Computer Science(), vol 11492. Springer, Cham.
        https://doi.org/10.1007/978-3-030-20351-1_38

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`SegNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object.
        trainer : Trainer, optional
            PyTorch Lightning trainer object. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module", True)

        self.dimensionality = cfg_dict.get("dimensionality", 2)
        if self.dimensionality != 2:
            raise NotImplementedError(f"Currently only 2D is supported for segmentation, got {self.dimensionality}D.")

        self.input_channels = cfg_dict.get("input_channels", 2)
        reconstruction_out_chans = cfg_dict.get("reconstruction_module_output_channels", 2)
        self.segmentation_module_output_channels = cfg_dict.get("segmentation_module_output_channels", 1)
        chans = cfg_dict.get("channels", 32)
        num_pools = cfg_dict.get("num_pools", 4)
        drop_prob = cfg_dict.get("drop_prob", 0.0)
        normalize = cfg_dict.get("normalize", False)
        padding = cfg_dict.get("padding", False)
        padding_size = cfg_dict.get("padding_size", 11)
        self.norm_groups = cfg_dict.get("norm_groups", 2)
        num_cascades = cfg_dict.get("num_cascades", 5)

        self.reconstruction_encoder = nn.ModuleList(
            [
                UnetEncoder(
                    chans=chans,
                    num_pools=num_pools,
                    in_chans=self.input_channels,
                    drop_prob=drop_prob,
                    normalize=normalize,
                    padding=padding,
                    padding_size=padding_size,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.reconstruction_decoder = nn.ModuleList(
            [
                UnetDecoder(
                    chans=chans,
                    num_pools=num_pools,
                    out_chans=reconstruction_out_chans,
                    drop_prob=drop_prob,
                    normalize=normalize,
                    padding=padding,
                    padding_size=padding_size,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.segmentation_decoder = nn.ModuleList(
            [
                UnetDecoder(
                    chans=chans,
                    num_pools=num_pools,
                    out_chans=self.segmentation_module_output_channels,
                    drop_prob=drop_prob,
                    normalize=normalize,
                    padding=padding,
                    padding_size=padding_size,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )

        self.segmentation_final_layer = torch.nn.Sequential(
            ConvNonlinear(
                self.segmentation_module_output_channels * num_cascades,
                self.segmentation_module_output_channels,
                conv_dim=cfg_dict.get("segmentation_final_layer_conv_dim", 2),
                kernel_size=cfg_dict.get("segmentation_final_layer_kernel_size", 3),
                dilation=cfg_dict.get("segmentation_final_layer_dilation", 1),
                bias=cfg_dict.get("segmentation_final_layer_bias", False),
                nonlinear=cfg_dict.get("segmentation_final_layer_nonlinear", "relu"),
            )
        )

        self.magnitude_input = cfg_dict.get("magnitude_input", True)
        self.normalize_segmentation_output = cfg_dict.get("normalize_segmentation_output", True)

        self.dc = DC()

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,  # pylint: disable=unused-argument
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> Tuple[Any, Any]:
        """Forward pass of :class:`SegNet`.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        init_reconstruction_pred : torch.Tensor
            Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2]
        target_reconstruction : torch.Tensor
            Target reconstruction. Shape [batch_size, n_x, n_y, 2]
        sigma : float
            Standard deviation of the noise. Default is ``1.0``.

        Returns
        -------
        Tuple[Union[List, torch.Tensor], torch.Tensor]
            Tuple containing the predicted reconstruction and segmentation.
        """
        if self.consecutive_slices > 1:
            batch, slices = y.shape[0], y.shape[1]
            y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])
            sensitivity_maps = sensitivity_maps.reshape(
                sensitivity_maps.shape[0] * sensitivity_maps.shape[1],
                *sensitivity_maps.shape[2:],
            )
            mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])

        # In case of deviating number of coils, we need to pad up to maximum number of coils == number of input \
        # channels for the reconstruction module
        num_coils = y.shape[1]
        if num_coils * 2 != self.input_channels:
            num_coils_to_add = (self.input_channels - num_coils * 2) // 2
            dummy_coil_data = torch.zeros_like(torch.movedim(y, self.coil_dim, 0)[0]).unsqueeze(self.coil_dim)
            for _ in range(num_coils_to_add):
                y = torch.cat([y, dummy_coil_data], dim=self.coil_dim)
                sensitivity_maps = torch.cat([sensitivity_maps, dummy_coil_data], dim=self.coil_dim)

        y_prediction = y.clone()
        pred_segmentations = []
        for re, rd, sd in zip(self.reconstruction_encoder, self.reconstruction_decoder, self.segmentation_decoder):
            init_reconstruction_pred = ifft2(
                y_prediction, self.fft_centered, self.fft_normalization, self.spatial_dims
            )
            output = re(init_reconstruction_pred)
            reconstruction_encoder_prediction, padding_size = output[0].copy(), output[2]

            pred_segmentation_input = reconstruction_encoder_prediction
            if self.magnitude_input:
                pred_segmentation_input = [torch.abs(x) for x in pred_segmentation_input]

            pred_segmentations.append(sd(pred_segmentation_input, iscomplex=False, pad_sizes=padding_size))
            reconstruction_decoder_prediction = rd(*output)
            reconstruction_decoder_prediction_kspace = fft2(
                reconstruction_decoder_prediction,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            y_prediction = self.dc(reconstruction_decoder_prediction_kspace, y, mask)

        pred_reconstruction = self.process_intermediate_pred(y_prediction, sensitivity_maps, True)

        pred_segmentation = self.segmentation_final_layer(torch.cat(pred_segmentations, dim=1))

        if self.normalize_segmentation_output:
            pred_segmentation = (pred_segmentation - pred_segmentation.min()) / (
                pred_segmentation.max() - pred_segmentation.min()
            )

        pred_segmentation = torch.abs(pred_segmentation)

        pred_segmentations.append(pred_segmentation)

        if self.consecutive_slices > 1:
            # get batch size and number of slices from y, because if the reconstruction module is used they will not
            # be saved before
            pred_reconstruction = pred_reconstruction.view([batch, slices, *pred_reconstruction.shape[1:]])
            pred_segmentations = [x.view([batch, slices, *x.shape[1:]]) for x in pred_segmentations]

        return pred_reconstruction, pred_segmentations

    def process_segmentation_loss(self, target: torch.Tensor, prediction: torch.Tensor, attrs: Dict) -> Dict:
        """Processes the segmentation loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, nr_classes, n_x, n_y].
        prediction : torch.Tensor
            Prediction of shape [batch_size, nr_classes, n_x, n_y].
        attrs : Dict
            Attributes of the data with pre normalization values.

        Returns
        -------
        Dict
            Dictionary containing the (multiple) loss values. For example, if the cross entropy loss and the dice loss
            are used, the dictionary will contain the keys ``cross_entropy_loss``, ``dice_loss``, and
            (combined) ``segmentation_loss``.
        """
        if self.unnormalize_loss_inputs:
            target, prediction = self.__unnormalize_for_loss_or_log__(  # type: ignore
                target, prediction, None, attrs, attrs["r"]
            )
        losses = {}
        for name, loss_func in self.segmentation_losses.items():
            cascades_loss = []
            for i in range(len(prediction)):  # pylint: disable=consider-using-enumerate
                loss = loss_func(target, prediction[i])
                if isinstance(loss, tuple):
                    # In case of the dice loss, the loss is a tuple of the form (dice, dice loss)
                    loss = loss[1]
                cascades_loss.append(loss)
            losses[name] = torch.stack(cascades_loss).mean().to(target.device)
        return self.total_segmentation_loss(**losses) * self.total_segmentation_loss_weight

    def process_intermediate_pred(
        self,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        do_coil_combination: bool = False,
    ) -> torch.Tensor:
        """Processes the intermediate prediction.

        Parameters
        ----------
        prediction : torch.Tensor
            Intermediate prediction. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        do_coil_combination : bool
            Whether to do coil combination. In this case the prediction is in k-space. Default is ``False``.

        Returns
        -------
        torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Processed prediction.
        """
        # Take the last time step of the prediction
        if do_coil_combination:
            prediction = ifft2(
                prediction,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            prediction = coil_combination_method(
                prediction, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
            )
        prediction = torch.view_as_complex(prediction)
        return prediction
