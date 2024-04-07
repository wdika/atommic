# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
from typing import Dict, List, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import fft2
from atommic.collections.common.parts.utils import check_stacked_complex, expand_op
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.rim_base.rim_block import RIMBlock
from atommic.core.classes.common import typecheck

__all__ = ["CIRIM"]


class CIRIM(BaseMRIReconstructionModel):
    """Implementation of the Cascades of Independently Recurrent Inference Machines, as presented in
    [Karkalousos2022]_.

    References
    ----------
    .. [Karkalousos2022] Karkalousos D, Noteboom S, Hulst HE, Vos FM, Caan MWA. Assessment of data consistency through
        cascades of independently recurrent inference machines for fast and robust accelerated MRI reconstruction.
        Phys Med Biol. 2022 Jun 8;67(12). doi: 10.1088/1361-6560/ac6cc2. PMID: 35508147.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`CIRIM`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # make time-steps size divisible by 8 for fast fp16 training
        self.time_steps = 8 * math.ceil(cfg_dict.get("time_steps") / 8)
        self.no_dc = cfg_dict.get("no_dc")
        self.reconstruction_module = torch.nn.ModuleList(
            [
                RIMBlock(
                    recurrent_layer=cfg_dict.get("recurrent_layer"),
                    conv_filters=cfg_dict.get("conv_filters"),
                    conv_kernels=cfg_dict.get("conv_kernels"),
                    conv_dilations=cfg_dict.get("conv_dilations"),
                    conv_bias=cfg_dict.get("conv_bias"),
                    recurrent_filters=cfg_dict.get("recurrent_filters"),
                    recurrent_kernels=cfg_dict.get("recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("recurrent_bias"),
                    depth=cfg_dict.get("depth"),
                    time_steps=self.time_steps,
                    conv_dim=cfg_dict.get("conv_dim"),
                    no_dc=self.no_dc,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    dimensionality=cfg_dict.get("dimensionality"),
                    coil_combination_method=self.coil_combination_method,
                )
                for _ in range(cfg_dict.get("num_cascades"))
            ]
        )

        # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
        self.keep_prediction = cfg_dict.get("keep_prediction")

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        initial_prediction: torch.Tensor,
        sigma: float = 1.0,
    ) -> Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]:
        """Forward pass of :class:`CIRIM`.

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
        List of torch.Tensor
            List of the intermediate predictions for each cascade. Shape [batch_size, n_x, n_y].
        """
        prediction = y.clone()
        initial_prediction = None if initial_prediction is None or initial_prediction.dim() < 4 else initial_prediction
        hx = None
        cascades_predictions = []
        for i, cascade in enumerate(self.reconstruction_module):
            # Forward pass through the cascades
            prediction, hx = cascade(
                prediction,
                y,
                sensitivity_maps,
                mask,
                initial_prediction if i == 0 else prediction,
                hx,
                sigma,
                keep_prediction=False if i == 0 else self.keep_prediction,
            )
            cascades_predictions.append([check_stacked_complex(p) for p in prediction])
            prediction = prediction[-1]
        return cascades_predictions

    def process_reconstruction_loss(  # noqa: MC0001
        self,
        target: torch.Tensor,
        prediction: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        attrs: Union[Dict, torch.Tensor],
        r: Union[int, torch.Tensor],
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """Processes the reconstruction loss for the CIRIM model. It differs from the base class in that it can handle
        multiple cascades and time steps.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        mask : torch.Tensor
            Mask of shape [batch_size, n_x, n_y, 2]. It will be used if self.ssdu is True, to enforce data consistency
            on the prediction.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            atommic.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        # If kspace reconstruction loss is used, the target needs to be transformed to k-space.
        if self.kspace_reconstruction_loss:
            # If inputs are complex, then they need to be viewed as real.
            if target.shape[-1] != 2 and torch.is_complex(target):
                target = torch.view_as_real(target)
            # If SSDU is used, then the coil-combined inputs need to be expanded to multiple coils using the
            # sensitivity maps.
            if self.ssdu:
                target = expand_op(target, sensitivity_maps, self.coil_dim)
            # Transform to k-space.
            target = fft2(target, self.fft_centered, self.fft_normalization, self.spatial_dims)
            # Ensure loss inputs are both viewed in the same way.
            target = self.__abs_output__(target)
        elif not self.unnormalize_loss_inputs:
            # Ensure loss inputs are both viewed in the same way.
            target = self.__abs_output__(target)
            # Normalize inputs to [0, 1]
            target = torch.abs(target / torch.max(torch.abs(target)))

        def compute_reconstruction_loss(t, p, s):
            if self.unnormalize_loss_inputs:
                # we do the unnormalization here to avoid explicitly iterating through list of predictions, which
                # might be a list of lists.
                t, p, s = self.__unnormalize_for_loss_or_log__(t, p, s, attrs, r)

            # If kspace reconstruction loss is used, the target needs to be transformed to k-space.
            if self.kspace_reconstruction_loss:
                # If inputs are complex, then they need to be viewed as real.
                if p.shape[-1] != 2 and torch.is_complex(p):
                    p = torch.view_as_real(p)
                # If SSDU is used, then the coil-combined inputs need to be expanded to multiple coils using the
                # sensitivity maps.
                if self.ssdu:
                    p = expand_op(p, s, self.coil_dim)
                # Transform to k-space.
                p = fft2(p, self.fft_centered, self.fft_normalization, self.spatial_dims)
                # If SSDU is used, then apply the mask to the prediction to enforce data consistency.
                if self.ssdu:
                    p = p * mask
                # Ensure loss inputs are both viewed in the same way.
                p = self.__abs_output__(p)
            elif not self.unnormalize_loss_inputs:
                p = self.__abs_output__(p)
                # Normalize inputs to [0, 1]
                p = torch.abs(p / torch.max(torch.abs(p)))

            if "ssim" in str(loss_func).lower():
                return loss_func(
                    t.unsqueeze(dim=self.coil_dim),
                    p.unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(
                        [max(torch.max(t).item(), torch.max(p).item()) - min(torch.min(t).item(), torch.min(p).item())]
                    )
                    .unsqueeze(dim=0)
                    .to(t.device),
                )

            return loss_func(t, p)

        if self.accumulate_predictions:
            cascades_weights = torch.logspace(-1, 0, steps=len(prediction)).to(target.device)
            cascades_loss = []
            for cascade_pred in prediction:
                time_steps_weights = torch.logspace(-1, 0, steps=self.time_steps).to(target.device)
                if self.num_echoes > 0:
                    time_steps_loss = [
                        torch.mean(
                            torch.stack(
                                [
                                    compute_reconstruction_loss(
                                        target[echo].unsqueeze(0), time_step_pred[echo].unsqueeze(0), sensitivity_maps
                                    )
                                    for echo in range(target.shape[0])
                                ]
                            )
                        ).to(target)
                        for time_step_pred in cascade_pred
                    ]
                else:
                    time_steps_loss = [
                        compute_reconstruction_loss(target, time_step_pred, sensitivity_maps)
                        for time_step_pred in cascade_pred
                    ]
                cascade_loss = sum(x * w for x, w in zip(time_steps_loss, time_steps_weights)) / self.time_steps
                cascades_loss.append(cascade_loss)
            loss = sum(x * w for x, w in zip(cascades_loss, cascades_weights)) / len(prediction)
        else:
            # keep the last prediction of the last cascade
            prediction = prediction[-1][-1]
            loss = compute_reconstruction_loss(target, prediction, sensitivity_maps)

        return loss
