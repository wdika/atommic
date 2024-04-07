# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
from typing import Dict, List, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import Tensor

from atommic.collections.common.parts.fft import fft2
from atommic.collections.common.parts.utils import coil_combination_method, rnn_weights_init
from atommic.collections.quantitative.nn.base import BaseqMRIReconstructionModel, SignalForwardModel
from atommic.collections.quantitative.nn.qrim_base.qrim_block import qRIMBlock
from atommic.collections.quantitative.parts.transforms import R2star_B0_S0_phi_mapping
from atommic.collections.reconstruction.nn.rim_base.rim_block import RIMBlock
from atommic.core.classes.common import typecheck

__all__ = ["qCIRIM"]


class qCIRIM(BaseqMRIReconstructionModel):
    """Implementation of the quantitative Recurrent Inference Machines (qRIM), as presented in [Zhang2022]_.

    Also implements the qCIRIM model, which is a qRIM model with cascades.

    References
    ----------
    .. [Zhang2022] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW.
        A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent
        inference machine. NeuroImage. 2022 Dec 1;264:119680.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`qCIRIM`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            Trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        quantitative_module_dimensionality = cfg_dict.get("quantitative_module_dimensionality")
        if quantitative_module_dimensionality != 2:
            raise ValueError(
                f"Only 2D is currently supported for qMRI models.Found {quantitative_module_dimensionality}"
            )

        quantitative_module_no_dc = cfg_dict.get("quantitative_module_no_dc")
        if not quantitative_module_no_dc:
            raise ValueError("qCIRIM does not support explicit DC component.")

        self.reconstruction_module = torch.nn.ModuleList([])

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module")
        if self.use_reconstruction_module:
            self.reconstruction_module_recurrent_filters = cfg_dict.get("reconstruction_module_recurrent_filters")
            self.reconstruction_module_time_steps = 8 * math.ceil(cfg_dict.get("reconstruction_module_time_steps") / 8)
            self.reconstruction_module_no_dc = cfg_dict.get("reconstruction_module_no_dc")
            self.reconstruction_module_num_cascades = cfg_dict.get("reconstruction_module_num_cascades")

            for _ in range(self.reconstruction_module_num_cascades):
                self.reconstruction_module.append(
                    RIMBlock(
                        recurrent_layer=cfg_dict.get("reconstruction_module_recurrent_layer"),
                        conv_filters=cfg_dict.get("reconstruction_module_conv_filters"),
                        conv_kernels=cfg_dict.get("reconstruction_module_conv_kernels"),
                        conv_dilations=cfg_dict.get("reconstruction_module_conv_dilations"),
                        conv_bias=cfg_dict.get("reconstruction_module_conv_bias"),
                        recurrent_filters=self.reconstruction_module_recurrent_filters,
                        recurrent_kernels=cfg_dict.get("reconstruction_module_recurrent_kernels"),
                        recurrent_dilations=cfg_dict.get("reconstruction_module_recurrent_dilations"),
                        recurrent_bias=cfg_dict.get("reconstruction_module_recurrent_bias"),
                        depth=cfg_dict.get("reconstruction_module_depth"),
                        time_steps=self.reconstruction_module_time_steps,
                        conv_dim=cfg_dict.get("reconstruction_module_conv_dim"),
                        fft_centered=self.fft_centered,
                        fft_normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                        coil_dim=self.coil_dim - 1,
                        dimensionality=cfg_dict.get("reconstruction_module_dimensionality"),
                        coil_combination_method=self.coil_combination_method,
                    )
                )

            # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
            self.reconstruction_module_keep_prediction = cfg_dict.get("reconstruction_module_keep_prediction")

            # initialize weights if not using pretrained cirim
            if not cfg_dict.get("pretrained", False):
                std_init_range = 1 / self.reconstruction_module_recurrent_filters[0] ** 0.5
                self.reconstruction_module.apply(lambda module: rnn_weights_init(module, std_init_range))

            self.dc_weight = torch.nn.Parameter(torch.ones(1))
            self.reconstruction_module_accumulate_predictions = cfg_dict.get(
                "reconstruction_module_accumulate_predictions"
            )

        self.quantitative_maps_scaling_factor = cfg_dict.get("quantitative_maps_scaling_factor")

        quantitative_module_num_cascades = cfg_dict.get("quantitative_module_num_cascades")
        self.quantitative_module = torch.nn.ModuleList(
            [
                qRIMBlock(
                    recurrent_layer=cfg_dict.get("quantitative_module_recurrent_layer"),
                    conv_filters=cfg_dict.get("quantitative_module_conv_filters"),
                    conv_kernels=cfg_dict.get("quantitative_module_conv_kernels"),
                    conv_dilations=cfg_dict.get("quantitative_module_conv_dilations"),
                    conv_bias=cfg_dict.get("quantitative_module_conv_bias"),
                    recurrent_filters=cfg_dict.get("quantitative_module_recurrent_filters"),
                    recurrent_kernels=cfg_dict.get("quantitative_module_recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("quantitative_module_recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("quantitative_module_recurrent_bias"),
                    depth=cfg_dict.get("quantitative_module_depth"),
                    time_steps=cfg_dict.get("quantitative_module_time_steps"),
                    conv_dim=cfg_dict.get("quantitative_module_conv_dim"),
                    linear_forward_model=SignalForwardModel(
                        sequence=cfg_dict.get("quantitative_module_signal_forward_model_sequence")
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    coil_combination_method=self.coil_combination_method,
                    dimensionality=quantitative_module_dimensionality,
                )
                for _ in range(quantitative_module_num_cascades)
            ]
        )
        self.quantitative_maps_regularization_factors = cfg_dict.get(
            "quantitative_maps_regularization_factors", [150.0, 150.0, 1000.0, 150.0]
        )

        self.accumulate_predictions = cfg_dict.get("quantitative_module_accumulate_predictions")

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        R2star_map_init: torch.Tensor,
        S0_map_init: torch.Tensor,
        B0_map_init: torch.Tensor,
        phi_map_init: torch.Tensor,
        TEs: List,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        initial_prediction: torch.Tensor,
        anatomy_mask: torch.Tensor,
        sampling_mask: torch.Tensor,
        sigma: float = 1.0,
    ) -> Union[List[List[Tensor]], List[Tensor]]:
        """Forward pass of :class:`qCIRIM`.

        Parameters
        ----------
        R2star_map_init : torch.Tensor
            Initial R2* map of shape [batch_size, n_x, n_y].
        S0_map_init : torch.Tensor
            Initial S0 map of shape [batch_size, n_x, n_y].
        B0_map_init : torch.Tensor
            Initial B0 map of shape [batch_size, n_x, n_y].
        phi_map_init : torch.Tensor
            Initial phase map of shape [batch_size, n_x, n_y].
        TEs : List
            List of echo times.
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2].
        initial_prediction : torch.Tensor
            Initial prediction of shape [batch_size, n_x, n_y, 2].
        anatomy_mask : torch.Tensor
            Brain mask of shape [batch_size, 1, n_x, n_y, 1].
        sampling_mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        sigma : float
            Standard deviation of the noise. Default is ``1.0``.

        Returns
        -------
        List of list of torch.Tensor or torch.Tensor
             If self.accumulate_loss is True, returns a list of all intermediate predictions.
             If False, returns the final estimate.
        """
        if self.use_reconstruction_module:
            cascades_echoes_reconstruction_predictions = []
            sigma = 1.0
            for echo in range(y.shape[1]):
                reconstruction_prediction = y[:, echo, ...].clone()
                initial_reconstruction_prediction_echo = None
                hx = None
                cascades_reconstruction_predictions = []
                for i, cascade in enumerate(self.reconstruction_module):
                    # Forward pass through the cascades
                    reconstruction_prediction, hx = cascade(
                        reconstruction_prediction,
                        y[:, echo, ...],
                        sensitivity_maps,
                        sampling_mask[:, 0, ...],
                        initial_reconstruction_prediction_echo if i == 0 else reconstruction_prediction,
                        hx,
                        sigma,
                        keep_prediction=False if i == 0 else self.reconstruction_module_keep_prediction,
                    )
                    cascades_reconstruction_predictions.append(
                        [torch.view_as_complex(x) for x in reconstruction_prediction]
                    )
                    reconstruction_prediction = reconstruction_prediction[-1]
                cascades_echoes_reconstruction_predictions.append(cascades_reconstruction_predictions)

            # cascades_echoes_reconstruction_predictions is of length n_echoes, len(self.reconstruction_module),
            # self.reconstruction_module_time_steps. We want to concatenate the echoes for each cascade and time step.
            reconstruction_prediction = []
            for cascade in range(len(cascades_echoes_reconstruction_predictions[0])):
                reconstruction_prediction_cascade = []
                for time_step in range(len(cascades_echoes_reconstruction_predictions[0][cascade])):
                    reconstruction_prediction_time_step = []
                    for echo in range(  # pylint: disable=consider-using-enumerate
                        len(cascades_echoes_reconstruction_predictions)
                    ):
                        reconstruction_prediction_time_step.append(
                            cascades_echoes_reconstruction_predictions[echo][cascade][time_step]
                        )
                    reconstruction_prediction_time_step = torch.stack(reconstruction_prediction_time_step, dim=1)
                    if reconstruction_prediction_time_step.shape[-1] != 2:  # type: ignore
                        reconstruction_prediction_time_step = torch.view_as_real(reconstruction_prediction_time_step)
                    reconstruction_prediction_cascade.append(reconstruction_prediction_time_step)
                reconstruction_prediction.append(reconstruction_prediction_cascade)

            final_reconstruction_prediction = reconstruction_prediction[-1][-1]
            if not self.reconstruction_module_accumulate_predictions:
                reconstruction_prediction = final_reconstruction_prediction

            y = fft2(
                coil_combination_method(
                    final_reconstruction_prediction.unsqueeze(self.coil_dim),
                    sensitivity_maps.unsqueeze(self.coil_dim - 1),
                    method=self.coil_combination_method,
                    dim=self.coil_dim - 1,
                ),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )

            R2star_maps_init = []
            S0_maps_init = []
            B0_maps_init = []
            phi_maps_init = []
            for batch_idx in range(final_reconstruction_prediction.shape[0]):
                R2star_map_init, S0_map_init, B0_map_init, phi_map_init = R2star_B0_S0_phi_mapping(
                    final_reconstruction_prediction[batch_idx],
                    TEs,
                    anatomy_mask,
                    scaling_factor=self.quantitative_maps_scaling_factor,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                R2star_maps_init.append(R2star_map_init.squeeze(0))
                S0_maps_init.append(S0_map_init.squeeze(0))
                B0_maps_init.append(B0_map_init.squeeze(0))
                phi_maps_init.append(phi_map_init.squeeze(0))
            R2star_map_init = torch.stack(R2star_maps_init, dim=0).to(y)
            S0_map_init = torch.stack(S0_maps_init, dim=0).to(y)
            B0_map_init = torch.stack(B0_maps_init, dim=0).to(y)
            phi_map_init = torch.stack(phi_maps_init, dim=0).to(y)
        else:
            reconstruction_prediction = initial_prediction.clone()

        R2star_map_init = R2star_map_init / self.quantitative_maps_regularization_factors[0]
        S0_map_init = S0_map_init / self.quantitative_maps_regularization_factors[1]
        B0_map_init = B0_map_init / self.quantitative_maps_regularization_factors[2]
        phi_map_init = phi_map_init / self.quantitative_maps_regularization_factors[3]

        qmaps_prediction = torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], dim=1)
        hx = None
        cascades_R2star_maps_prediction = []
        cascades_S0_maps_prediction = []
        cascades_B0_maps_prediction = []
        cascades_phi_maps_prediction = []
        for i, cascade in enumerate(self.quantitative_module):
            # Forward pass through the cascades
            qmaps_prediction, hx = cascade(qmaps_prediction, y, sensitivity_maps, sampling_mask, TEs, hx)
            # Keep the intermediate predictions
            for qmaps_pred in qmaps_prediction:
                cascades_R2star_maps_prediction.append(
                    qmaps_pred[:, 0, ...] * self.quantitative_maps_regularization_factors[0]
                )
                cascades_S0_maps_prediction.append(
                    qmaps_pred[:, 1, ...] * self.quantitative_maps_regularization_factors[1]
                )
                cascades_B0_maps_prediction.append(
                    qmaps_pred[:, 2, ...] * self.quantitative_maps_regularization_factors[2]
                )
                cascades_phi_maps_prediction.append(
                    qmaps_pred[:, 3, ...] * self.quantitative_maps_regularization_factors[3]
                )
            # Keep the final prediction for the next cascade
            qmaps_prediction = qmaps_prediction[-1]

        if not self.accumulate_predictions:
            reconstruction_prediction = (
                reconstruction_prediction[-1][-1] if self.use_reconstruction_module else torch.empty([])
            )
            cascades_R2star_maps_prediction = cascades_R2star_maps_prediction[-1][-1]
            cascades_S0_maps_prediction = cascades_S0_maps_prediction[-1][-1]
            cascades_B0_maps_prediction = cascades_B0_maps_prediction[-1][-1]
            cascades_phi_maps_prediction = cascades_phi_maps_prediction[-1][-1]

        return [
            reconstruction_prediction,
            cascades_R2star_maps_prediction,
            cascades_S0_maps_prediction,
            cascades_B0_maps_prediction,
            cascades_phi_maps_prediction,
        ]

    def process_quantitative_loss(
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        anatomy_mask: torch.Tensor,
        quantitative_map: str,
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """Processes the quantitative loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        anatomy_mask : torch.Tensor
            Mask of specified anatomy, e.g. brain. Shape [n_x, n_y].
        quantitative_map : str
            Type of quantitative map to regularize the loss. Must be one of {"R2star", "S0", "B0", "phi"}.
        loss_func : torch.nn.Module
            Loss function. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        target = torch.abs(self.__abs_output__(target / torch.max(torch.abs(target))))
        anatomy_mask = torch.abs(anatomy_mask).to(target)

        def compute_quantitative_loss(t, p, m):
            p = torch.abs(self.__abs_output__(p / torch.max(torch.abs(p))))

            if "ssim" in str(loss_func).lower():
                return (
                    loss_func(
                        t * m,
                        p * m,
                        data_range=torch.tensor([max(torch.max(t * m).item(), torch.max(p * m).item())])
                        .unsqueeze(dim=0)
                        .to(t),
                    )
                    * self.quantitative_parameters_regularization_factors[quantitative_map]
                )
            return loss_func(t * m, p * m) / self.quantitative_parameters_regularization_factors[quantitative_map]

        if self.accumulate_predictions:
            cascades_loss = []
            for cascade_pred in prediction:
                time_steps_loss = [
                    compute_quantitative_loss(target, time_step_pred, anatomy_mask) for time_step_pred in cascade_pred
                ]
                cascades_loss.append(torch.sum(torch.stack(time_steps_loss, dim=0)) / len(prediction))
            loss = sum(cascades_loss) / len(self.quantitative_module)
        else:
            loss = compute_quantitative_loss(target, prediction, anatomy_mask)
        return loss

    def process_reconstruction_loss(  # noqa: MC0001
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        attrs: Dict,
        r: int,
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """Processes the reconstruction loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
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
        if self.unnormalize_loss_inputs:
            for cascade in range(len(prediction)):  # pylint: disable=consider-using-enumerate
                for time_steps in range(len(prediction[cascade])):
                    target, prediction[cascade][time_steps], sensitivity_maps = self.__unnormalize_for_loss_or_log__(
                        target, prediction[cascade][time_steps], sensitivity_maps, attrs, r
                    )

        # If kspace reconstruction loss is used, the target needs to be transformed in k-space.
        if self.kspace_quantitative_loss:
            # If inputs are complex, then they need to be viewed as real.
            if target.shape[-1] != 2 and torch.is_complex(target):
                target = torch.view_as_real(target)

            # Transform to k-space.
            target = fft2(target, self.fft_centered, self.fft_normalization, self.spatial_dims)

            # Ensure loss inputs are both viewed in the same way.
            target = self.__abs_output__(target / torch.max(torch.abs(target)))

            for cascade in range(len(prediction)):  # pylint: disable=consider-using-enumerate
                for time_steps in range(len(prediction[cascade])):
                    if prediction[cascade][time_steps].shape[-1] != 2 and torch.is_complex(
                        prediction[cascade][time_steps]
                    ):
                        prediction[cascade][time_steps] = torch.view_as_real(prediction[cascade][time_steps])
                    prediction[cascade][time_steps] = fft2(
                        prediction[cascade][time_steps],
                        self.fft_centered,
                        self.fft_normalization,
                        self.spatial_dims,
                    )
                    prediction[cascade][time_steps] = self.__abs_output__(
                        prediction[cascade][time_steps] / torch.max(torch.abs(prediction[cascade][time_steps]))
                    )

        elif not self.unnormalize_loss_inputs:
            # Ensure loss inputs are both viewed in the same way.
            target = self.__abs_output__(target / torch.max(torch.abs(target)))
            for cascade in range(len(prediction)):  # pylint: disable=consider-using-enumerate
                for time_steps in range(len(prediction[cascade])):
                    prediction[cascade][time_steps] = self.__abs_output__(
                        prediction[cascade][time_steps] / torch.max(torch.abs(prediction[cascade][time_steps]))
                    )

        def compute_reconstruction_loss(t, p):
            p = torch.abs(p / torch.max(torch.abs(p)))
            t = torch.abs(t / torch.max(torch.abs(t)))
            return torch.mean(torch.tensor([loss_func(t[:, echo], p[:, echo]) for echo in range(t.shape[1])]))

        if self.reconstruction_module_accumulate_predictions:
            cascades_loss = []
            for cascade_prediction in prediction:
                cascade_time_steps_loss = [
                    compute_reconstruction_loss(target, cascade_prediction_time_step_prediction).mean()
                    for cascade_prediction_time_step_prediction in cascade_prediction
                ]
                cascade_loss = [
                    x
                    * torch.logspace(-1, 0, steps=self.reconstruction_module_time_steps).to(cascade_time_steps_loss[0])
                    for x in cascade_time_steps_loss
                ]
                cascades_loss.append(sum(sum(cascade_loss) / len(self.reconstruction_module)))
            loss = sum(cascades_loss) / len(prediction)
        else:
            loss = compute_reconstruction_loss(target, prediction)
        return loss
