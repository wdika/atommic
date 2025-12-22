# coding=utf-8
__author__ = "Dimitris Karkalousos"

import os
import warnings
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader

# do not import BaseMRIModel, BaseSensitivityModel, and DistributedMetricSum directly to avoid circular imports
import atommic.collections.common as atommic_common
from atommic.collections.common.data.subsample import create_masker
from atommic.collections.common.losses import VALID_RECONSTRUCTION_LOSSES, VALID_SEGMENTATION_LOSSES
from atommic.collections.common.losses.aggregator import AggregatorLoss
from atommic.collections.common.losses.wasserstein import SinkhornDistance
from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import (
    check_stacked_complex,
    coil_combination_method,
    complex_abs,
    complex_abs_sq,
    expand_op,
    is_none,
    unnormalize,
)
from atommic.collections.multitask.rs.data import mrirs_loader
from atommic.collections.multitask.rs.parts.transforms import RSMRIDataTransforms
from atommic.collections.reconstruction.losses.na import NoiseAwareLoss
from atommic.collections.reconstruction.losses.ssim import SSIMLoss
from atommic.collections.reconstruction.metrics import mse, nmse, psnr, ssim
from atommic.collections.segmentation.losses.cross_entropy import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
from atommic.collections.segmentation.losses.dice import Dice, GeneralisedDice
from atommic.collections.segmentation.losses.focal import FocalLoss
from atommic.collections.segmentation.losses.utils import one_hot

__all__ = ["BaseMRIReconstructionSegmentationModel"]


class BaseMRIReconstructionSegmentationModel(atommic_common.nn.base.BaseMRIModel, ABC):  # type: ignore
    """Base class of all (multitask) MRI reconstruction and MRI segmentation models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):  # noqa: MC0001
        """Inits :class:`BaseMRIReconstructionSegmentationModel`.

        Parameters
        ----------
        cfg : DictConfig
            The configuration file.
        trainer : Trainer
            The PyTorch Lightning trainer. Default is ``None``.
        """
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Initialize the Fast-Fourier Transform parameters.
        self.fft_centered = cfg_dict.get("fft_centered", False)
        self.fft_normalization = cfg_dict.get("fft_normalization", "backward")
        self.spatial_dims = cfg_dict.get("spatial_dims", None)
        self.coil_dim = cfg_dict.get("coil_dim", 1)

        # Initialize the dimensionality of the data. It can be 2D or 2.5D -> meaning 2D with > 1 slices or 3D.
        self.dimensionality = cfg_dict.get("dimensionality", 2)
        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

        # Set the output channels of the segmentation module.
        self.segmentation_module_output_channels = cfg_dict.get("segmentation_module_output_channels", 2)

        # Initialize the coil combination method. It can be either "SENSE" or "RSS" (root-sum-of-squares) or
        # "RSS-complex" (root-sum-of-squares of the complex-valued data).
        self.coil_combination_method = cfg_dict.get("coil_combination_method", "SENSE")

        # Refers to Self-Supervised Data Undersampling (SSDU). If True, then the model is trained with only
        # undersampled data.
        self.ssdu = cfg_dict.get("ssdu", False)

        # Refers to Noise-to-Recon. If True, then the model can either be trained with only undersampled data or with
        # both undersampled and (a percentage of) fully-sampled data.
        self.n2r = cfg_dict.get("n2r", False)

        # Initialize the sensitivity network if cfg_dict.get("estimate_coil_sensitivity_maps_with_nn") is True.
        self.estimate_coil_sensitivity_maps_with_nn = cfg_dict.get("estimate_coil_sensitivity_maps_with_nn", False)

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module")
        if self.use_reconstruction_module:
            # Initialize loss related parameters.
            self.kspace_reconstruction_loss = cfg_dict.get("kspace_reconstruction_loss", False)
            self.n2r_loss_weight = cfg_dict.get("n2r_loss_weight", 1.0) if self.n2r else 1.0
            self.reconstruction_losses = {}
            reconstruction_loss = cfg_dict.get("reconstruction_loss")
            reconstruction_losses_ = {}
            if reconstruction_loss is not None:
                for k, v in reconstruction_loss.items():
                    if k not in VALID_RECONSTRUCTION_LOSSES:
                        raise ValueError(
                            f"Reconstruction loss {k} is not supported. Please choose one of the following: "
                            f"{VALID_RECONSTRUCTION_LOSSES}."
                        )
                    if v is None or v == 0.0:
                        warnings.warn(
                            f"The weight of reconstruction loss {k} is set to 0.0. This loss will not be used."
                        )
                    else:
                        reconstruction_losses_[k] = v
            else:
                # Default reconstruction loss is L1.
                reconstruction_losses_["l1"] = 1.0
            if sum(reconstruction_losses_.values()) != 1.0:
                warnings.warn("Sum of reconstruction losses weights is not 1.0. Adjusting weights to sum up to 1.0.")
                total_weight = sum(reconstruction_losses_.values())
                reconstruction_losses_ = {k: v / total_weight for k, v in reconstruction_losses_.items()}
            for name in VALID_RECONSTRUCTION_LOSSES:
                if name in reconstruction_losses_:
                    if name == "ssim":
                        if self.ssdu:
                            raise ValueError("SSIM loss is not supported for SSDU.")
                        self.reconstruction_losses[name] = SSIMLoss()
                    elif name == "mse":
                        self.reconstruction_losses[name] = MSELoss()
                    elif name == "wasserstein":
                        self.reconstruction_losses[name] = SinkhornDistance()
                    elif name == "noise_aware":
                        self.reconstruction_losses[name] = NoiseAwareLoss()
                    elif name == "l1":
                        self.reconstruction_losses[name] = L1Loss()
            # replace losses names by 'loss_1', 'loss_2', etc. to properly iterate in the aggregator loss
            self.reconstruction_losses = {f"loss_{i+1}": v for i, v in enumerate(self.reconstruction_losses.values())}
            self.total_reconstruction_losses = len(self.reconstruction_losses)
            self.total_reconstruction_loss_weight = cfg_dict.get("total_reconstruction_loss_weight", 1.0)

            # Initialize the reconstruction metrics.
            self.reconstruction_loss_weight = cfg_dict.get("reconstruction_loss_weight", 1.0)

            # Set normalization parameters for logging
            self.unnormalize_loss_inputs = cfg_dict.get("unnormalize_loss_inputs", False)
            self.unnormalize_log_outputs = cfg_dict.get("unnormalize_log_outputs", False)
            self.normalization_type = cfg_dict.get("normalization_type", "max")

            # Refers to cascading or iterative reconstruction methods.
            self.accumulate_predictions = cfg_dict.get("accumulate_predictions", False)

            # Refers to the type of the complex-valued data. It can be either "stacked" or "complex_abs" or
            # "complex_sqrt_abs".
            self.complex_valued_type = cfg_dict.get("complex_valued_type", "stacked")

        # Set normalization parameters for logging
        self.unnormalize_loss_inputs = cfg_dict.get("unnormalize_loss_inputs", False)
        self.unnormalize_log_outputs = cfg_dict.get("unnormalize_log_outputs", False)
        self.normalization_type = cfg_dict.get("normalization_type", "max")
        self.normalize_segmentation_output = cfg_dict.get("normalize_segmentation_output", True)

        # Whether to log multiple modalities, e.g. T1, T2, and FLAIR will be stacked and logged.
        self.log_multiple_modalities = cfg_dict.get("log_multiple_modalities", False)

        # Set segmentation approach defaults
        self.segmentation_mode = cfg_dict.get("segmentation_mode", "multilabel")
        self.segmentation_activation = cfg_dict.get("segmentation_activation", "sigmoid")
        self.segmentation_classes_thresholds = cfg_dict.get("segmentation_classes_thresholds", None)
        self.segmentation_output_mode = cfg_dict.get("segmentation_output_mode", "binary")

        # Initialize loss related parameters.
        self.segmentation_losses = {}
        segmentation_loss = cfg_dict.get("segmentation_loss")
        segmentation_losses_ = {}
        if segmentation_loss is not None:
            for k, v in segmentation_loss.items():
                if k not in VALID_SEGMENTATION_LOSSES:
                    raise ValueError(
                        f"Segmentation loss {k} is not supported. Please choose one of the following: "
                        f"{VALID_SEGMENTATION_LOSSES}."
                    )
                if v is None or v == 0.0:
                    warnings.warn(f"The weight of segmentation loss {k} is set to 0.0. This loss will not be used.")
                else:
                    segmentation_losses_[k] = v
        else:
            # Default segmentation loss is Dice.
            segmentation_losses_["dice"] = 1.0
        if sum(segmentation_losses_.values()) != 1.0:
            warnings.warn("Sum of segmentation losses weights is not 1.0. Adjusting weights to sum up to 1.0.")
            total_weight = sum(segmentation_losses_.values())
            segmentation_losses_ = {k: v / total_weight for k, v in segmentation_losses_.items()}
        for name in VALID_SEGMENTATION_LOSSES:
            if name in segmentation_losses_:
                if name == "categorical_cross_entropy":
                    if self.segmentation_mode == "multilabel":
                        raise ValueError(
                            "Categorical cross-entropy loss is not supported for multilabel segmentation. "
                            "Please use binary cross-entropy."
                        )
                    self.segmentation_losses[name] = CategoricalCrossEntropyLoss(
                        include_background=cfg_dict.get("cross_entropy_loss_include_background", True),
                        num_samples=cfg_dict.get("cross_entropy_loss_num_samples", 50),
                        ignore_index=cfg_dict.get("cross_entropy_loss_ignore_index", -100),
                        reduction=cfg_dict.get("cross_entropy_loss_reduction", "mean"),
                        label_smoothing=cfg_dict.get("cross_entropy_loss_label_smoothing", 0.0),
                        weight=cfg_dict.get("cross_entropy_loss_classes_weight", None),
                        to_onehot_y=cfg_dict.get("cross_entropy_loss_to_onehot_y", False),
                        num_segmentation_classes=self.segmentation_module_output_channels,
                    )
                elif name == "binary_cross_entropy":
                    if self.segmentation_mode == "multiclass":
                        raise ValueError(
                            "Binary cross-entropy loss is not supported for multiclass segmentation. "
                            "Please use categorical cross-entropy."
                        )
                    self.segmentation_losses[name] = BinaryCrossEntropyLoss(
                        include_background=cfg_dict.get("cross_entropy_loss_include_background", False),
                        num_samples=cfg_dict.get("cross_entropy_loss_num_samples", 50),
                        reduction=cfg_dict.get("cross_entropy_loss_reduction", "mean"),
                        weight=cfg_dict.get("cross_entropy_loss_classes_weight", None),
                        to_onehot_y=cfg_dict.get("cross_entropy_loss_to_onehot_y", False),
                        num_segmentation_classes=self.segmentation_module_output_channels,
                    )
                elif name == "focal_loss":
                    use_softmax = cfg_dict.get("focal_loss_use_softmax", False)
                    if self.segmentation_mode == "multiclass" and not use_softmax:
                        raise ValueError(
                            "Focal loss without softmax operations is not supported for multiclass segmentation. "
                            "Change focal_loss_use_softmax = true"
                        )
                    self.segmentation_losses[name] = FocalLoss(
                        include_background=cfg_dict.get("focal_loss_include_background", False),
                        reduction=cfg_dict.get("focal_loss_reduction", "mean"),
                        weight=cfg_dict.get("focal_loss_classes_weight", None),
                        alpha=cfg_dict.get("focal_loss_alpha", None),
                        gamma=cfg_dict.get("focal_loss_gamma", 2.0),
                        use_softmax=use_softmax,
                        to_onehot_y=cfg_dict.get("focal_loss_to_onehot_y", False),
                        num_segmentation_classes=self.segmentation_module_output_channels,
                    )
                elif name == "dice":
                    use_softmax = cfg_dict.get("dice_loss_softmax", False)
                    if self.segmentation_mode == "multiclass" and not use_softmax:
                        warnings.warn(
                            "Dice loss without softmax operation is not advised for multiclass segmentation. "
                            "Change dice_loss_softmax = true"
                        )
                    self.segmentation_losses[name] = Dice(
                        include_background=cfg_dict.get("dice_loss_include_background", False),
                        to_onehot_y=cfg_dict.get("dice_loss_to_onehot_y", False),
                        sigmoid=cfg_dict.get("dice_loss_sigmoid", False),
                        softmax=use_softmax,
                        other_act=cfg_dict.get("dice_loss_other_act", None),
                        squared_pred=cfg_dict.get("dice_loss_squared_pred", False),
                        jaccard=cfg_dict.get("dice_loss_jaccard", False),
                        flatten=cfg_dict.get("dice_loss_flatten", False),
                        reduction=cfg_dict.get("dice_loss_reduction", "mean"),
                        smooth_nr=cfg_dict.get("dice_loss_smooth_nr", 1e-5),
                        smooth_dr=cfg_dict.get("dice_loss_smooth_dr", 1e-5),
                        batch=cfg_dict.get("dice_loss_batch", True),
                        num_segmentation_classes=self.segmentation_module_output_channels,
                    )
                elif name == "generalized_dice":
                    use_softmax = cfg_dict.get("dice_loss_softmax", False)
                    if self.segmentation_mode == "multiclass" and not use_softmax:
                        warnings.warn(
                            "Generalised dice loss without softmax operation is not advised for multiclass "
                            "segmentation. Change dice_loss_softmax = true"
                        )
                    self.segmentation_losses[name] = GeneralisedDice(
                        include_background=cfg_dict.get("dice_loss_include_background", False),
                        to_onehot_y=cfg_dict.get("dice_loss_to_onehot_y", False),
                        sigmoid=cfg_dict.get("dice_loss_sigmoid", True),
                        softmax=use_softmax,
                        other_act=cfg_dict.get("dice_loss_other_act", None),
                        reduction=cfg_dict.get("dice_loss_reduction", "mean"),
                        w_type=cfg_dict.get("dice_loss_w_type", "square"),
                        smooth_nr=cfg_dict.get("dice_loss_smooth_nr", 1e-5),
                        smooth_dr=cfg_dict.get("dice_loss_smooth_dr", 1e-5),
                        batch=cfg_dict.get("dice_loss_batch", True),
                        num_segmentation_classes=self.segmentation_module_output_channels,
                    )

        self.segmentation_losses = {f"loss_{i+1}": v for i, v in enumerate(self.segmentation_losses.values())}
        self.total_segmentation_losses = len(self.segmentation_losses)
        self.total_segmentation_loss_weight = cfg_dict.get("total_segmentation_loss_weight", 1.0)

        # Set the metrics
        cross_entropy_metric_include_background = cfg_dict.get("cross_entropy_metric_include_background", False)
        cross_entropy_metric_num_samples = cfg_dict.get("cross_entropy_metric_num_samples", 50)
        cross_entropy_metric_ignore_index = cfg_dict.get("cross_entropy_metric_ignore_index", -100)
        cross_entropy_metric_reduction = cfg_dict.get("cross_entropy_metric_reduction", "mean")
        cross_entropy_metric_label_smoothing = cfg_dict.get("cross_entropy_metric_label_smoothing", 0.0)
        cross_entropy_metric_classes_weight = cfg_dict.get("cross_entropy_metric_classes_weight", None)
        cross_entropy_metric_to_onehot_y = cfg_dict.get("cross_entropy_metric_to_onehot_y", False)

        dice_metric_include_background = cfg_dict.get("dice_metric_include_background", False)
        dice_metric_to_onehot_y = cfg_dict.get("dice_metric_to_onehot_y", False)
        dice_metric_sigmoid = cfg_dict.get("dice_metric_sigmoid", True)
        dice_metric_softmax = cfg_dict.get("dice_metric_softmax", False)
        dice_metric_other_act = cfg_dict.get("dice_metric_other_act", None)
        dice_metric_squared_pred = cfg_dict.get("dice_metric_squared_pred", False)
        dice_metric_jaccard = cfg_dict.get("dice_metric_jaccard", False)
        dice_metric_flatten = cfg_dict.get("dice_metric_flatten", False)
        dice_metric_reduction = cfg_dict.get("dice_metric_reduction", "mean")
        dice_metric_smooth_nr = cfg_dict.get("dice_metric_smooth_nr", 1e-5)
        dice_metric_smooth_dr = cfg_dict.get("dice_metric_smooth_dr", 1e-5)
        dice_metric_batch = cfg_dict.get("dice_metric_batch", True)

        self.metric_computation_mode = cfg_dict.get("metric_computation_mode", "per_slice")
        if self.metric_computation_mode not in ["per_slice", "per_volume"]:
            raise ValueError(
                f"metric_computation_mode = {self.metric_computation_mode} is not supported. Please select "
                "'per_slice' or 'per_volume'."
            )

        # Initialize the module
        super().__init__(cfg=cfg, trainer=trainer)

        if self.estimate_coil_sensitivity_maps_with_nn:
            self.coil_sensitivity_maps_nn = atommic_common.nn.base.BaseSensitivityModel(  # type: ignore
                cfg_dict.get("coil_sensitivity_maps_nn_chans", 8),
                cfg_dict.get("coil_sensitivity_maps_nn_pools", 4),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
                coil_dim=self.coil_dim,
                mask_type=cfg_dict.get("coil_sensitivity_maps_nn_mask_type", "2D"),
                normalize=cfg_dict.get("coil_sensitivity_maps_nn_normalize", True),
                mask_center=cfg_dict.get("coil_sensitivity_maps_nn_mask_center", True),
            )

        if self.use_reconstruction_module:
            # Set aggregation loss
            self.total_reconstruction_loss = AggregatorLoss(
                num_inputs=self.total_reconstruction_losses, weights=list(reconstruction_losses_.values())
            )

            self.MSE = atommic_common.nn.base.DistributedMetricSum()  # type: ignore
            self.NMSE = atommic_common.nn.base.DistributedMetricSum()  # type: ignore
            self.SSIM = atommic_common.nn.base.DistributedMetricSum()  # type: ignore
            self.PSNR = atommic_common.nn.base.DistributedMetricSum()  # type: ignore
            self.TotExamples = atommic_common.nn.base.DistributedMetricSum()  # type: ignore

            self.mse_vals_reconstruction: Dict = defaultdict(dict)
            self.nmse_vals_reconstruction: Dict = defaultdict(dict)
            self.ssim_vals_reconstruction: Dict = defaultdict(dict)
            self.psnr_vals_reconstruction: Dict = defaultdict(dict)

        if self.segmentation_mode == "multilabel":
            self.cross_entropy_metric = BinaryCrossEntropyLoss(
                include_background=cross_entropy_metric_include_background,
                num_samples=cross_entropy_metric_num_samples,
                weight=cross_entropy_metric_classes_weight,
                reduction=cross_entropy_metric_reduction,
                to_onehot_y=cross_entropy_metric_to_onehot_y,
                num_segmentation_classes=self.segmentation_module_output_channels,
            )
        else:
            self.cross_entropy_metric = CategoricalCrossEntropyLoss(
                include_background=cross_entropy_metric_include_background,
                num_samples=cross_entropy_metric_num_samples,
                ignore_index=cross_entropy_metric_ignore_index,
                reduction=cross_entropy_metric_reduction,
                label_smoothing=cross_entropy_metric_label_smoothing,
                weight=cross_entropy_metric_classes_weight,
                to_onehot_y=cross_entropy_metric_to_onehot_y,
                num_segmentation_classes=self.segmentation_module_output_channels,
            )

        self.dice_metric = Dice(
            include_background=dice_metric_include_background,
            to_onehot_y=dice_metric_to_onehot_y,
            sigmoid=dice_metric_sigmoid,
            softmax=dice_metric_softmax,
            other_act=dice_metric_other_act,
            squared_pred=dice_metric_squared_pred,
            jaccard=dice_metric_jaccard,
            flatten=dice_metric_flatten,
            reduction=dice_metric_reduction,
            smooth_nr=dice_metric_smooth_nr,
            smooth_dr=dice_metric_smooth_dr,
            batch=dice_metric_batch,
            num_segmentation_classes=self.segmentation_module_output_channels,
        )

        # Set aggregation loss
        self.total_segmentation_loss = AggregatorLoss(
            num_inputs=self.total_segmentation_losses,
            weights=list(segmentation_losses_.values()),
        )

        # Set distributed metrics
        self.CROSS_ENTROPY = atommic_common.nn.base.DistributedMetricSum()  # type: ignore
        self.DICE = atommic_common.nn.base.DistributedMetricSum()  # type: ignore
        self.cross_entropy_vals: Dict = defaultdict(dict)
        self.dice_vals: Dict = defaultdict(dict)
        self.TotExamples = atommic_common.nn.base.DistributedMetricSum()  # type: ignore

    def __abs_output__(self, x: torch.Tensor) -> torch.Tensor:
        """Converts the input to absolute value."""
        if x.shape[-1] == 2 or torch.is_complex(x):
            if torch.is_complex(x):
                x = torch.view_as_real(x)
            if self.complex_valued_type == "stacked":
                x = check_stacked_complex(x)
            elif self.complex_valued_type == "complex_abs":
                x = complex_abs(x)
            elif self.complex_valued_type == "complex_sqrt_abs":
                x = complex_abs_sq(x)
        return x

    def __unnormalize_for_loss_or_log__(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        sensitivity_maps: Union[torch.Tensor, None],
        attrs: Dict,
        r: int,
        batch_idx: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        """
        Unnormalizes the data for computing the loss or logging.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : torch.Tensor
            Prediction data of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor or None
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2] or None.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        batch_idx : int
            Batch index. Default is ``1``.

        Returns
        -------
        target : torch.Tensor
            Unnormalized target data.
        prediction : torch.Tensor
            Unnormalized prediction data.
        sensitivity_maps : torch.Tensor
            Unnormalized sensitivity maps.
        """
        if self.n2r and not attrs["n2r_supervised"][batch_idx]:
            target = unnormalize(
                target,
                {
                    "min": (
                        attrs["prediction_min"][batch_idx]
                        if "prediction_min" in attrs
                        else attrs[f"prediction_min_{r}"][batch_idx]
                    ),
                    "max": (
                        attrs["prediction_max"][batch_idx]
                        if "prediction_max" in attrs
                        else attrs[f"prediction_max_{r}"][batch_idx]
                    ),
                    "mean": (
                        attrs["prediction_mean"][batch_idx]
                        if "prediction_mean" in attrs
                        else attrs[f"prediction_mean_{r}"][batch_idx]
                    ),
                    "std": (
                        attrs["prediction_std"][batch_idx]
                        if "prediction_std" in attrs
                        else attrs[f"prediction_std_{r}"][batch_idx]
                    ),
                },
                self.normalization_type,
            )
            prediction = unnormalize(
                prediction,
                {
                    "min": (
                        attrs["noise_prediction_min"][batch_idx]
                        if "noise_prediction_min" in attrs
                        else attrs[f"noise_prediction_min_{r}"][batch_idx]
                    ),
                    "max": (
                        attrs["noise_prediction_max"][batch_idx]
                        if "noise_prediction_max" in attrs
                        else attrs[f"noise_prediction_max_{r}"][batch_idx]
                    ),
                    attrs["noise_prediction_mean"][batch_idx]
                    if "noise_prediction_mean" in attrs
                    else "mean": attrs[f"noise_prediction_mean_{r}"][batch_idx],
                    attrs["noise_prediction_std"][batch_idx]
                    if "noise_prediction_std" in attrs
                    else "std": attrs[f"noise_prediction_std_{r}"][batch_idx],
                },
                self.normalization_type,
            )
        else:
            target = unnormalize(
                target,
                {
                    "min": (
                        attrs["target_min"][batch_idx]
                        if "target_min" in attrs
                        else attrs[f"target_min_{r}"][batch_idx]
                    ),
                    "max": (
                        attrs["target_max"][batch_idx]
                        if "target_max" in attrs
                        else attrs[f"target_max_{r}"][batch_idx]
                    ),
                    "mean": (
                        attrs["target_mean"][batch_idx]
                        if "target_mean" in attrs
                        else attrs[f"target_mean_{r}"][batch_idx]
                    ),
                    "std": (
                        attrs["target_std"][batch_idx]
                        if "target_std" in attrs
                        else attrs[f"target_std_{r}"][batch_idx]
                    ),
                },
                self.normalization_type,
            )
            prediction = unnormalize(
                prediction,
                {
                    "min": (
                        attrs["prediction_min"][batch_idx]
                        if "prediction_min" in attrs
                        else attrs[f"prediction_min_{r}"][batch_idx]
                    ),
                    "max": (
                        attrs["prediction_max"][batch_idx]
                        if "prediction_max" in attrs
                        else attrs[f"prediction_max_{r}"][batch_idx]
                    ),
                    "mean": (
                        attrs["prediction_mean"][batch_idx]
                        if "prediction_mean" in attrs
                        else attrs[f"prediction_mean_{r}"][batch_idx]
                    ),
                    "std": (
                        attrs["prediction_std"][batch_idx]
                        if "prediction_std" in attrs
                        else attrs[f"prediction_std_{r}"][batch_idx]
                    ),
                },
                self.normalization_type,
            )

        if sensitivity_maps is not None:
            sensitivity_maps = unnormalize(
                sensitivity_maps,
                {
                    "min": attrs["sensitivity_maps_min"][batch_idx],
                    "max": attrs["sensitivity_maps_max"][batch_idx],
                    "mean": attrs["sensitivity_maps_mean"][batch_idx],
                    "std": attrs["sensitivity_maps_std"][batch_idx],
                },
                self.normalization_type,
            )

        return target, prediction, sensitivity_maps

    def process_reconstruction_loss(
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        attrs: Union[Dict, torch.Tensor],
        r: Union[int, torch.Tensor],
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
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        loss_func : torch.nn.Module
            Loss function. Default is ``torch.nn.L1Loss()``.

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
            target = self.__abs_output__(target / torch.max(torch.abs(target)))
        elif not self.unnormalize_loss_inputs:
            target = self.__abs_output__(target / torch.max(torch.abs(target)))

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
                p = self.__abs_output__(p / torch.max(torch.abs(p)))
            elif not self.unnormalize_loss_inputs:
                p = self.__abs_output__(p / torch.max(torch.abs(p)))

            if "ssim" in str(loss_func).lower():
                p = torch.abs(p / torch.max(torch.abs(p)))
                t = torch.abs(t / torch.max(torch.abs(t)))

                return loss_func(
                    t,
                    p,
                    data_range=torch.tensor([max(torch.max(t).item(), torch.max(p).item())]).unsqueeze(dim=0).to(t),
                )

            return loss_func(t, p)

        return compute_reconstruction_loss(target, prediction, sensitivity_maps)

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
            target, prediction, _ = self.__unnormalize_for_loss_or_log__(target, prediction, None, attrs, attrs["r"])
        losses = {}
        for name, loss_func in self.segmentation_losses.items():
            loss = loss_func(target, prediction)
            if isinstance(loss, tuple):
                # In case of the dice loss, the loss is a tuple of the form (dice, dice loss)
                loss = loss[1]
            losses[name] = loss
        return self.total_segmentation_loss(**losses) * self.total_segmentation_loss_weight

    def __compute_loss__(
        self,
        predictions_reconstruction: Union[list, torch.Tensor],
        predictions_reconstruction_n2r: Union[list, torch.Tensor],
        target_reconstruction: torch.Tensor,
        predictions_segmentation: Union[list, torch.Tensor],
        target_segmentation: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        ssdu_loss_mask: torch.Tensor,
        attrs: Dict,
        r: int,
    ) -> torch.Tensor:
        """Computes the reconstruction loss.

        Parameters
        ----------
        predictions_reconstruction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        predictions_reconstruction_n2r : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2], if Noise-to-Recon is used. Otherwise, None.
        target_reconstruction : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        predictions_segmentation : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, nr_classes, n_x, n_y].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        ssdu_loss_mask : torch.Tensor
            SSDU loss mask of shape [batch_size, 1, n_x, n_y, 1]. It will be used if self.ssdu is True, to enforce
            data consistency on the prediction.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        # TODO: check why this have to go
        # if self.consecutive_slices > 1:
        #     batch_size, slices = target_segmentation.shape[:2]
        #     target_segmentation = target_segmentation.reshape(batch_size * slices, *target_segmentation.shape[2:])

        segmentation_loss = self.process_segmentation_loss(target_segmentation, predictions_segmentation, attrs)

        if self.use_reconstruction_module:
            if predictions_reconstruction_n2r is not None and not attrs["n2r_supervised"]:
                # Noise-to-Recon with/without SSDU
                target = predictions_reconstruction
                predictions_reconstruction = predictions_reconstruction_n2r
                weight = self.n2r_loss_weight
            else:
                # Supervised learning or Noise-to-Recon with SSDU
                target = target_reconstruction
                weight = 1.0
            losses = {}
            for name, loss_func in self.reconstruction_losses.items():
                losses[name] = (
                    self.process_reconstruction_loss(
                        target,
                        predictions_reconstruction,
                        sensitivity_maps,
                        ssdu_loss_mask,
                        attrs,
                        r,
                        loss_func=loss_func,
                    )
                    * weight
                )
            reconstruction_loss = self.total_reconstruction_loss(**losses)
        else:
            reconstruction_loss = torch.tensor(0.0)

        loss = (
            self.total_segmentation_loss_weight * segmentation_loss
            + self.total_reconstruction_loss_weight * reconstruction_loss
        )

        if self.accumulate_predictions:
            loss = sum(list(loss))

        return loss

    def __compute_and_log_metrics_and_outputs__(  # noqa: MC0001
        self,
        predictions_reconstruction: Union[list, torch.Tensor],
        target_reconstruction: torch.Tensor,
        predictions_segmentation: Union[list, torch.Tensor],
        target_segmentation: torch.Tensor,
        attrs: Dict,
    ):
        """Computes the metrics and logs the outputs.

        Parameters
        ----------
        predictions_reconstruction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        target_reconstruction : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        predictions_segmentation : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, nr_classes, n_x, n_y].
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        """
        if isinstance(predictions_reconstruction, list):
            while isinstance(predictions_reconstruction, list):
                predictions_reconstruction = predictions_reconstruction[-1]

        if isinstance(predictions_segmentation, list):
            while isinstance(predictions_segmentation, list):
                predictions_segmentation = predictions_segmentation[-1]

        if self.consecutive_slices > 1:
            # reshape the target and prediction to [batch_size, self.consecutive_slices, nr_classes, n_x, n_y]
            batch_size = int(target_segmentation.shape[0] // self.consecutive_slices)
            target_segmentation = target_segmentation.reshape(
                batch_size, self.consecutive_slices, *target_segmentation.shape[1:]
            )
            # TODO: check why this have to go
            # target_reconstruction = target_reconstruction.reshape(
            #     batch_size, self.consecutive_slices, *target_reconstruction.shape[2:]
            # )
            predictions_segmentation = predictions_segmentation.reshape(
                batch_size, self.consecutive_slices, *predictions_segmentation.shape[2:]
            )
            # TODO: check why this have to go
            # predictions_reconstruction = predictions_reconstruction.reshape(
            #     batch_size, self.consecutive_slices, *predictions_reconstruction.shape[1:]
            # )

            target_segmentation = target_segmentation[:, self.consecutive_slices // 2]
            target_reconstruction = target_reconstruction[:, self.consecutive_slices // 2]
            predictions_segmentation = predictions_segmentation[:, self.consecutive_slices // 2]
            predictions_reconstruction = predictions_reconstruction[:, self.consecutive_slices // 2]

        # TODO: check this
        if self.num_echoes > 1:
            # find the batch size
            batch_size = target_reconstruction.shape[0] / self.num_echoes
            # reshape to [batch_size, num_echoes, n_x, n_y]
            target_reconstruction = target_reconstruction.reshape(
                (int(batch_size), self.num_echoes, *target_reconstruction.shape[1:])
            )
            predictions_reconstruction = predictions_reconstruction.reshape(
                (int(batch_size), self.num_echoes, *predictions_reconstruction.shape[1:])
            )
        fname = attrs["fname"]
        slice_idx = attrs["slice_idx"]

        # Iterate over the batch and log the target and predictions.
        for _batch_idx_ in range(target_segmentation.shape[0]):
            output_predictions_reconstruction = predictions_reconstruction[_batch_idx_]
            output_target_reconstruction = target_reconstruction[_batch_idx_]
            output_predictions_segmentation = predictions_segmentation[_batch_idx_]
            output_target_segmentation = target_segmentation[_batch_idx_]

            if self.unnormalize_log_outputs:
                # Unnormalize target and predictions with pre normalization values. This is only for logging purposes.
                # For the loss computation, the self.unnormalize_loss_inputs flag is used.
                (
                    output_target_segmentation,
                    output_predictions_segmentation,
                    _,
                ) = self.__unnormalize_for_loss_or_log__(  # type: ignore
                    output_target_segmentation,
                    output_predictions_segmentation,
                    None,
                    attrs,
                    attrs["r"],
                    batch_idx=_batch_idx_,
                )
                (
                    output_target_reconstruction,
                    output_predictions_reconstruction,
                    _,
                ) = self.__unnormalize_for_loss_or_log__(  # type: ignore
                    output_target_reconstruction,
                    output_predictions_reconstruction,
                    None,
                    attrs,
                    attrs["r"],
                    batch_idx=_batch_idx_,
                )

            # Normalize target and predictions to [0, 1] for logging.
            if torch.is_complex(output_target_reconstruction) and output_target_reconstruction.shape[-1] != 2:
                output_target_reconstruction = torch.view_as_real(output_target_reconstruction)
            if output_target_reconstruction.shape[-1] == 2:
                output_target_reconstruction = complex_abs(output_target_reconstruction)
            output_target_reconstruction = output_target_reconstruction / torch.max(
                torch.abs(output_target_reconstruction)
            )

            if (
                torch.is_complex(output_predictions_reconstruction)
                and output_predictions_reconstruction.shape[-1] != 2
            ):
                output_predictions_reconstruction = torch.view_as_real(output_predictions_reconstruction)
            if output_predictions_reconstruction.shape[-1] == 2:
                output_predictions_reconstruction = complex_abs(output_predictions_reconstruction)
            output_predictions_reconstruction = output_predictions_reconstruction / torch.max(
                torch.abs(output_predictions_reconstruction)
            )
            output_predictions_reconstruction = output_predictions_reconstruction.detach().cpu().float()
            output_target_reconstruction = output_target_reconstruction.detach().cpu().float()
            output_target_segmentation = output_target_segmentation.detach().cpu().float()
            output_predictions_segmentation = output_predictions_segmentation.detach().cpu().float()

            if self.segmentation_mode == 'multiclass':
                output_predictions_segmentation = torch.softmax(output_predictions_segmentation, dim=0).float()
                if self.segmentation_output_mode == "binary":
                    output_predictions_segmentation = output_predictions_segmentation.argmax(dim=0, keepdim=True)
                    output_predictions_segmentation = one_hot(
                        output_predictions_segmentation, num_classes=self.segmentation_module_output_channels, dim=0
                    )
            else:
                # When using wandb plots needs to be between [0,1]. When using "multilabel" with/without thresholding
                # the outputs are logits and exceed this range.
                output_predictions_segmentation = output_predictions_segmentation.clamp(0, 1).float()
                if self.segmentation_output_mode == "binary":
                    output_predictions_segmentation = torch.where(output_predictions_segmentation > 0.5, 1, 0).float()

            # Log target and predictions, if log_image is True for this slice.
            if attrs["log_image"][_batch_idx_]:
                key = f"{fname[_batch_idx_]}_slice_{int(slice_idx[_batch_idx_])}"

                if self.log_multiple_modalities:
                    # concatenate the reconstruction predictions for logging
                    output_target_reconstruction = torch.cat(
                        [output_target_reconstruction[i] for i in range(output_target_reconstruction.shape[0])], dim=-1
                    )

                if self.use_reconstruction_module:
                    # TODO: check this
                    if self.num_echoes > 1:
                        for i in range(output_target_reconstruction.shape[0]):
                            self.log_image(
                                f"{key}/a/reconstruction_abs/target echo: {i+1}/predictions echo: {i+1}/error echo: "
                                f"{i+1}",
                                torch.cat(
                                    [
                                        output_target_reconstruction[i],
                                        output_predictions_reconstruction[i],
                                        torch.abs(
                                            output_target_reconstruction[i] - output_predictions_reconstruction[i]
                                        ),
                                    ],
                                    dim=-1,
                                ),
                            )
                    else:
                        self.log_image(
                            f"{key}/a/reconstruction_abs/target/predictions/error",
                            torch.cat(
                                [
                                    output_target_reconstruction,
                                    output_predictions_reconstruction,
                                    torch.abs(output_target_reconstruction - output_predictions_reconstruction),
                                ],
                                dim=-1,
                            ),
                        )

                # concatenate the segmentation classes for logging
                target_segmentation_class = torch.cat(
                    [output_target_segmentation[i] for i in range(output_target_segmentation.shape[0])], dim=-1
                )
                output_predictions_segmentation_class = torch.cat(
                    [output_predictions_segmentation[i] for i in range(output_predictions_segmentation.shape[0])],
                    dim=-1,
                )
                self.log_image(f"{key}/b/segmentation/target", target_segmentation_class)
                self.log_image(f"{key}/c/segmentation/predictions", output_predictions_segmentation_class)
                self.log_image(
                    f"{key}/d/segmentation/error",
                    torch.abs(target_segmentation_class - output_predictions_segmentation_class),
                )

            # Compute metrics and log them.
            output_predictions_reconstruction = output_predictions_reconstruction.numpy()
            output_target_reconstruction = output_target_reconstruction.numpy()

            self.mse_vals_reconstruction[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(
                mse(output_target_reconstruction, output_predictions_reconstruction)
            ).view(1)
            self.nmse_vals_reconstruction[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(
                nmse(output_target_reconstruction, output_predictions_reconstruction)
            ).view(1)

            max_value = max(np.max(output_target_reconstruction), np.max(output_predictions_reconstruction)) - min(
                np.min(output_target_reconstruction), np.min(output_predictions_reconstruction)
            )

            self.ssim_vals_reconstruction[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(
                ssim(output_target_reconstruction, output_predictions_reconstruction, maxval=max_value)
            ).view(1)
            self.psnr_vals_reconstruction[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(
                psnr(output_target_reconstruction, output_predictions_reconstruction, maxval=max_value)
            ).view(1)

            if self.cross_entropy_metric is not None:
                self.cross_entropy_vals[fname[_batch_idx_]][
                    str(slice_idx[_batch_idx_].item())
                ] = self.cross_entropy_metric(
                    output_target_segmentation.to(self.device),
                    output_predictions_segmentation.to(self.device),
                )
            dice_score, _ = self.dice_metric(output_target_segmentation, output_predictions_segmentation)
            self.dice_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = dice_score

    def __check_noise_to_recon_inputs__(
        self, y: torch.Tensor, mask: torch.Tensor, initial_prediction: torch.Tensor, attrs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Checks if Noise-to-Recon [1] is used.

        References
        ----------
        .. [1] Desai, AD, Ozturkler, BM, Sandino, CM, et al. Noise2Recon: Enabling SNR-robust MRI reconstruction with
        semi-supervised and self-supervised learning. Magn Reson Med. 2023; 90(5): 2052-2070. doi: 10.1002/mrm.29759

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1,  n_x, n_y, 1].
        initial_prediction : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2].
        attrs : Dict
            Attributes dictionary. Even though Noise-to-Recon is an unsupervised method, a percentage of the data might
             be used for supervised learning. In this case, the ``attrs["n2r_supervised"]`` will be True. So we know
             which data are used for supervised learning and which for unsupervised.

        Returns
        -------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1,  n_x, n_y, 1].
        initial_prediction : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2].
        n2r_y : torch.Tensor
            Subsampled k-space data for Noise-to-Recon. Shape [batch_size, n_coils, n_x, n_y, 2].
        n2r_mask : torch.Tensor
            Sampling mask for Noise-to-Recon. Shape [batch_size, 1,  n_x, n_y, 1].
        n2r_initial_prediction : torch.Tensor
            Initial prediction for Noise-to-Recon. Shape [batch_size, n_x, n_y, 2].
        """
        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):
            y, n2r_y = y
            mask, n2r_mask = mask
            initial_prediction, n2r_initial_prediction = initial_prediction
        else:
            n2r_y = None
            n2r_mask = None
            n2r_initial_prediction = None
        return y, mask, initial_prediction, n2r_y, n2r_mask, n2r_initial_prediction

    def __process_unsupervised_inputs__(
        self,
        n2r_y: torch.Tensor,
        mask: torch.Tensor,
        n2r_mask: torch.Tensor,
        n2r_initial_prediction: torch.Tensor,
        attrs: Dict,
        r: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process inputs if Noise-to-Recon and/or SSDU are used.

        Parameters
        ----------
        n2r_y : Union[List[torch.Tensor], torch.Tensor]
            Noise-to-Recon subsampled k-space data, if Noise-to-Recon is used. If multiple accelerations are used, then
            it is a list of torch.Tensor. Shape [batch_size, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1,  n_x, n_y, 1].
        n2r_mask : Union[List[torch.Tensor], torch.Tensor]
            Noise-to-Recon sampling mask, if Noise-to-Recon is used. If multiple accelerations are used, then
            it is a list of torch.Tensor. Shape [batch_size, 1,  n_x, n_y, 1].
        n2r_initial_prediction : Union[List[torch.Tensor], torch.Tensor]
            Noise-to-Recon initial prediction, if Noise-to-Recon is used. If multiple accelerations are used, then
            it is a list of torch.Tensor. Shape [batch_size, n_x, n_y, 2].
        attrs : Dict
            Attributes dictionary. Even though Noise-to-Recon is an unsupervised method, a percentage of the data might
             be used for supervised learning. In this case, the ``attrs["n2r_supervised"]`` will be True. So we know
             which data are used for supervised learning and which for unsupervised.
        r : int
            Random index used to select the acceleration.

        Returns
        -------
        n2r_y : torch.Tensor
            Noise-to-Recon subsampled k-space data, if Noise-to-Recon is used. If multiple accelerations are used, then
            one factor is randomly selected. Shape [batch_size, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1,  n_x, n_y, 1].
        n2r_mask : torch.Tensor
            Noise-to-Recon sampling mask, if Noise-to-Recon is used. If multiple accelerations are used, then one
            factor is randomly selected. Shape [batch_size, 1,  n_x, n_y, 1].
        n2r_initial_prediction : torch.Tensor
            Noise-to-Recon initial prediction, if Noise-to-Recon is used. If multiple accelerations are used, then one
            factor is randomly selected. Shape [batch_size, n_x, n_y, 2].
        loss_mask : torch.Tensor
            SSDU loss mask, if SSDU is used. Shape [batch_size, 1,  n_x, n_y, 1].
        """
        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):
            # Noise-to-Recon with/without SSDU.

            if isinstance(n2r_y, list):
                # Check multiple accelerations for Noise-to-Recon
                n2r_y = n2r_y[r]
                if n2r_mask is not None:
                    n2r_mask = n2r_mask[r]
                n2r_initial_prediction = n2r_initial_prediction[r]

            # Check if SSDU is used
            if self.ssdu:
                mask, loss_mask = mask
            else:
                loss_mask = torch.ones_like(mask)

            # Ensure that the mask has the same number of dimensions as the input mask.
            if n2r_mask.dim() < mask.dim():
                n2r_mask = None
        elif self.ssdu and not self.n2r:
            # SSDU without Noise-to-Recon.
            mask, loss_mask = mask
        else:
            loss_mask = torch.ones_like(mask)

        return n2r_y, n2r_mask, n2r_initial_prediction, mask, loss_mask

    @staticmethod
    def __process_inputs__(
        kspace: Union[List, torch.Tensor],
        y: Union[List, torch.Tensor],
        mask: Union[List, torch.Tensor],
        initial_prediction: Union[List, torch.Tensor],
        target: Union[List, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Processes lists of inputs to torch.Tensor. In the case where multiple accelerations are used, then the
        inputs are lists. This function converts the lists to torch.Tensor by randomly selecting one acceleration. If
        only one acceleration is used, then the inputs are torch.Tensor and are returned as is.

        Parameters
        ----------
        kspace : Union[List, torch.Tensor]
            Full k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        y : Union[List, torch.Tensor]
            Subsampled k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : Union[List, torch.Tensor]
            Sampling mask of length n_accelerations or shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction : Union[List, torch.Tensor]
            Initial prediction of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        target : Union[List, torch.Tensor]
            Target data of length n_accelerations or shape [batch_size, n_x, n_y, 2].

        Returns
        -------
        kspace : torch.Tensor
            Full k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction : torch.Tensor
            Initial prediction of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        r : int
            Random index used to select the acceleration.
        """
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
            initial_prediction = initial_prediction[r]
        else:
            r = 0
        if isinstance(kspace, list):
            kspace = kspace[r]
            target = target[r]
        elif isinstance(target, list):
            target = target[r]
        return kspace, y, mask, initial_prediction, target, r

    def inference_step(  # noqa: MC0001
        self,
        kspace: torch.Tensor,
        y: Union[List[torch.Tensor], torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: Union[List[torch.Tensor], torch.Tensor],
        initial_prediction_reconstruction: Union[List, torch.Tensor],
        target_reconstruction: torch.Tensor,
        target_segmentation: torch.Tensor,
        fname: str,
        slice_idx: int,
        acceleration: float,
        attrs: Dict,
    ):
        """Performs an inference step, i.e., computes the predictions of the model.

        Parameters
        ----------
        kspace : torch.Tensor
            Fully sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
        y : Union[List[torch.Tensor], torch.Tensor]
            Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
            Shape [batch_size, n_coils, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
        mask : Union[List[torch.Tensor], torch.Tensor]
            Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if Unsupervised
            Learning methods are used, it contains their masks. Shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction_reconstruction : Union[List, torch.Tensor]
            Initial reconstruction prediction. If multiple accelerations are used, then it is a list of torch.Tensor.
            Shape [batch_size, n_x, n_y, 2].
        target_reconstruction : torch.Tensor
            Target reconstruction data. Shape [batch_size, n_x, n_y].
        target_segmentation : torch.Tensor
            Target segmentation data. Shape [batch_size, n_x, n_y].
        fname : str
            File name.
        slice_idx : int
            Slice index.
        acceleration : float
            Acceleration factor of the sampling mask, randomly selected if multiple accelerations are used.
        attrs : Dict
            Attributes dictionary.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of processed inputs and model's predictions, with keys:
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask, randomly selected if multiple accelerations are used.
                'predictions_reconstruction' : Union[List[torch.Tensor], torch.Tensor]
                    Model's predictions. If accumulate predictions is True, then it is a list of torch.Tensor.
                    Shape [batch_size, n_x, n_y, 2].
                'predictions_reconstruction_n2r' : Union[List[torch.Tensor], torch.Tensor]
                    Model's predictions for Noise-to-Recon, if Noise-to-Recon is used. If accumulate predictions is
                    True, then it is a list of torch.Tensor. Shape [batch_size, n_x, n_y, 2].
                'target_reconstruction' : torch.Tensor
                    Target data. Shape [batch_size, n_x, n_y].
                'target_segmentation' : torch.Tensor
                    Target segmentation data. Shape [batch_size, n_x, n_y].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'loss_mask' : torch.Tensor
                    SSDU loss mask, if SSDU is used. Shape [batch_size, 1,  n_x, n_y, 1].
                'attrs' : dict
                    Attributes dictionary.
                'r' : int
                    Random index used for selected acceleration.
        """
        # Check if Noise-to-Recon is used
        (
            y,
            mask,
            initial_prediction_reconstruction,
            n2r_y,
            n2r_mask,
            n2r_initial_prediction_reconstruction,
        ) = self.__check_noise_to_recon_inputs__(y, mask, initial_prediction_reconstruction, attrs)

        # Process inputs to randomly select one acceleration factor, in case multiple accelerations are used.
        kspace, y, mask, initial_prediction_reconstruction, target_reconstruction, r = self.__process_inputs__(
            kspace, y, mask, initial_prediction_reconstruction, target_reconstruction
        )

        # Process inputs if Noise-to-Recon and/or SSDU are used.
        n2r_y, n2r_mask, n2r_initial_prediction_reconstruction, mask, loss_mask = self.__process_unsupervised_inputs__(
            n2r_y, mask, n2r_mask, n2r_initial_prediction_reconstruction, attrs, r
        )

        # Check if a network is used for coil sensitivity maps estimation.
        if self.estimate_coil_sensitivity_maps_with_nn:
            # Estimate coil sensitivity maps with a network.
            sensitivity_maps = self.coil_sensitivity_maps_nn(kspace, mask, sensitivity_maps)
            # (Re-)compute the initial prediction with the estimated sensitivity maps. This also means that the
            # self.coil_combination_method is set to "SENSE", since in "RSS" the sensitivity maps are not used.
            initial_prediction_reconstruction = coil_combination_method(
                ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )
            if n2r_initial_prediction_reconstruction is not None:
                n2r_initial_prediction_reconstruction = coil_combination_method(
                    ifft2(n2r_y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                    sensitivity_maps,
                    self.coil_combination_method,
                    self.coil_dim,
                )

        # TODO: check this
        if self.num_echoes > 1:
            # stack the echoes along the batch dimension
            kspace = kspace.view(-1, *kspace.shape[2:])
            y = y.view(-1, *y.shape[2:])
            mask = mask.view(-1, *mask.shape[2:])
            initial_prediction_reconstruction = initial_prediction_reconstruction.view(
                -1, *initial_prediction_reconstruction.shape[2:]
            )
            target_reconstruction = target_reconstruction.view(-1, *target_reconstruction.shape[2:])
            sensitivity_maps = torch.repeat_interleave(sensitivity_maps, repeats=kspace.shape[0], dim=0).squeeze(1)

        # Model forward pass
        predictions_reconstruction, predictions_segmentation = self.forward(
            y,
            sensitivity_maps,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            attrs["noise"],
        )

        # TODO: check this
        if self.consecutive_slices > 1:
            ## reshape the target and prediction segmentation to [batch_size * consecutive_slices, nr_classes, n_x, n_y]
            batch_size, slices = target_segmentation.shape[:2]
            target_segmentation = target_segmentation.reshape(batch_size * slices, *target_segmentation.shape[2:])
            if isinstance(predictions_segmentation, list):
                for i, prediction_segmentation in enumerate(predictions_segmentation):
                    predictions_segmentation[i] = prediction_segmentation.reshape(
                        batch_size * slices, *prediction_segmentation.shape[2:]
                    )
            else:
                predictions_segmentation = predictions_segmentation.reshape(
                    batch_size * slices, *predictions_segmentation.shape[2:]
                )

        if not is_none(self.segmentation_classes_thresholds) and self.segmentation_mode == 'multilabel':
            for class_idx, thres in enumerate(self.segmentation_classes_thresholds):
                if self.segmentation_activation == "sigmoid":
                    if isinstance(predictions_segmentation, list):
                        cond = [torch.sigmoid(pred[:, class_idx]) for pred in predictions_segmentation]
                    else:
                        cond = torch.sigmoid(predictions_segmentation[:, class_idx])
                elif self.segmentation_activation == "softmax":
                    if isinstance(predictions_segmentation, list):
                        cond = [torch.softmax(pred[:, class_idx], dim=1) for pred in predictions_segmentation]
                    else:
                        cond = torch.softmax(predictions_segmentation[:, class_idx], dim=1)
                else:
                    if isinstance(predictions_segmentation, list):
                        cond = [pred[:, class_idx] for pred in predictions_segmentation]
                    else:
                        cond = predictions_segmentation[:, class_idx]

                if isinstance(predictions_segmentation, list):
                    for idx, pred in enumerate(predictions_segmentation):
                        predictions_segmentation[idx][:, class_idx] = torch.where(
                            cond[idx] >= thres,
                            predictions_segmentation[idx][:, class_idx],
                            torch.zeros_like(predictions_segmentation[idx][:, class_idx]),
                        )
                else:
                    predictions_segmentation[:, class_idx] = torch.where(
                        cond >= thres,
                        predictions_segmentation[:, class_idx],
                        torch.zeros_like(predictions_segmentation[:, class_idx]),
                    )

        # Noise-to-Recon forward pass, if Noise-to-Recon is used.
        predictions_reconstruction_n2r = None
        if self.n2r and n2r_mask is not None:
            predictions_reconstruction_n2r = self.forward(
                n2r_y,
                sensitivity_maps,
                n2r_mask,
                n2r_initial_prediction_reconstruction,
                target_reconstruction,
                attrs["noise"],
            )

        # Get acceleration factor from acceleration list, if multiple accelerations are used. Or if batch size > 1.
        if isinstance(acceleration, list):
            if acceleration[0].shape[0] > 1:
                acceleration[0] = acceleration[0][0]
            acceleration = np.round(acceleration[r].item())
        else:
            if acceleration.shape[0] > 1:  # type: ignore
                acceleration = acceleration[0]  # type: ignore
            acceleration = np.round(acceleration.item())  # type: ignore

        # Pass r to the attrs dictionary, so that it can be used in unnormalize_for_loss_or_log if needed.
        attrs["r"] = r

        return {
            "fname": fname,
            "slice_idx": slice_idx,
            "acceleration": acceleration,
            "predictions_reconstruction": predictions_reconstruction,
            "predictions_reconstruction_n2r": predictions_reconstruction_n2r,
            "predictions_segmentation": predictions_segmentation,
            "target_reconstruction": target_reconstruction,
            "target_segmentation": target_segmentation,
            "sensitivity_maps": sensitivity_maps,
            "loss_mask": loss_mask,
            "attrs": attrs,
            "r": r,
        }

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Performs a training step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : Union[torch.Tensor, None]
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'initial_prediction_reconstruction': torch.Tensor
                    Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2].
                'target_reconstruction': torch.Tensor
                    Target reconstruction. Shape [batch_size, n_x, n_y].
                'target_segmentation': Union[torch.Tensor, None]
                    Target segmentation. Shape [batch_size, n_x, n_y].
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        (
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            target_segmentation,
            fname,
            slice_idx,
            acceleration,
            attrs,
        ) = batch

        outputs = self.inference_step(
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            target_segmentation,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        # Compute loss
        train_loss = self.__compute_loss__(
            outputs["predictions_reconstruction"],
            outputs["predictions_reconstruction_n2r"],
            outputs["target_reconstruction"],
            outputs["predictions_segmentation"],
            outputs["target_segmentation"],
            outputs["sensitivity_maps"],
            outputs["loss_mask"],
            outputs["attrs"],
            outputs["r"],
        )

        # Log loss for the chosen acceleration factor and the learning rate in the selected logger.
        logs = {
            f'train_loss_{outputs["acceleration"]}x': train_loss.item(),
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }

        self.log(
            "train_joint_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=target_segmentation.shape[0],  # type: ignore
            sync_dist=True,
        )

        return {"loss": train_loss, "log": logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : Union[torch.Tensor, None]
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'initial_prediction_reconstruction': torch.Tensor
                    Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2].
                'target_reconstruction': torch.Tensor
                    Target reconstruction. Shape [batch_size, n_x, n_y].
                'target_segmentation': Union[torch.Tensor, None]
                    Target segmentation. Shape [batch_size, n_x, n_y].
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.
        batch_idx : int
            Batch index.
        """
        (
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            target_segmentation,
            fname,
            slice_idx,
            acceleration,
            attrs,
        ) = batch

        outputs = self.inference_step(
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            target_segmentation,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        predictions_reconstruction = outputs["predictions_reconstruction"]
        predictions_reconstruction_n2r = outputs["predictions_reconstruction_n2r"]
        target_reconstruction = outputs["target_reconstruction"]
        predictions_segmentation = outputs["predictions_segmentation"]
        target_segmentation = outputs["target_segmentation"]

        # Compute loss
        val_loss = self.__compute_loss__(
            predictions_reconstruction,
            predictions_reconstruction_n2r,
            target_reconstruction,
            predictions_segmentation,
            target_segmentation,
            outputs["sensitivity_maps"],
            outputs["loss_mask"],
            outputs["attrs"],
            outputs["r"],
        )

        self.validation_step_outputs.append({"val_loss": val_loss})

        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            predictions_reconstruction,
            target_reconstruction,
            predictions_segmentation,
            target_segmentation,
            outputs["attrs"],
        )

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a test step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : Union[torch.Tensor, None]
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'initial_prediction_reconstruction': torch.Tensor
                    Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2].
                'target_reconstruction': torch.Tensor
                    Target reconstruction. Shape [batch_size, n_x, n_y].
                'target_segmentation': Union[torch.Tensor, None]
                    Target segmentation. Shape [batch_size, n_x, n_y].
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.
        batch_idx : int
            Batch index.
        """
        (
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            target_segmentation,
            fname,
            slice_idx,
            acceleration,
            attrs,
        ) = batch

        outputs = self.inference_step(
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            target_segmentation,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        predictions_reconstruction = outputs["predictions_reconstruction"]
        predictions_segmentation = outputs["predictions_segmentation"]

        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            predictions_reconstruction,
            outputs["target_reconstruction"],
            predictions_segmentation,
            outputs["target_segmentation"],
            outputs["attrs"],
        )

        if isinstance(predictions_segmentation, list):
            while isinstance(predictions_segmentation, list):
                predictions_segmentation = predictions_segmentation[-1]

        if self.segmentation_mode == 'multiclass':
            predictions_segmentation = torch.softmax(predictions_segmentation, dim=1).float()
            if self.segmentation_output_mode == "binary":
                predictions_segmentation = predictions_segmentation.argmax(dim=1, keepdim=True)
                predictions_segmentation = one_hot(
                    predictions_segmentation, num_classes=self.segmentation_module_output_channels, dim=1
                )
        elif self.segmentation_output_mode == "binary":
            predictions_segmentation = torch.where(predictions_segmentation > 0.5, 1, 0).float()

        predictions_segmentation = predictions_segmentation.detach().cpu().numpy()

        if self.use_reconstruction_module:
            if isinstance(predictions_reconstruction, list):
                while isinstance(predictions_reconstruction, list):
                    predictions_reconstruction = predictions_reconstruction[-1]

            # TODO: check this
            if self.consecutive_slices > 1:
                # reshape the target and prediction to [batch_size, self.consecutive_slices, nr_classes, n_x, n_y]
                batch_size = int(target_segmentation.shape[0] / self.consecutive_slices)
                predictions_segmentation = predictions_segmentation.reshape(
                    batch_size, self.consecutive_slices, *predictions_segmentation.shape[1:]
                )
                predictions_segmentation = predictions_segmentation[:, self.consecutive_slices // 2]
                predictions_reconstruction = predictions_reconstruction[:, self.consecutive_slices // 2]

            # TODO: check this
            if self.num_echoes > 1:
                # find the batch size
                batch_size = target_reconstruction.shape[0] / self.num_echoes
                # reshape to [batch_size, num_echoes, n_x, n_y]
                predictions_reconstruction = predictions_reconstruction.reshape(
                    (int(batch_size), self.num_echoes, *predictions_reconstruction.shape[1:])
                )

            # If "16" or "16-mixed" fp is used, ensure complex type will be supported when saving the predictions.
            predictions_reconstruction = (
                torch.view_as_complex(torch.view_as_real(predictions_reconstruction).type(torch.float32))
                .detach()
                .cpu()
                .numpy()
            )

        predictions = (
            (predictions_segmentation, predictions_reconstruction)
            if self.use_reconstruction_module
            else (predictions_segmentation, predictions_segmentation)
        )

        self.test_step_outputs.append([str(fname[0]), slice_idx, predictions])  # type: ignore

    def on_validation_epoch_end(self):  # noqa: MC0001
        """Called at the end of validation epoch to aggregate outputs.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        self.log("val_loss", torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean(), sync_dist=True)

        # Log metrics.
        if self.cross_entropy_metric is not None:
            cross_entropy_vals = defaultdict(dict)
            for k, v in self.cross_entropy_vals.items():
                cross_entropy_vals[k].update(v)

        dice_vals = defaultdict(dict)
        for k, v in self.dice_vals.items():
            dice_vals[k].update(v)

        metrics_segmentation = {"Cross_Entropy": 0, "DICE": 0}

        if self.use_reconstruction_module:
            mse_vals_reconstruction = defaultdict(dict)
            nmse_vals_reconstruction = defaultdict(dict)
            ssim_vals_reconstruction = defaultdict(dict)
            psnr_vals_reconstruction = defaultdict(dict)

            for k, v in self.mse_vals_reconstruction.items():
                mse_vals_reconstruction[k].update(v)
            for k, v in self.nmse_vals_reconstruction.items():
                nmse_vals_reconstruction[k].update(v)
            for k, v in self.ssim_vals_reconstruction.items():
                ssim_vals_reconstruction[k].update(v)
            for k, v in self.psnr_vals_reconstruction.items():
                psnr_vals_reconstruction[k].update(v)

            metrics_reconstruction = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}

        if self.metric_computation_mode == "per_volume":
            local_examples = 0
            for fname in dice_vals:
                local_examples += 1
                if self.cross_entropy_metric is not None:
                    metrics_segmentation["Cross_Entropy"] = metrics_segmentation["Cross_Entropy"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in cross_entropy_vals[fname].items()])
                    )
                metrics_segmentation["DICE"] = metrics_segmentation["DICE"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in dice_vals[fname].items()])
                )

                if self.use_reconstruction_module:
                    metrics_reconstruction["MSE"] = metrics_reconstruction["MSE"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in mse_vals_reconstruction[fname].items()])
                    )
                    metrics_reconstruction["NMSE"] = metrics_reconstruction["NMSE"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in nmse_vals_reconstruction[fname].items()])
                    )
                    metrics_reconstruction["SSIM"] = metrics_reconstruction["SSIM"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in ssim_vals_reconstruction[fname].items()])
                    )
                    metrics_reconstruction["PSNR"] = metrics_reconstruction["PSNR"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in psnr_vals_reconstruction[fname].items()])
                    )
        else:  # per-slice
            if self.cross_entropy_metric is not None:
                metrics_segmentation["Cross_Entropy"] = torch.sum(
                    torch.stack([v for x in cross_entropy_vals.values() for v in x.values()])
                )
            dice_vals = [v for x in dice_vals.values() for v in x.values()]
            local_examples = len(dice_vals)
            metrics_segmentation["DICE"] = torch.sum(torch.stack(dice_vals))

            if self.use_reconstruction_module:
                metrics_segmentation["MSE"] = torch.sum(
                    torch.stack([v for x in mse_vals_reconstruction.values() for v in x.values()])
                )
                metrics_segmentation["NMSE"] = torch.sum(
                    torch.stack([v for x in nmse_vals_reconstruction.values() for v in x.values()])
                )
                metrics_segmentation["SSIM"] = torch.sum(
                    torch.stack([v for x in ssim_vals_reconstruction.values() for v in x.values()])
                )
                metrics_segmentation["PSNR"] = torch.sum(
                    torch.stack([v for x in psnr_vals_reconstruction.values() for v in x.values()])
                )

        # reduce across ddp via sum
        if self.cross_entropy_metric is not None:
            metrics_segmentation["Cross_Entropy"] = self.CROSS_ENTROPY(metrics_segmentation["Cross_Entropy"])
        metrics_segmentation["DICE"] = self.DICE(metrics_segmentation["DICE"])

        if self.use_reconstruction_module:
            metrics_reconstruction["MSE"] = self.MSE(metrics_reconstruction["MSE"])
            metrics_reconstruction["NMSE"] = self.NMSE(metrics_reconstruction["NMSE"])
            metrics_reconstruction["SSIM"] = self.SSIM(metrics_reconstruction["SSIM"])
            metrics_reconstruction["PSNR"] = self.PSNR(metrics_reconstruction["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics_segmentation.items():
            self.log(f"val_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)
        if self.use_reconstruction_module:
            for metric, value in metrics_reconstruction.items():
                self.log(f"val_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):  # noqa: MC0001
        """Called at the end of test epoch to aggregate outputs, log metrics and save predictions.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        # Log metrics.
        if self.cross_entropy_metric is not None:
            cross_entropy_vals = defaultdict(dict)
            for k, v in self.cross_entropy_vals.items():
                cross_entropy_vals[k].update(v)

        dice_vals = defaultdict(dict)
        for k, v in self.dice_vals.items():
            dice_vals[k].update(v)

        metrics_segmentation = {"Cross_Entropy": 0, "DICE": 0}

        if self.use_reconstruction_module:
            mse_vals_reconstruction = defaultdict(dict)
            nmse_vals_reconstruction = defaultdict(dict)
            ssim_vals_reconstruction = defaultdict(dict)
            psnr_vals_reconstruction = defaultdict(dict)

            for k, v in self.mse_vals_reconstruction.items():
                mse_vals_reconstruction[k].update(v)
            for k, v in self.nmse_vals_reconstruction.items():
                nmse_vals_reconstruction[k].update(v)
            for k, v in self.ssim_vals_reconstruction.items():
                ssim_vals_reconstruction[k].update(v)
            for k, v in self.psnr_vals_reconstruction.items():
                psnr_vals_reconstruction[k].update(v)

            metrics_reconstruction = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}

        if self.metric_computation_mode == "per_volume":
            local_examples = 0
            for fname in dice_vals:
                local_examples += 1
                if self.cross_entropy_metric is not None:
                    metrics_segmentation["Cross_Entropy"] = metrics_segmentation["Cross_Entropy"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in cross_entropy_vals[fname].items()])
                    )
                metrics_segmentation["DICE"] = metrics_segmentation["DICE"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in dice_vals[fname].items()])
                )

                if self.use_reconstruction_module:
                    metrics_reconstruction["MSE"] = metrics_reconstruction["MSE"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in mse_vals_reconstruction[fname].items()])
                    )
                    metrics_reconstruction["NMSE"] = metrics_reconstruction["NMSE"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in nmse_vals_reconstruction[fname].items()])
                    )
                    metrics_reconstruction["SSIM"] = metrics_reconstruction["SSIM"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in ssim_vals_reconstruction[fname].items()])
                    )
                    metrics_reconstruction["PSNR"] = metrics_reconstruction["PSNR"] + torch.mean(
                        torch.cat([v.view(-1) for _, v in psnr_vals_reconstruction[fname].items()])
                    )
        else:  # per-slice
            if self.cross_entropy_metric is not None:
                metrics_segmentation["Cross_Entropy"] = torch.sum(
                    torch.stack([v for x in cross_entropy_vals.values() for v in x.values()])
                )
            dice_vals = [v for x in dice_vals.values() for v in x.values()]
            local_examples = len(dice_vals)
            metrics_segmentation["DICE"] = torch.sum(torch.stack(dice_vals))

            if self.use_reconstruction_module:
                metrics_segmentation["MSE"] = torch.sum(
                    torch.stack([v for x in mse_vals_reconstruction.values() for v in x.values()])
                )
                metrics_segmentation["NMSE"] = torch.sum(
                    torch.stack([v for x in nmse_vals_reconstruction.values() for v in x.values()])
                )
                metrics_segmentation["SSIM"] = torch.sum(
                    torch.stack([v for x in ssim_vals_reconstruction.values() for v in x.values()])
                )
                metrics_segmentation["PSNR"] = torch.sum(
                    torch.stack([v for x in psnr_vals_reconstruction.values() for v in x.values()])
                )

        # reduce across ddp via sum
        if self.cross_entropy_metric is not None:
            metrics_segmentation["Cross_Entropy"] = self.CROSS_ENTROPY(metrics_segmentation["Cross_Entropy"])
        metrics_segmentation["DICE"] = self.DICE(metrics_segmentation["DICE"])

        if self.use_reconstruction_module:
            metrics_reconstruction["MSE"] = self.MSE(metrics_reconstruction["MSE"])
            metrics_reconstruction["NMSE"] = self.NMSE(metrics_reconstruction["NMSE"])
            metrics_reconstruction["SSIM"] = self.SSIM(metrics_reconstruction["SSIM"])
            metrics_reconstruction["PSNR"] = self.PSNR(metrics_reconstruction["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics_segmentation.items():
            self.log(f"test_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)
        if self.use_reconstruction_module:
            for metric, value in metrics_reconstruction.items():
                self.log(f"test_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)

        segmentations = defaultdict(list)
        for fname, slice_num, output in self.test_step_outputs:
            segmentations_pred, _ = output
            segmentations[fname].append((slice_num, segmentations_pred))

        for fname in segmentations:
            segmentations[fname] = np.stack([out for _, out in sorted(segmentations[fname])])

        # TODO: check why this have to go
        # if self.consecutive_slices > 1:
        #     # iterate over the slices and always keep the middle slice
        #     for fname in segmentations:
        #         segmentations[fname] = segmentations[fname][:, self.consecutive_slices // 2] #TODO remove, is already done in the test_step to minimize memory load

        if self.use_reconstruction_module:
            reconstructions = defaultdict(list)
            for fname, slice_num, output in self.test_step_outputs:
                _, reconstructions_pred = output
                reconstructions[fname].append((slice_num, reconstructions_pred))

            for fname in reconstructions:
                reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])

            # TODO: check why this have to go
            # if self.consecutive_slices > 1: #TODO remove, is already done in the test_step to minimize memory load
            #     # iterate over the slices and always keep the middle slice
            #     for fname in reconstructions:
            #         reconstructions[fname] = reconstructions[fname][:, self.consecutive_slices // 2]
        else:
            reconstructions = None

        if "wandb" in self.logger.__module__.lower():
            out_dir = Path(os.path.join(self.logger.save_dir, "predictions"))
        else:
            out_dir = Path(os.path.join(self.logger.log_dir, "predictions"))
        out_dir.mkdir(exist_ok=True, parents=True)

        if reconstructions is not None:
            for (fname, segmentations_pred), (_, reconstructions_pred) in zip(
                segmentations.items(), reconstructions.items()
            ):
                with h5py.File(out_dir / fname, "w") as hf:
                    hf.create_dataset("segmentation", data=segmentations_pred)
                    hf.create_dataset("reconstruction", data=reconstructions_pred)
        else:
            for fname, segmentations_pred in segmentations.items():
                with h5py.File(out_dir / fname, "w") as hf:
                    hf.create_dataset("segmentation", data=segmentations_pred)

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg : DictConfig
            Configuration file.

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            Dataloader.
        """
        mask_root = cfg.get("mask_path", None)
        mask_args = cfg.get("mask_args", None)
        shift_mask = mask_args.get("shift_mask", False)
        mask_type = mask_args.get("type", None)

        mask_func = None
        mask_center_scale = 0.02

        if is_none(mask_root) and not is_none(mask_type):
            accelerations = mask_args.get("accelerations", [1])
            accelerations = list(accelerations)
            if len(accelerations) == 1:
                accelerations = accelerations * 2
            center_fractions = mask_args.get("center_fractions", [1])
            center_fractions = list(center_fractions)
            if len(center_fractions) == 1:
                center_fractions = center_fractions * 2
            mask_center_scale = mask_args.get("center_scale", 0.02)

            mask_func = [create_masker(mask_type, center_fractions, accelerations)]

        complex_data = cfg.get("complex_data", True)

        dataset_format = cfg.get("dataset_format", None)
        if dataset_format.lower() in (
            "skm-tea-echo1",
            "skm-tea-echo2",
            "skm-tea-echo1+echo2",
            "skm-tea-echo1+echo2-mc",
        ):
            dataloader = mrirs_loader.SKMTEARSMRIDataset
        else:
            dataloader = mrirs_loader.RSMRIDataset

        dataset = dataloader(
            root=cfg.get("data_path"),
            coil_sensitivity_maps_root=cfg.get("coil_sensitivity_maps_path", None),
            mask_root=mask_root,
            noise_root=cfg.get("noise_path", None),
            initial_predictions_root=cfg.get("initial_predictions_path"),
            dataset_format=dataset_format,
            sample_rate=cfg.get("sample_rate", 1.0),
            volume_sample_rate=cfg.get("volume_sample_rate", None),
            use_dataset_cache=cfg.get("use_dataset_cache", False),
            dataset_cache_file=cfg.get("dataset_cache_file", None),
            num_cols=cfg.get("num_cols", None),
            consecutive_slices=cfg.get("consecutive_slices", 1),
            data_saved_per_slice=cfg.get("data_saved_per_slice", False),
            n2r_supervised_rate=cfg.get("n2r_supervised_rate", 0.0),
            complex_target=cfg.get("complex_target", False),
            log_images_rate=cfg.get("log_images_rate", 1.0),
            transform=RSMRIDataTransforms(
                complex_data=complex_data,
                segmentation_mode=cfg.get("segmentation_mode", "multilabel"),
                dataset_format=dataset_format,
                apply_prewhitening=cfg.get("apply_prewhitening", False),
                find_patch_size=cfg.get("find_patch_size", False),
                prewhitening_scale_factor=cfg.get("prewhitening_scale_factor", 1.0),
                prewhitening_patch_start=cfg.get("prewhitening_patch_start", 10),
                prewhitening_patch_length=cfg.get("prewhitening_patch_length", 30),
                apply_gcc=cfg.get("apply_gcc", False),
                gcc_virtual_coils=cfg.get("gcc_virtual_coils", 10),
                gcc_calib_lines=cfg.get("gcc_calib_lines", 10),
                gcc_align_data=cfg.get("gcc_align_data", False),
                apply_random_motion=cfg.get("apply_random_motion", False),
                random_motion_type=cfg.get("random_motion_type", "gaussian"),
                random_motion_percentage=cfg.get("random_motion_percentage", [10, 10]),
                random_motion_angle=cfg.get("random_motion_angle", 10),
                random_motion_translation=cfg.get("random_motion_translation", 10),
                random_motion_center_percentage=cfg.get("random_motion_center_percentage", 0.02),
                random_motion_num_segments=cfg.get("random_motion_num_segments", 8),
                random_motion_random_num_segments=cfg.get("random_motion_random_num_segments", True),
                random_motion_non_uniform=cfg.get("random_motion_non_uniform", False),
                estimate_coil_sensitivity_maps=cfg.get("estimate_coil_sensitivity_maps", False),
                coil_sensitivity_maps_type=cfg.get("coil_sensitivity_maps_type", "espirit"),
                coil_sensitivity_maps_gaussian_sigma=cfg.get("coil_sensitivity_maps_gaussian_sigma", 0.0),
                coil_sensitivity_maps_espirit_threshold=cfg.get("coil_sensitivity_maps_espirit_threshold", 0.05),
                coil_sensitivity_maps_espirit_kernel_size=cfg.get("coil_sensitivity_maps_espirit_kernel_size", 6),
                coil_sensitivity_maps_espirit_crop=cfg.get("coil_sensitivity_maps_espirit_crop", 0.95),
                coil_sensitivity_maps_espirit_max_iters=cfg.get("coil_sensitivity_maps_espirit_max_iters", 30),
                coil_combination_method=cfg.get("coil_combination_method", "SENSE"),
                dimensionality=cfg.get("dimensionality", 2),
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=mask_center_scale,
                remask=cfg.get("remask", False),
                ssdu=cfg.get("ssdu", False),
                ssdu_mask_type=cfg.get("ssdu_mask_type", "Gaussian"),
                ssdu_rho=cfg.get("ssdu_rho", 0.4),
                ssdu_acs_block_size=cfg.get("ssdu_acs_block_size", (4, 4)),
                ssdu_gaussian_std_scaling_factor=cfg.get("ssdu_gaussian_std_scaling_factor", 4.0),
                ssdu_outer_kspace_fraction=cfg.get("ssdu_outer_kspace_fraction", 0.0),
                ssdu_export_and_reuse_masks=cfg.get("ssdu_export_and_reuse_masks", False),
                n2r=cfg.get("n2r", False),
                n2r_supervised_rate=cfg.get("n2r_supervised_rate", 0.0),
                n2r_probability=cfg.get("n2r_probability", 0.5),
                n2r_std_devs=cfg.get("n2r_std_devs", (0.0, 0.0)),
                n2r_rhos=cfg.get("n2r_rhos", (0.4, 0.4)),
                n2r_use_mask=cfg.get("n2r_use_mask", True),
                unsupervised_masked_target=cfg.get("unsupervised_masked_target", False),
                crop_size=cfg.get("crop_size", None),
                kspace_crop=cfg.get("kspace_crop", False),
                crop_before_masking=cfg.get("crop_before_masking", False),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size", None),
                normalize_inputs=cfg.get("normalize_inputs", True),
                normalization_type=cfg.get("normalization_type", "max"),
                kspace_normalization=cfg.get("kspace_normalization", False),
                fft_centered=cfg.get("fft_centered", False),
                fft_normalization=cfg.get("fft_normalization", "backward"),
                spatial_dims=cfg.get("spatial_dims", None),
                coil_dim=cfg.get("coil_dim", 1),
                consecutive_slices=cfg.get("consecutive_slices", 1),
                use_seed=cfg.get("use_seed", True),
            ),
            segmentations_root=cfg.get("segmentations_path"),
            segmentation_classes=cfg.get("segmentation_classes", 2),
            segmentation_classes_to_remove=cfg.get("segmentation_classes_to_remove", None),
            segmentation_classes_to_combine=cfg.get("segmentation_classes_to_combine", None),
            segmentation_classes_to_separate=cfg.get("segmentation_classes_to_separate", None),
            segmentation_classes_thresholds=cfg.get("segmentation_classes_thresholds", None),
            complex_data=complex_data,
        )
        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.get("batch_size", 1),
            sampler=sampler,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
