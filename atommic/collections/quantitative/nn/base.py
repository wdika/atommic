# coding=utf-8
__author__ = "Dimitris Karkalousos"

import os
import warnings
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import Tensor
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader

# do not import BaseMRIModel and BaseSensitivityModel directly to avoid circular imports
import atommic.collections.common as atommic_common
from atommic.collections.common.data.subsample import create_masker
from atommic.collections.common.losses import VALID_RECONSTRUCTION_LOSSES
from atommic.collections.common.losses.aggregator import AggregatorLoss
from atommic.collections.common.losses.wasserstein import SinkhornDistance
from atommic.collections.common.parts import fft, utils
from atommic.collections.quantitative.data.qmri_loader import AHEADqMRIDataset
from atommic.collections.quantitative.parts.transforms import qMRIDataTransforms
from atommic.collections.reconstruction.losses.na import NoiseAwareLoss
from atommic.collections.reconstruction.losses.ssim import SSIMLoss
from atommic.collections.reconstruction.losses.haarpsi import HaarPSILoss
from atommic.collections.reconstruction.metrics.reconstruction_metrics import mse, nmse, psnr, ssim, haarpsi
from atommic.collections.reconstruction.nn.base import DistributedMetricSum

__all__ = ["BaseqMRIReconstructionModel", "SignalForwardModel"]


class BaseqMRIReconstructionModel(atommic_common.nn.base.BaseMRIModel, ABC):
    """Base class of all quantitative MRIReconstruction models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`BaseqMRIReconstructionModel`.

        Parameters
        ----------
        cfg: DictConfig
            The configuration file.
        trainer: Trainer
            The PyTorch Lightning trainer.
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

        # Initialize loss related parameters.
        self.kspace_quantitative_loss = cfg_dict.get("kspace_quantitative_loss", False)
        self.n2r_loss_weight = cfg_dict.get("n2r_loss_weight", 1.0) if self.n2r else 1.0
        self.quantitative_losses = {}
        quantitative_loss = cfg_dict.get("quantitative_loss")
        quantitative_losses_ = {}
        if quantitative_loss is not None:
            for k, v in quantitative_loss.items():
                if k not in VALID_RECONSTRUCTION_LOSSES:
                    raise ValueError(
                        f"Quantitative loss {k} is not supported. Please choose one of the following: "
                        f"{VALID_RECONSTRUCTION_LOSSES}."
                    )
                if v is None or v == 0.0:
                    warnings.warn(f"The weight of quantitative loss {k} is set to 0.0. This loss will not be used.")
                else:
                    quantitative_losses_[k] = v
        else:
            # Default quantitative loss is L1.
            quantitative_losses_["l1"] = 1.0
        if sum(quantitative_losses_.values()) != 1.0:
            warnings.warn("Sum of quantitative losses weights is not 1.0. Adjusting weights to sum up to 1.0.")
            total_weight = sum(quantitative_losses_.values())
            quantitative_losses_ = {k: v / total_weight for k, v in quantitative_losses_.items()}
        for name in VALID_RECONSTRUCTION_LOSSES:
            if name in quantitative_losses_:
                if name == "ssim":
                    if self.ssdu:
                        raise ValueError("SSIM loss is not supported for SSDU.")
                    self.quantitative_losses[name] = SSIMLoss()
                elif name == "haarpsi":
                    if self.ssdu:
                        raise ValueError("HaarPSI loss is not supported for SSDU.")
                    self.quantitative_losses[name] = HaarPSILoss()
                elif name == "mse":
                    self.quantitative_losses[name] = MSELoss()
                elif name == "wasserstein":
                    self.quantitative_losses[name] = SinkhornDistance()
                elif name == "noise_aware":
                    self.quantitative_losses[name] = NoiseAwareLoss()
                elif name == "l1":
                    self.quantitative_losses[name] = L1Loss()
        # replace losses names by 'loss_1', 'loss_2', etc. to properly iterate in the aggregator loss
        self.quantitative_losses = {f"loss_{i+1}": v for i, v in enumerate(self.quantitative_losses.values())}
        self.total_quantitative_losses = len(self.quantitative_losses)
        self.total_quantitative_loss_weight = cfg_dict.get("total_quantitative_loss_weight", 1.0)
        self.total_quantitative_reconstruction_loss_weight = cfg_dict.get(
            "total_quantitative_reconstruction_loss_weight", 1.0
        )
        quantitative_parameters_regularization_factors = cfg_dict.get("quantitative_parameters_regularization_factors")
        self.quantitative_parameters_regularization_factors = {
            "R2star": quantitative_parameters_regularization_factors[0]["R2star"],
            "S0": quantitative_parameters_regularization_factors[1]["S0"],
            "B0": quantitative_parameters_regularization_factors[2]["B0"],
            "phi": quantitative_parameters_regularization_factors[3]["phi"],
        }

        # Set normalization parameters for logging
        self.unnormalize_loss_inputs = cfg_dict.get("unnormalize_loss_inputs", False)
        self.unnormalize_log_outputs = cfg_dict.get("unnormalize_log_outputs", False)
        self.normalization_type = cfg_dict.get("normalization_type", "max")

        # Refers to cascading or iterative reconstruction methods.
        self.accumulate_predictions = cfg_dict.get("accumulate_predictions", False)

        # Refers to the type of the complex-valued data. It can be either "stacked" or "complex_abs" or
        # "complex_sqrt_abs".
        self.complex_valued_type = cfg_dict.get("complex_valued_type", "stacked")

        # Initialize the module
        super().__init__(cfg=cfg, trainer=trainer)

        if self.estimate_coil_sensitivity_maps_with_nn:
            self.coil_sensitivity_maps_nn = atommic_common.nn.base.BaseSensitivityModel(
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

        # Set aggregation loss
        self.total_quantitative_loss = AggregatorLoss(
            num_inputs=self.total_quantitative_losses, weights=list(quantitative_losses_.values())
        )

        self.MSE = DistributedMetricSum()
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.HAARPSI = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()

        # Set evaluation metrics dictionaries
        self.mse_vals_reconstruction: Dict = defaultdict(dict)
        self.nmse_vals_reconstruction: Dict = defaultdict(dict)
        self.ssim_vals_reconstruction: Dict = defaultdict(dict)
        self.psnr_vals_reconstruction: Dict = defaultdict(dict)
        self.haarpsi_vals_reconstruction: Dict = defaultdict(dict)

        self.mse_vals_R2star: Dict = defaultdict(dict)
        self.nmse_vals_R2star: Dict = defaultdict(dict)
        self.ssim_vals_R2star: Dict = defaultdict(dict)
        self.psnr_vals_R2star: Dict = defaultdict(dict)
        self.haarpsi_vals_R2star: Dict = defaultdict(dict)

        self.mse_vals_S0: Dict = defaultdict(dict)
        self.nmse_vals_S0: Dict = defaultdict(dict)
        self.ssim_vals_S0: Dict = defaultdict(dict)
        self.psnr_vals_S0: Dict = defaultdict(dict)
        self.haarpsi_vals_S0: Dict = defaultdict(dict)

        self.mse_vals_B0: Dict = defaultdict(dict)
        self.nmse_vals_B0: Dict = defaultdict(dict)
        self.ssim_vals_B0: Dict = defaultdict(dict)
        self.psnr_vals_B0: Dict = defaultdict(dict)
        self.haarpsi_vals_B0: Dict = defaultdict(dict)

        self.mse_vals_phi: Dict = defaultdict(dict)
        self.nmse_vals_phi: Dict = defaultdict(dict)
        self.ssim_vals_phi: Dict = defaultdict(dict)
        self.psnr_vals_phi: Dict = defaultdict(dict)
        self.haarpsi_vals_phi: Dict = defaultdict(dict)

    def __abs_output__(self, x: torch.Tensor) -> torch.Tensor:
        """Converts the input to absolute value."""
        if x.shape[-1] == 2 or torch.is_complex(x):
            if torch.is_complex(x):
                x = torch.view_as_real(x)
            if self.complex_valued_type == "stacked":
                x = utils.check_stacked_complex(x)
            elif self.complex_valued_type == "utils.complex_abs":
                x = utils.complex_abs(x)
            elif self.complex_valued_type == "complex_sqrt_abs":
                x = utils.complex_abs_sq(x)
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
            target = utils.unnormalize(
                target,
                {
                    "min": attrs["prediction_min"][batch_idx]
                    if "prediction_min" in attrs
                    else attrs[f"prediction_min_{r}"][batch_idx],
                    "max": attrs["prediction_max"][batch_idx]
                    if "prediction_max" in attrs
                    else attrs[f"prediction_max_{r}"][batch_idx],
                    "mean": attrs["prediction_mean"][batch_idx]
                    if "prediction_mean" in attrs
                    else attrs[f"prediction_mean_{r}"][batch_idx],
                    "std": attrs["prediction_std"][batch_idx]
                    if "prediction_std" in attrs
                    else attrs[f"prediction_std_{r}"][batch_idx],
                },
                self.normalization_type,
            )
            prediction = utils.unnormalize(
                prediction,
                {
                    "min": attrs["noise_prediction_min"][batch_idx]
                    if "noise_prediction_min" in attrs
                    else attrs[f"noise_prediction_min_{r}"][batch_idx],
                    "max": attrs["noise_prediction_max"][batch_idx]
                    if "noise_prediction_max" in attrs
                    else attrs[f"noise_prediction_max_{r}"][batch_idx],
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
            target = utils.unnormalize(
                target,
                {
                    "min": attrs["target_min"][batch_idx]
                    if "target_min" in attrs
                    else attrs[f"target_min_{r}"][batch_idx],
                    "max": attrs["target_max"][batch_idx]
                    if "target_max" in attrs
                    else attrs[f"target_max_{r}"][batch_idx],
                    "mean": attrs["target_mean"][batch_idx]
                    if "target_mean" in attrs
                    else attrs[f"target_mean_{r}"][batch_idx],
                    "std": attrs["target_std"][batch_idx]
                    if "target_std" in attrs
                    else attrs[f"target_std_{r}"][batch_idx],
                },
                self.normalization_type,
            )
            prediction = utils.unnormalize(
                prediction,
                {
                    "min": attrs["prediction_min"][batch_idx]
                    if "prediction_min" in attrs
                    else attrs[f"prediction_min_{r}"][batch_idx],
                    "max": attrs["prediction_max"][batch_idx]
                    if "prediction_max" in attrs
                    else attrs[f"prediction_max_{r}"][batch_idx],
                    "mean": attrs["prediction_mean"][batch_idx]
                    if "prediction_mean" in attrs
                    else attrs[f"prediction_mean_{r}"][batch_idx],
                    "std": attrs["prediction_std"][batch_idx]
                    if "prediction_std" in attrs
                    else attrs[f"prediction_std_{r}"][batch_idx],
                },
                self.normalization_type,
            )

        if sensitivity_maps is not None:
            sensitivity_maps = utils.unnormalize(
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

    def __unnormalize_qmaps_for_loss_or_log__(
        self,
        target_R2star_map: torch.Tensor,
        prediction_R2star_map: torch.Tensor,
        target_S0_map: torch.Tensor,
        prediction_S0_map: torch.Tensor,
        target_B0_map: torch.Tensor,
        prediction_B0_map: torch.Tensor,
        target_phi_map: torch.Tensor,
        prediction_phi_map: torch.Tensor,
        attrs: Dict,
        r: int,
        batch_idx: int = 1,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unnormalizes the quantitative maps for computing the loss or logging.

        Parameters
        ----------
        target_R2star_map : torch.Tensor
            Target R2star map of shape [batch_size, n_x, n_y].
        prediction_R2star_map : torch.Tensor
            Prediction R2star map of shape [batch_size, n_x, n_y].
        target_S0_map : torch.Tensor
            Target S0 map of shape [batch_size, n_x, n_y].
        prediction_S0_map : torch.Tensor
            Prediction S0 map of shape [batch_size, n_x, n_y].
        target_B0_map : torch.Tensor
            Target B0 map of shape [batch_size, n_x, n_y].
        prediction_B0_map : torch.Tensor
            Prediction B0 map of shape [batch_size, n_x, n_y].
        target_phi_map : torch.Tensor
            Target phi map of shape [batch_size, n_x, n_y].
        prediction_phi_map : torch.Tensor
            Prediction phi map of shape [batch_size, n_x, n_y].
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        batch_idx : int
            Batch index. Default is ``1``.

        Returns
        -------
        target_R2star_map : torch.Tensor
            Unnormalized target R2star map.
        prediction_R2star_map : torch.Tensor
            Unnormalized prediction R2star map.
        target_S0_map : torch.Tensor
            Unnormalized target S0 map.
        prediction_S0_map : torch.Tensor
            Unnormalized prediction S0 map.
        target_B0_map : torch.Tensor
            Unnormalized target B0 map.
        prediction_B0_map : torch.Tensor
            Unnormalized prediction B0 map.
        target_phi_map : torch.Tensor
            Unnormalized target phi map.
        prediction_phi_map : torch.Tensor
            Unnormalized prediction phi map.
        """
        target_R2star_map = utils.unnormalize(
            target_R2star_map,
            {
                "min": attrs["R2star_map_target_min"][batch_idx]
                if "target_min" in attrs
                else attrs[f"R2star_map_target_min_{r}"][batch_idx],
                "max": attrs["R2star_map_target_max"][batch_idx]
                if "R2star_map_target_max" in attrs
                else attrs[f"R2star_map_target_max_{r}"][batch_idx],
                "mean": attrs["R2star_map_target_mean"][batch_idx]
                if "R2star_map_target_mean" in attrs
                else attrs[f"R2star_map_target_mean_{r}"][batch_idx],
                "std": attrs["R2star_map_target_std"][batch_idx]
                if "R2star_map_target_std" in attrs
                else attrs[f"R2star_map_target_std_{r}"][batch_idx],
                "var": attrs["R2star_map_target_var"][batch_idx]
                if "R2star_map_target_var" in attrs
                else attrs[f"R2star_map_target_var_{r}"][batch_idx],
            },
            self.normalization_type,
        )
        prediction_R2star_map = utils.unnormalize(
            prediction_R2star_map,
            {
                "min": attrs["R2star_map_init_min"][batch_idx]
                if "R2star_map_init_min" in attrs
                else attrs[f"R2star_map_init_min_{r}"][batch_idx],
                "max": attrs["R2star_map_init_max"][batch_idx]
                if "R2star_map_init_max" in attrs
                else attrs[f"R2star_map_init_max_{r}"][batch_idx],
                "mean": attrs["R2star_map_init_mean"][batch_idx]
                if "R2star_map_init_mean" in attrs
                else attrs[f"R2star_map_init_mean_{r}"][batch_idx],
                "std": attrs["R2star_map_init_std"][batch_idx]
                if "R2star_map_init_std" in attrs
                else attrs[f"R2star_map_init_std_{r}"][batch_idx],
                "var": attrs["R2star_map_init_var"][batch_idx]
                if "R2star_map_init_var" in attrs
                else attrs[f"R2star_map_init_var_{r}"][batch_idx],
            },
            self.normalization_type,
        )
        target_S0_map = utils.unnormalize(
            target_S0_map,
            {
                "min": attrs["S0_map_target_min"][batch_idx]
                if "S0_map_target_min" in attrs
                else attrs[f"S0_map_target_min_{r}"][batch_idx],
                "max": attrs["S0_map_target_max"][batch_idx]
                if "S0_map_target_max" in attrs
                else attrs[f"S0_map_target_max_{r}"][batch_idx],
                "mean": attrs["S0_map_target_mean"][batch_idx]
                if "S0_map_target_mean" in attrs
                else attrs[f"S0_map_target_mean_{r}"][batch_idx],
                "std": attrs["S0_map_target_std"][batch_idx]
                if "S0_map_target_std" in attrs
                else attrs[f"S0_map_target_std_{r}"][batch_idx],
                "var": attrs["S0_map_target_var"][batch_idx]
                if "S0_map_target_var" in attrs
                else attrs[f"S0_map_target_var_{r}"][batch_idx],
            },
            self.normalization_type,
        )
        prediction_S0_map = utils.unnormalize(
            prediction_S0_map,
            {
                "min": attrs["S0_map_init_min"][batch_idx]
                if "S0_map_init_min" in attrs
                else attrs[f"S0_map_init_min_{r}"][batch_idx],
                "max": attrs["S0_map_init_max"][batch_idx]
                if "S0_map_init_max" in attrs
                else attrs[f"S0_map_init_max_{r}"][batch_idx],
                "mean": attrs["S0_map_init_mean"][batch_idx]
                if "S0_map_init_mean" in attrs
                else attrs[f"S0_map_init_mean_{r}"][batch_idx],
                "std": attrs["S0_map_init_std"][batch_idx]
                if "S0_map_init_std" in attrs
                else attrs[f"S0_map_init_std_{r}"][batch_idx],
                "var": attrs["S0_map_init_var"][batch_idx]
                if "S0_map_init_var" in attrs
                else attrs[f"S0_map_init_var_{r}"][batch_idx],
            },
            self.normalization_type,
        )
        target_B0_map = utils.unnormalize(
            target_B0_map,
            {
                "min": attrs["B0_map_target_min"][batch_idx]
                if "B0_map_target_min" in attrs
                else attrs[f"B0_map_target_min_{r}"][batch_idx],
                "max": attrs["B0_map_target_max"][batch_idx]
                if "B0_map_target_max" in attrs
                else attrs[f"B0_map_target_max_{r}"][batch_idx],
                "mean": attrs["B0_map_target_mean"][batch_idx]
                if "B0_map_target_mean" in attrs
                else attrs[f"B0_map_target_mean_{r}"][batch_idx],
                "std": attrs["B0_map_target_std"][batch_idx]
                if "B0_map_target_std" in attrs
                else attrs[f"B0_map_target_std_{r}"][batch_idx],
                "var": attrs["B0_map_target_var"][batch_idx]
                if "B0_map_target_var" in attrs
                else attrs[f"B0_map_target_var_{r}"][batch_idx],
            },
            self.normalization_type,
        )
        prediction_B0_map = utils.unnormalize(
            prediction_B0_map,
            {
                "min": attrs["B0_map_init_min"][batch_idx]
                if "B0_map_init_min" in attrs
                else attrs[f"B0_map_init_min_{r}"][batch_idx],
                "max": attrs["B0_map_init_max"][batch_idx]
                if "B0_map_init_max" in attrs
                else attrs[f"B0_map_init_max_{r}"][batch_idx],
                "mean": attrs["B0_map_init_mean"][batch_idx]
                if "B0_map_init_mean" in attrs
                else attrs[f"B0_map_init_mean_{r}"][batch_idx],
                "std": attrs["B0_map_init_std"][batch_idx]
                if "B0_map_init_std" in attrs
                else attrs[f"B0_map_init_std_{r}"][batch_idx],
                "var": attrs["B0_map_init_var"][batch_idx]
                if "B0_map_init_var" in attrs
                else attrs[f"B0_map_init_var_{r}"][batch_idx],
            },
            self.normalization_type,
        )
        target_phi_map = utils.unnormalize(
            target_phi_map,
            {
                "min": attrs["phi_map_target_min"][batch_idx]
                if "phi_map_target_min" in attrs
                else attrs[f"phi_map_target_min_{r}"][batch_idx],
                "max": attrs["phi_map_target_max"][batch_idx]
                if "phi_map_target_max" in attrs
                else attrs[f"phi_map_target_max_{r}"][batch_idx],
                "mean": attrs["phi_map_target_mean"][batch_idx]
                if "phi_map_target_mean" in attrs
                else attrs[f"phi_map_target_mean_{r}"][batch_idx],
                "std": attrs["phi_map_target_std"][batch_idx]
                if "phi_map_target_std" in attrs
                else attrs[f"phi_map_target_std_{r}"][batch_idx],
                "var": attrs["phi_map_target_var"][batch_idx]
                if "phi_map_target_var" in attrs
                else attrs[f"phi_map_target_var_{r}"][batch_idx],
            },
            self.normalization_type,
        )
        prediction_phi_map = utils.unnormalize(
            prediction_phi_map,
            {
                "min": attrs["phi_map_init_min"][batch_idx]
                if "phi_map_init_min" in attrs
                else attrs[f"phi_map_init_min_{r}"][batch_idx],
                "max": attrs["phi_map_init_max"][batch_idx]
                if "phi_map_init_max" in attrs
                else attrs[f"phi_map_init_max_{r}"][batch_idx],
                "mean": attrs["phi_map_init_mean"][batch_idx]
                if "phi_map_init_mean" in attrs
                else attrs[f"phi_map_init_mean_{r}"][batch_idx],
                "std": attrs["phi_map_init_std"][batch_idx]
                if "phi_map_init_std" in attrs
                else attrs[f"phi_map_init_std_{r}"][batch_idx],
                "var": attrs["phi_map_init_var"][batch_idx]
                if "phi_map_init_var" in attrs
                else attrs[f"phi_map_init_var_{r}"][batch_idx],
            },
            self.normalization_type,
        )

        return (
            target_R2star_map,
            prediction_R2star_map,
            target_S0_map,
            prediction_S0_map,
            target_B0_map,
            prediction_B0_map,
            target_phi_map,
            prediction_phi_map,
        )

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
        if isinstance(prediction, list):
            while isinstance(prediction, list):
                prediction = prediction[-1]

        target = torch.abs(self.__abs_output__(target / torch.max(torch.abs(target))))
        prediction = torch.abs(self.__abs_output__(prediction / torch.max(torch.abs(prediction))))
        anatomy_mask = torch.abs(anatomy_mask).to(target)

        if "ssim" in str(loss_func).lower():
            return (
                loss_func(
                    target * anatomy_mask,
                    prediction * anatomy_mask,
                    data_range=torch.tensor(
                        [max(torch.max(target * anatomy_mask).item(), torch.max(prediction * anatomy_mask).item())]
                    )
                    .unsqueeze(dim=0)
                    .to(target),
                )
                * self.quantitative_parameters_regularization_factors[quantitative_map]
            )

        if "haarpsi" in str(loss_func).lower():
            return (
                loss_func(
                    target * anatomy_mask,
                    prediction * anatomy_mask,
                    data_range=torch.tensor(
                        [max(torch.max(target * anatomy_mask).item(), torch.max(prediction * anatomy_mask).item())]
                    )
                    .unsqueeze(dim=0)
                    .to(target),
                )
                * self.quantitative_parameters_regularization_factors[quantitative_map]
            )

        return (
            loss_func(target * anatomy_mask, prediction * anatomy_mask)
            / self.quantitative_parameters_regularization_factors[quantitative_map]
        )

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
            Loss function. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        if isinstance(prediction, list):
            while isinstance(prediction, list):
                prediction = prediction[-1]

        if self.unnormalize_loss_inputs:
            target, prediction, sensitivity_maps = self.__unnormalize_for_loss_or_log__(
                target, prediction, sensitivity_maps, attrs, r
            )

        # If kspace reconstruction loss is used, the target needs to be transformed to k-space.
        if self.kspace_quantitative_loss:
            # If inputs are complex, then they need to be viewed as real.
            if target.shape[-1] != 2 and torch.is_complex(target):
                target = torch.view_as_real(target)
            if prediction.shape[-1] != 2 and torch.is_complex(prediction):
                prediction = torch.view_as_real(prediction)

            # Transform to k-space.
            target = fft.fft2(target, self.fft_centered, self.fft_normalization, self.spatial_dims)
            prediction = fft.fft2(prediction, self.fft_centered, self.fft_normalization, self.spatial_dims)

            target = self.__abs_output__(target / torch.max(torch.abs(target)))
            prediction = self.__abs_output__(prediction / torch.max(torch.abs(prediction)))
        elif not self.unnormalize_loss_inputs:
            target = self.__abs_output__(target / torch.max(torch.abs(target)))
            prediction = self.__abs_output__(prediction / torch.max(torch.abs(prediction)))

        prediction = torch.abs(prediction / torch.max(torch.abs(prediction)))
        target = torch.abs(target / torch.max(torch.abs(target)))

        return torch.mean(
            torch.tensor([loss_func(target[:, echo], prediction[:, echo]) for echo in range(target.shape[1])])
        )

    def __compute_loss__(
        self,
        target_reconstruction: torch.Tensor,
        prediction_reconstruction: Union[list, torch.Tensor],
        R2star_map_prediction: torch.Tensor,
        R2star_map_target: torch.Tensor,
        S0_map_prediction: torch.Tensor,
        S0_map_target: torch.Tensor,
        B0_map_prediction: torch.Tensor,
        B0_map_target: torch.Tensor,
        phi_map_prediction: torch.Tensor,
        phi_map_target: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        anatomy_mask: torch.Tensor,
        attrs: Dict,
        r: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the quantitative loss.

        Parameters
        ----------
        target_reconstruction : torch.Tensor
            Reconstruction target data of shape [batch_size, n_x, n_y, 2].
        prediction_reconstruction : Union[list, torch.Tensor]
            Reconstruction prediction(s) of shape [batch_size, n_x, n_y, 2].
        R2star_map_prediction : torch.Tensor
            R2* map prediction of shape [batch_size, n_x, n_y].
        R2star_map_target : torch.Tensor
            R2* map target of shape [batch_size, n_x, n_y].
        S0_map_prediction : torch.Tensor
            S0 map prediction of shape [batch_size, n_x, n_y].
        S0_map_target : torch.Tensor
            S0 map target of shape [batch_size, n_x, n_y].
        B0_map_prediction : torch.Tensor
            B0 map prediction of shape [batch_size, n_x, n_y].
        B0_map_target : torch.Tensor
            B0 map target of shape [batch_size, n_x, n_y].
        phi_map_prediction : torch.Tensor
            Phi map prediction of shape [batch_size, n_x, n_y].
        phi_map_target : torch.Tensor
            Phi map target of shape [batch_size, n_x, n_y].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        anatomy_mask : torch.Tensor
            Mask of specified anatomy, e.g. brain. Shape [n_x, n_y].
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.

        Returns
        -------
        lossR2star : torch.Tensor
            R2* loss.
        lossS0 : torch.Tensor
            S0 loss.
        lossB0 : torch.Tensor
            B0 loss.
        lossPhi : torch.Tensor
            Phi loss.
        quantitative_loss : torch.Tensor
            Reconstruction loss.
        loss : torch.Tensor
            Total loss.
        """
        if self.unnormalize_loss_inputs:
            (
                R2star_map_target,
                R2star_map_prediction,
                S0_map_target,
                S0_map_prediction,
                B0_map_target,
                B0_map_prediction,
                phi_map_target,
                phi_map_prediction,
            ) = self.__unnormalize_qmaps_for_loss_or_log__(
                R2star_map_target,
                R2star_map_prediction,
                S0_map_target,
                S0_map_prediction,
                B0_map_target,
                B0_map_prediction,
                phi_map_target,
                phi_map_prediction,
                attrs,
                attrs["r"],
                R2star_map_target.shape[0],
            )

        lossesR2star = {}
        lossesS0 = {}
        lossesB0 = {}
        lossesPhi = {}
        for name, loss_func in self.quantitative_losses.items():
            lossesR2star[name] = self.process_quantitative_loss(
                R2star_map_target, R2star_map_prediction, anatomy_mask, "R2star", loss_func
            )
            lossesS0[name] = self.process_quantitative_loss(
                S0_map_target, S0_map_prediction, anatomy_mask, "S0", loss_func
            )
            lossesB0[name] = self.process_quantitative_loss(
                B0_map_target, B0_map_prediction, anatomy_mask, "B0", loss_func
            )
            lossesPhi[name] = self.process_quantitative_loss(
                phi_map_target, phi_map_prediction, anatomy_mask, "phi", loss_func
            )

        lossR2star = self.total_quantitative_loss(**lossesR2star)
        lossS0 = self.total_quantitative_loss(**lossesS0)
        lossB0 = self.total_quantitative_loss(**lossesB0)
        lossPhi = self.total_quantitative_loss(**lossesPhi)
        qmpas_loss = [lossR2star, lossS0, lossB0, lossPhi]

        total_quantitative_loss = sum(qmpas_loss) / len(qmpas_loss) * self.total_quantitative_loss_weight

        # Reconstruction loss, if self.use_reconstruction_module is True, then the loss is accumulated.
        if self.use_reconstruction_module:
            losses = {}
            for name, loss_func in self.quantitative_losses.items():
                losses[name] = self.process_reconstruction_loss(
                    target_reconstruction,
                    prediction_reconstruction,
                    sensitivity_maps,
                    attrs,
                    r,
                    loss_func,
                )
            quantitative_reconstruction_loss = (
                self.total_quantitative_loss(**losses) * self.total_quantitative_reconstruction_loss_weight
            )
        else:
            quantitative_reconstruction_loss = torch.tensor(0.0).to(R2star_map_target)

        total_quantitative_loss = total_quantitative_loss + quantitative_reconstruction_loss

        return lossR2star, lossS0, lossB0, lossPhi, quantitative_reconstruction_loss, total_quantitative_loss

    def __compute_and_log_metrics_and_outputs__(  # pylint: disable=too-many-statements
        self,
        prediction_R2star_map: Union[list, torch.Tensor],
        prediction_S0_map: Union[list, torch.Tensor],
        prediction_B0_map: Union[list, torch.Tensor],
        prediction_phi_map: Union[list, torch.Tensor],
        prediction_reconstruction: Union[list, torch.Tensor],
        target_R2star_map: Union[list, torch.Tensor],
        target_S0_map: Union[list, torch.Tensor],
        target_B0_map: Union[list, torch.Tensor],
        target_phi_map: Union[list, torch.Tensor],
        target_reconstruction: Union[list, torch.Tensor],
        anatomy_mask: torch.Tensor,
        attrs: Dict,
        fname: str,
        slice_idx: int,
        acceleration: float,
    ):
        """Computes the metrics and logs the outputs.

        Parameters
        ----------
        prediction_R2star_map : Union[list, torch.Tensor]
            R2* map prediction(s) of shape [batch_size, n_x, n_y].
        prediction_S0_map : Union[list, torch.Tensor]
            S0 map prediction(s) of shape [batch_size, n_x, n_y].
        prediction_B0_map : Union[list, torch.Tensor]
            B0 map prediction(s) of shape [batch_size, n_x, n_y].
        prediction_phi_map : Union[list, torch.Tensor]
            Phi map prediction(s) of shape [batch_size, n_x, n_y].
        prediction_reconstruction : Union[list, torch.Tensor]
            Reconstruction prediction(s) of shape [batch_size, n_x, n_y, 2].
        target_R2star_map : Union[list, torch.Tensor]
            R2* map target(s) of shape [batch_size, n_x, n_y].
        target_S0_map : Union[list, torch.Tensor]
            S0 map target(s) of shape [batch_size, n_x, n_y].
        target_B0_map : Union[list, torch.Tensor]
            B0 map target(s) of shape [batch_size, n_x, n_y].
        target_phi_map : Union[list, torch.Tensor]
            Phi map target(s) of shape [batch_size, n_x, n_y].
        target_reconstruction : Union[list, torch.Tensor]
            Reconstruction target(s) of shape [batch_size, n_x, n_y, 2].
        anatomy_mask : torch.Tensor
            Mask of specified anatomy, e.g. brain. Shape [n_x, n_y].
        attrs : Dict
            Attributes of the data with pre normalization values.
        fname : str
            File name.
        slice_idx : int
            Slice index.
        acceleration : float
            Acceleration factor.
        """

        # if predictions are lists, e.g. in case of cascades or time steps or both, unpack them and keep the last one
        def unpack_if_list_and_abs(x):
            if isinstance(x, list):
                while isinstance(x, list):
                    x = x[-1]
            x = self.__abs_output__(x)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return x

        # Add dummy dimensions to target and predictions for logging.
        prediction_R2star_map = unpack_if_list_and_abs(prediction_R2star_map) * anatomy_mask
        prediction_S0_map = unpack_if_list_and_abs(prediction_S0_map) * anatomy_mask
        prediction_B0_map = unpack_if_list_and_abs(prediction_B0_map) * anatomy_mask
        prediction_phi_map = unpack_if_list_and_abs(prediction_phi_map) * anatomy_mask
        target_R2star_map = unpack_if_list_and_abs(target_R2star_map) * anatomy_mask
        target_S0_map = unpack_if_list_and_abs(target_S0_map) * anatomy_mask
        target_B0_map = unpack_if_list_and_abs(target_B0_map) * anatomy_mask
        target_phi_map = unpack_if_list_and_abs(target_phi_map) * anatomy_mask
        if self.use_reconstruction_module:
            prediction_reconstruction = unpack_if_list_and_abs(prediction_reconstruction)
            target_reconstruction = unpack_if_list_and_abs(target_reconstruction)

        # Iterate over the batch and log the target and predictions.
        for _batch_idx_ in range(target_R2star_map.shape[0]):
            output_target_R2star_map = target_R2star_map[_batch_idx_]
            output_prediction_R2star_map = prediction_R2star_map[_batch_idx_]
            output_target_S0_map = target_S0_map[_batch_idx_]
            output_prediction_S0_map = prediction_S0_map[_batch_idx_]
            output_target_B0_map = target_B0_map[_batch_idx_]
            output_prediction_B0_map = prediction_B0_map[_batch_idx_]
            output_target_phi_map = target_phi_map[_batch_idx_]
            output_prediction_phi_map = prediction_phi_map[_batch_idx_]
            if self.use_reconstruction_module:
                output_target_reconstruction = target_reconstruction[_batch_idx_]
                output_prediction_reconstruction = prediction_reconstruction[_batch_idx_]

            if self.unnormalize_log_outputs:
                (
                    output_target_R2star_map,
                    output_prediction_R2star_map,
                    output_target_S0_map,
                    output_prediction_S0_map,
                    output_target_B0_map,
                    output_prediction_B0_map,
                    output_target_phi_map,
                    output_prediction_phi_map,
                ) = self.__unnormalize_qmaps_for_loss_or_log__(
                    output_target_R2star_map,
                    output_prediction_R2star_map,
                    output_target_S0_map,
                    output_prediction_S0_map,
                    output_target_B0_map,
                    output_prediction_B0_map,
                    output_target_phi_map,
                    output_prediction_phi_map,
                    attrs,
                    attrs["r"],
                    _batch_idx_,
                )

            output_target_R2star_map = (
                torch.abs(output_target_R2star_map / torch.max(torch.abs(output_target_R2star_map))).detach().cpu()
            )
            output_prediction_R2star_map = (
                torch.abs(output_prediction_R2star_map / torch.max(torch.abs(output_prediction_R2star_map)))
                .detach()
                .cpu()
            )
            output_target_S0_map = (
                torch.abs(output_target_S0_map / torch.max(torch.abs(output_target_S0_map))).detach().cpu()
            )
            output_prediction_S0_map = (
                torch.abs(output_prediction_S0_map / torch.max(torch.abs(output_prediction_S0_map))).detach().cpu()
            )
            output_target_B0_map = (
                torch.abs(output_target_B0_map / torch.max(torch.abs(output_target_B0_map))).detach().cpu()
            )
            output_prediction_B0_map = (
                torch.abs(output_prediction_B0_map / torch.max(torch.abs(output_prediction_B0_map))).detach().cpu()
            )
            output_target_phi_map = (
                torch.abs(output_target_phi_map / torch.max(torch.abs(output_target_phi_map))).detach().cpu()
            )
            output_prediction_phi_map = (
                torch.abs(output_prediction_phi_map / torch.max(torch.abs(output_prediction_phi_map))).detach().cpu()
            )

            if self.use_reconstruction_module:
                output_target_reconstruction = (
                    torch.abs(output_target_reconstruction / torch.max(torch.abs(output_target_reconstruction)))
                    .detach()
                    .cpu()
                )
                output_prediction_reconstruction = (
                    torch.abs(
                        output_prediction_reconstruction / torch.max(torch.abs(output_prediction_reconstruction))
                    )
                    .detach()
                    .cpu()
                )

            slice_num = int(slice_idx[_batch_idx_].item())  # type: ignore

            # Log target and predictions, if log_image is True for this slice.
            if attrs["log_image"][_batch_idx_]:
                # if consecutive slices, select the middle slice
                if self.consecutive_slices > 1:
                    output_target_R2star_map = output_target_R2star_map[self.consecutive_slices // 2]
                    output_prediction_R2star_map = output_prediction_R2star_map[self.consecutive_slices // 2]
                    output_target_S0_map = output_target_S0_map[self.consecutive_slices // 2]
                    output_prediction_S0_map = output_prediction_S0_map[self.consecutive_slices // 2]
                    output_target_B0_map = output_target_B0_map[self.consecutive_slices // 2]
                    output_prediction_B0_map = output_prediction_B0_map[self.consecutive_slices // 2]
                    output_target_phi_map = output_target_phi_map[self.consecutive_slices // 2]
                    output_prediction_phi_map = output_prediction_phi_map[self.consecutive_slices // 2]

                key = f"{fname[_batch_idx_]}_slice_{int(slice_idx[_batch_idx_])}-Acc={acceleration}x"  # type: ignore

                output_target_qmaps = torch.cat(
                    [output_target_R2star_map, output_target_S0_map, output_target_B0_map, output_target_phi_map],
                    dim=-1,
                )
                output_prediction_qmaps = torch.cat(
                    [
                        output_prediction_R2star_map,
                        output_prediction_S0_map,
                        output_prediction_B0_map,
                        output_prediction_phi_map,
                    ],
                    dim=-1,
                )

                self.log_image(f"{key}/qmaps/target", output_target_qmaps)
                self.log_image(f"{key}/qmaps/reconstruction", output_prediction_qmaps)
                self.log_image(f"{key}/qmaps/error", output_target_qmaps - output_prediction_qmaps)

                if self.use_reconstruction_module:
                    output_target_reconstruction_echoes = torch.cat(
                        [output_target_reconstruction[i] for i in range(output_target_reconstruction.shape[0])], dim=-1
                    )
                    output_prediction_reconstruction_echoes = torch.cat(
                        [
                            output_prediction_reconstruction[i]
                            for i in range(output_prediction_reconstruction.shape[0])
                        ],
                        dim=-1,
                    )

                    self.log_image(f"{key}/reconstruction/target", output_target_reconstruction_echoes)
                    self.log_image(f"{key}/reconstruction/prediction", output_prediction_reconstruction_echoes)
                    self.log_image(
                        f"{key}/reconstruction/error",
                        torch.abs(output_target_reconstruction_echoes - output_prediction_reconstruction_echoes),
                    )

            if self.use_reconstruction_module:
                output_target_reconstruction = output_target_reconstruction.unsqueeze(1).numpy()
                output_prediction_reconstruction = output_prediction_reconstruction.unsqueeze(1).numpy()

                # compute metrics per echo time
                mses = []
                nmses = []
                ssims = []
                psnrs = []
                haarpsis = []
                for echo_time in range(output_target_reconstruction.shape[0]):
                    echo_output_target_reconstruction = output_target_reconstruction[echo_time, ...]
                    echo_output_prediction_reconstruction = output_prediction_reconstruction[echo_time, ...]

                    echo_output_target_reconstruction = np.abs(
                        echo_output_target_reconstruction / np.max(np.abs(echo_output_target_reconstruction))
                    )
                    echo_output_prediction_reconstruction = np.abs(
                        echo_output_prediction_reconstruction / np.max(np.abs(echo_output_prediction_reconstruction))
                    )

                    mses.append(
                        torch.tensor(
                            mse(echo_output_target_reconstruction, echo_output_prediction_reconstruction)
                        ).view(1)
                    )
                    nmses.append(
                        torch.tensor(
                            nmse(echo_output_target_reconstruction, echo_output_prediction_reconstruction)
                        ).view(1)
                    )

                    max_value = max(
                        np.max(echo_output_target_reconstruction), np.max(echo_output_prediction_reconstruction)
                    ) - min(np.min(echo_output_target_reconstruction), np.min(echo_output_prediction_reconstruction))

                    ssims.append(
                        torch.tensor(
                            ssim(
                                echo_output_target_reconstruction,
                                echo_output_prediction_reconstruction,
                                max_value,
                            )
                        ).view(1)
                    )
                    psnrs.append(
                        torch.tensor(
                            psnr(
                                echo_output_target_reconstruction,
                                echo_output_prediction_reconstruction,
                                max_value,
                            )
                        ).view(1)
                    )
                    max_value = max(
                        np.max(echo_output_target_reconstruction), np.max(echo_output_prediction_reconstruction)
                    )
                    haarpsis.append(
                        torch.tensor(
                            haarpsi(
                                echo_output_target_reconstruction,
                                echo_output_prediction_reconstruction,
                                max_value,
                            )
                        ).view(1)
                    )

                self.mse_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(mses).mean()
                self.nmse_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(nmses).mean()
                self.ssim_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(ssims).mean()
                self.psnr_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(psnrs).mean()
                self.haarpsi_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(haarpsis).mean()
            else:
                self.mse_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(0).view(1)
                self.nmse_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(0).view(1)
                self.ssim_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(0).view(1)
                self.psnr_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(0).view(1)
                self.haarpsi_vals_reconstruction[fname[_batch_idx_]][str(slice_num)] = torch.tensor(0).view(1)

            # compute metrics for quantitative maps
            output_target_R2star_map = output_target_R2star_map.numpy()
            output_prediction_R2star_map = output_prediction_R2star_map.numpy()
            output_target_S0_map = output_target_S0_map.numpy()
            output_prediction_S0_map = output_prediction_S0_map.numpy()
            output_target_B0_map = output_target_B0_map.numpy()
            output_prediction_B0_map = output_prediction_B0_map.numpy()
            output_target_phi_map = output_target_phi_map.numpy()
            output_prediction_phi_map = output_prediction_phi_map.numpy()

            self.mse_vals_R2star[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                mse(output_target_R2star_map, output_prediction_R2star_map)
            ).view(1)
            self.nmse_vals_R2star[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                nmse(output_target_R2star_map, output_prediction_R2star_map)
            ).view(1)

            max_value = max(np.max(output_target_R2star_map), np.max(output_prediction_R2star_map)) - min(
                np.min(output_target_R2star_map), np.min(output_prediction_R2star_map)
            )

            self.ssim_vals_R2star[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                ssim(output_target_R2star_map, output_prediction_R2star_map, maxval=max_value)
            ).view(1)
            self.psnr_vals_R2star[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                psnr(output_target_R2star_map, output_prediction_R2star_map, maxval=max_value)
            ).view(1)

            max_value = max(np.max(output_target_R2star_map), np.max(output_prediction_R2star_map))

            self.haarpsi_vals_R2star[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                haarpsi(output_target_R2star_map, output_prediction_R2star_map, maxval=max_value)
            ).view(1)

            self.mse_vals_S0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                mse(output_target_S0_map, output_prediction_S0_map)
            ).view(1)
            self.nmse_vals_S0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                nmse(output_target_S0_map, output_prediction_S0_map)
            ).view(1)

            max_value = max(np.max(output_target_S0_map), np.max(output_prediction_S0_map)) - min(
                np.min(output_target_S0_map), np.min(output_prediction_S0_map)
            )

            self.ssim_vals_S0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                ssim(output_target_S0_map, output_prediction_S0_map, maxval=max_value)
            ).view(1)
            self.psnr_vals_S0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                psnr(output_target_S0_map, output_prediction_S0_map, maxval=max_value)
            ).view(1)

            max_value = max(np.max(output_target_S0_map), np.max(output_prediction_S0_map))

            self.haarpsi_vals_S0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                haarpsi(output_target_S0_map, output_prediction_S0_map, maxval=max_value)
            ).view(1)

            self.mse_vals_B0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                mse(output_target_B0_map, output_prediction_B0_map)
            ).view(1)
            self.nmse_vals_B0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                nmse(output_target_B0_map, output_prediction_B0_map)
            ).view(1)

            max_value = max(np.max(output_target_B0_map), np.max(output_prediction_B0_map)) - min(
                np.min(output_target_B0_map), np.min(output_prediction_B0_map)
            )

            self.ssim_vals_B0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                ssim(output_target_B0_map, output_prediction_B0_map, maxval=max_value)
            ).view(1)
            self.psnr_vals_B0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                psnr(output_target_B0_map, output_prediction_B0_map, maxval=max_value)
            ).view(1)

            max_value = max(np.max(output_target_B0_map), np.max(output_prediction_B0_map))

            self.haarpsi_vals_B0[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                haarpsi(output_target_B0_map, output_prediction_B0_map, maxval=max_value)
            ).view(1)

            self.mse_vals_phi[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                mse(output_target_phi_map, output_prediction_phi_map)
            ).view(1)
            self.nmse_vals_phi[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                nmse(output_target_phi_map, output_prediction_phi_map)
            ).view(1)

            max_value = max(np.max(output_target_phi_map), np.max(output_prediction_phi_map)) - min(
                np.min(output_target_phi_map), np.min(output_prediction_phi_map)
            )  # TODO: Should this be added it was missing?
            self.ssim_vals_phi[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                ssim(output_target_phi_map, output_prediction_phi_map, maxval=max_value)
            ).view(1)
            self.psnr_vals_phi[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                psnr(output_target_phi_map, output_prediction_phi_map, maxval=max_value)
            ).view(1)

            max_value = max(np.max(output_target_phi_map), np.max(output_prediction_phi_map))

            self.haarpsi_vals_phi[fname[_batch_idx_]][str(slice_num)] = torch.tensor(
                haarpsi(output_target_phi_map, output_prediction_phi_map, maxval=max_value)
            ).view(1)

    @staticmethod
    def __process_inputs__(
        R2star_map_init: Union[list, torch.Tensor],
        S0_map_init: Union[list, torch.Tensor],
        B0_map_init: Union[list, torch.Tensor],
        phi_map_init: Union[list, torch.Tensor],
        kspace: Union[list, torch.Tensor],
        y: Union[list, torch.Tensor],
        mask: Union[list, torch.Tensor],
        initial_prediction_reconstruction: Union[List, torch.Tensor],
        target_reconstruction: Union[list, torch.Tensor],
    ) -> tuple[
        Union[Union[list, Tensor], Any],
        Union[Union[list, Tensor], Any],
        Union[Union[list, Tensor], Any],
        Union[Union[list, Tensor], Any],
        Union[Tensor, Any],
        Union[Tensor, Any],
        Union[Tensor, Any],
        Union[Tensor, Any],
        Union[Tensor, Any],
        Union[int, ndarray],
    ]:
        """Processes lists of inputs to torch.Tensor. In the case where multiple accelerations are used, then the
        inputs are lists. This function converts the lists to torch.Tensor by randomly selecting one acceleration. If
        only one acceleration is used, then the inputs are torch.Tensor and are returned as is.

        Parameters
        ----------
        R2star_map_init : Union[list, torch.Tensor]
            R2* map of length n_accelerations or shape [batch_size, n_x, n_y].
        S0_map_init : Union[list, torch.Tensor]
            S0 map of length n_accelerations or shape [batch_size, n_x, n_y].
        B0_map_init : Union[list, torch.Tensor]
            B0 map of length n_accelerations or shape [batch_size, n_x, n_y].
        phi_map_init : Union[list, torch.Tensor]
            Phi map of length n_accelerations or shape [batch_size, n_x, n_y].
        kspace : Union[list, torch.Tensor]
            Full k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        y : Union[list, torch.Tensor]
            Subsampled k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : Union[list, torch.Tensor]
            Sampling mask of length n_accelerations or shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction_reconstruction : Union[List, torch.Tensor]
            Initial reconstruction prediction. If multiple accelerations are used, then it is a list of torch.Tensor.
            Shape [batch_size, n_x, n_y, 2].
        target_reconstruction : torch.Tensor
            Target reconstruction data. Shape [batch_size, n_x, n_y, 2].

        Returns
        -------
        R2star_map_init : torch.Tensor
            R2* map of shape [batch_size, n_x, n_y].
        S0_map_init : torch.Tensor
            S0 map of shape [batch_size, n_x, n_y].
        B0_map_init : torch.Tensor
            B0 map of shape [batch_size, n_x, n_y].
        phi_map_init : torch.Tensor
            Phi map of shape [batch_size, n_x, n_y].
        kspace : torch.Tensor
            Full k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        r : int
            Random index used to select the acceleration.
        """
        if isinstance(y, list):
            r = np.random.randint(len(y))
            R2star_map_init = R2star_map_init[r]
            S0_map_init = S0_map_init[r]
            B0_map_init = B0_map_init[r]
            phi_map_init = phi_map_init[r]
            y = y[r]
            mask = mask[r]
            initial_prediction_reconstruction = initial_prediction_reconstruction[r]
        else:
            r = 0
        if isinstance(kspace, list):
            kspace = kspace[r]
            target_reconstruction = target_reconstruction[r]
        elif isinstance(target_reconstruction, list):
            target_reconstruction = target_reconstruction[r]
        return (
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            kspace,
            y,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            r,
        )

    def inference_step(
        self,
        R2star_map_initial_prediction: torch.Tensor,
        S0_map_initial_prediction: torch.Tensor,
        B0_map_initial_prediction: torch.Tensor,
        phi_map_initial_prediction: torch.Tensor,
        TEs: Union[List[torch.Tensor], torch.Tensor],
        kspace: torch.Tensor,
        y: Union[List[torch.Tensor], torch.Tensor],
        sensitivity_maps: torch.Tensor,
        sampling_mask: Union[List[torch.Tensor], torch.Tensor],
        anatomy_mask: Union[List[torch.Tensor], torch.Tensor],
        initial_prediction_reconstruction: Union[List, torch.Tensor],
        target_reconstruction: torch.Tensor,
        fname: str,
        slice_idx: int,
        acceleration: float,
        attrs: Dict,
    ):
        """Performs an inference step, i.e., computes the predictions of the model.

        Parameters
        ----------
        R2star_map_initial_prediction : torch.Tensor
            Initial R2* map prediction. Shape [batch_size, n_x, n_y].
        S0_map_initial_prediction : torch.Tensor
            Initial S0 map prediction. Shape [batch_size, n_x, n_y].
        B0_map_initial_prediction : torch.Tensor
            Initial B0 map prediction. Shape [batch_size, n_x, n_y].
        phi_map_initial_prediction : torch.Tensor
            Initial phi map prediction. Shape [batch_size, n_x, n_y].
        TEs : Union[List[torch.Tensor], torch.Tensor]
            Echo times. If multiple echoes are used, then it is a list of torch.Tensor. Shape [batch_size, n_echoes].
        kspace : torch.Tensor
            Fully sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
        y : Union[List[torch.Tensor], torch.Tensor]
            Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
            Shape [batch_size, n_coils, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
        sampling_mask : Union[List[torch.Tensor], torch.Tensor]
            Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if Unsupervised
            Learning methods are used, it contains their masks. Shape [batch_size, 1, n_x, n_y, 1].
        anatomy_mask : Union[List[torch.Tensor], torch.Tensor]
                    Mask of specified anatomy, e.g. brain. Shape [n_x, n_y].
        initial_prediction_reconstruction : Union[List, torch.Tensor]
            Initial reconstruction prediction. If multiple accelerations are used, then it is a list of torch.Tensor.
            Shape [batch_size, n_x, n_y, 2].
        target_reconstruction : torch.Tensor
            Target reconstruction data. Shape [batch_size, n_x, n_y, 2].
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
            Dictionary of loss and log.
        """
        # Process inputs to randomly select one acceleration factor, in case multiple accelerations are used.
        (
            R2star_map_initial_prediction,
            S0_map_initial_prediction,
            B0_map_initial_prediction,
            phi_map_initial_prediction,
            kspace,
            y,
            sampling_mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            r,
        ) = self.__process_inputs__(
            R2star_map_initial_prediction,
            S0_map_initial_prediction,
            B0_map_initial_prediction,
            phi_map_initial_prediction,
            kspace,
            y,
            sampling_mask,
            initial_prediction_reconstruction,
            target_reconstruction,
        )

        # Check if a network is used for coil sensitivity maps estimation.
        if self.estimate_coil_sensitivity_maps_with_nn:
            # Estimate coil sensitivity maps with a network.
            sensitivity_maps = self.coil_sensitivity_maps_nn(kspace, sampling_mask, sensitivity_maps)
            # (Re-)compute the initial prediction with the estimated sensitivity maps. This also means that the
            # self.coil_combination_method is set to "SENSE", since in "RSS" the sensitivity maps are not used.
            initial_prediction_reconstruction = utils.coil_combination_method(
                fft.ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )

        # Model forward pass
        predictions = self.forward(
            R2star_map_initial_prediction,
            S0_map_initial_prediction,
            B0_map_initial_prediction,
            phi_map_initial_prediction,
            TEs.tolist()[0],  # type: ignore
            y,
            sensitivity_maps,
            initial_prediction_reconstruction,
            torch.ones_like(anatomy_mask),
            sampling_mask,
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

        return {
            "fname": fname,
            "slice_idx": slice_idx,
            "acceleration": acceleration,
            "prediction_reconstruction": predictions[0],
            "prediction_R2star_map": predictions[1],
            "prediction_S0_map": predictions[2],
            "prediction_B0_map": predictions[3],
            "prediction_phi_map": predictions[4],
            "initial_prediction_reconstruction": initial_prediction_reconstruction,
            "target_reconstruction": target_reconstruction,
            "sensitivity_maps": sensitivity_maps,
            "r": r,
        }

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Performs a training step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'R2star_map_init' : List of torch.Tensor
                    R2* initial map. Shape [batch_size, n_x, n_y].
                'R2star_map_target' : torch.Tensor
                    R2* target map. Shape [batch_size, n_x, n_y].
                'S0_map_init' : List of torch.Tensor
                    S0 initial map. Shape [batch_size, n_x, n_y].
                'S0_map_target' : torch.Tensor
                    S0 target map. Shape [batch_size, n_x, n_y].
                'B0_map_init' : List of torch.Tensor
                    B0 initial map. Shape [batch_size, n_x, n_y].
                'B0_map_target' : torch.Tensor
                    B0 target map. Shape [batch_size, n_x, n_y].
                'phi_map_init' : List of torch.Tensor
                    Phi initial map. Shape [batch_size, n_x, n_y].
                'phi_map_target' : torch.Tensor
                    Phi target map. Shape [batch_size, n_x, n_y].
                'TEs' : List of float
                    Echo times. If multiple echoes are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_echoes].
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : List of torch.Tensor
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'anatomy_mask' : torch.Tensor
                    Mask of specified anatomy, e.g. brain. Shape [n_x, n_y].
                'initial_prediction' : Union[torch.Tensor, None]
                    Initial prediction. Shape [batch_size, n_x, n_y, 2] or None.
                'target' : Union[torch.Tensor, None]
                    Target data. Shape [batch_size, n_x, n_y] or None.
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
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            sampling_mask,
            anatomy_mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            fname,
            slice_idx,
            acceleration,
            attrs,
        ) = batch

        outputs = self.inference_step(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            sampling_mask,
            anatomy_mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        # Compute loss
        lossR2star, lossS0, lossB0, lossPhi, quantitative_reconstruction_loss, train_loss = self.__compute_loss__(
            outputs["target_reconstruction"],
            outputs["prediction_reconstruction"],
            outputs["prediction_R2star_map"],
            R2star_map_target,
            outputs["prediction_S0_map"],
            S0_map_target,
            outputs["prediction_B0_map"],
            B0_map_target,
            outputs["prediction_phi_map"],
            phi_map_target,
            outputs["sensitivity_maps"],
            anatomy_mask,
            attrs,  # type: ignore
            outputs["r"],
        )

        acceleration = np.round(outputs['acceleration'])
        tensorboard_logs = {
            f"train_loss_{acceleration}x": train_loss.item(),
            f"loss_reconstruction_{acceleration}x": quantitative_reconstruction_loss.item(),
            f"loss_R2star_{acceleration}x": lossR2star.item(),
            f"loss_S0_{acceleration}x": lossS0.item(),
            f"loss_B0_{acceleration}x": lossB0.item(),
            f"loss_phi_{acceleration}x": lossPhi.item(),
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=R2star_map_target.shape[0],  # type: ignore
            sync_dist=True,
        )
        if self.use_reconstruction_module:
            self.log(
                "quantitative_reconstruction_loss",
                quantitative_reconstruction_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=R2star_map_target.shape[0],  # type: ignore
                sync_dist=True,
            )
        self.log(
            "train_R2star_loss",
            lossR2star,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=R2star_map_target.shape[0],  # type: ignore
            sync_dist=True,
        )
        self.log(
            "train_S0_loss",
            lossS0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=R2star_map_target.shape[0],  # type: ignore
            sync_dist=True,
        )
        self.log(
            "train_B0_loss",
            lossB0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=R2star_map_target.shape[0],  # type: ignore
            sync_dist=True,
        )
        self.log(
            "train_phi_loss",
            lossPhi,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=R2star_map_target.shape[0],  # type: ignore
            sync_dist=True,
        )

        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'R2star_map_init' : List of torch.Tensor
                    R2* initial map. Shape [batch_size, n_x, n_y].
                'R2star_map_target' : torch.Tensor
                    R2* target map. Shape [batch_size, n_x, n_y].
                'S0_map_init' : List of torch.Tensor
                    S0 initial map. Shape [batch_size, n_x, n_y].
                'S0_map_target' : torch.Tensor
                    S0 target map. Shape [batch_size, n_x, n_y].
                'B0_map_init' : List of torch.Tensor
                    B0 initial map. Shape [batch_size, n_x, n_y].
                'B0_map_target' : torch.Tensor
                    B0 target map. Shape [batch_size, n_x, n_y].
                'phi_map_init' : List of torch.Tensor
                    Phi initial map. Shape [batch_size, n_x, n_y].
                'phi_map_target' : torch.Tensor
                    Phi target map. Shape [batch_size, n_x, n_y].
                'TEs' : List of float
                    Echo times. If multiple echoes are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_echoes].
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : List of torch.Tensor
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'anatomy_mask' : torch.Tensor
                    Mask of specified anatomy, e.g. brain. Shape [n_x, n_y].
                'initial_prediction' : Union[torch.Tensor, None]
                    Initial prediction. Shape [batch_size, n_x, n_y, 2] or None.
                'target' : Union[torch.Tensor, None]
                    Target data. Shape [batch_size, n_x, n_y] or None.
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
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            sampling_mask,
            anatomy_mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            fname,
            slice_idx,
            acceleration,
            attrs,
        ) = batch

        outputs = self.inference_step(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            sampling_mask,
            anatomy_mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        target_reconstruction = outputs["target_reconstruction"]
        prediction_reconstruction = outputs["prediction_reconstruction"]
        prediction_R2star_map = outputs["prediction_R2star_map"]
        prediction_S0_map = outputs["prediction_S0_map"]
        prediction_B0_map = outputs["prediction_B0_map"]
        prediction_phi_map = outputs["prediction_phi_map"]
        acceleration = outputs["acceleration"]

        # Compute loss
        _, _, _, _, _, val_loss = self.__compute_loss__(
            target_reconstruction,
            prediction_reconstruction,
            prediction_R2star_map,
            R2star_map_target,
            prediction_S0_map,
            S0_map_target,
            prediction_B0_map,
            B0_map_target,
            prediction_phi_map,
            phi_map_target,
            outputs["sensitivity_maps"],
            anatomy_mask,
            attrs,  # type: ignore
            outputs["r"],
        )
        self.validation_step_outputs.append({"val_loss": val_loss})

        attrs["r"] = outputs["r"]  # type: ignore

        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            prediction_R2star_map,
            prediction_S0_map,
            prediction_B0_map,
            prediction_phi_map,
            prediction_reconstruction,
            R2star_map_target,
            S0_map_target,
            B0_map_target,
            phi_map_target,
            target_reconstruction,
            anatomy_mask,
            attrs,  # type: ignore
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,  # type: ignore
        )

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a test step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'R2star_map_init' : List of torch.Tensor
                    R2* initial map. Shape [batch_size, n_x, n_y].
                'R2star_map_target' : torch.Tensor
                    R2* target map. Shape [batch_size, n_x, n_y].
                'S0_map_init' : List of torch.Tensor
                    S0 initial map. Shape [batch_size, n_x, n_y].
                'S0_map_target' : torch.Tensor
                    S0 target map. Shape [batch_size, n_x, n_y].
                'B0_map_init' : List of torch.Tensor
                    B0 initial map. Shape [batch_size, n_x, n_y].
                'B0_map_target' : torch.Tensor
                    B0 target map. Shape [batch_size, n_x, n_y].
                'phi_map_init' : List of torch.Tensor
                    Phi initial map. Shape [batch_size, n_x, n_y].
                'phi_map_target' : torch.Tensor
                    Phi target map. Shape [batch_size, n_x, n_y].
                'TEs' : List of float
                    Echo times. If multiple echoes are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_echoes].
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : List of torch.Tensor
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'anatomy_mask' : torch.Tensor
                    Mask of specified anatomy, e.g. brain. Shape [n_x, n_y].
                'initial_prediction' : Union[torch.Tensor, None]
                    Initial prediction. Shape [batch_size, n_x, n_y, 2] or None.
                'target' : Union[torch.Tensor, None]
                    Target data. Shape [batch_size, n_x, n_y] or None.
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
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            sampling_mask,
            anatomy_mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            fname,
            slice_idx,
            acceleration,
            attrs,
        ) = batch

        outputs = self.inference_step(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            sampling_mask,
            anatomy_mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        target_reconstruction = outputs["target_reconstruction"]
        prediction_reconstruction = outputs["prediction_reconstruction"]
        prediction_R2star_map = outputs["prediction_R2star_map"]
        prediction_S0_map = outputs["prediction_S0_map"]
        prediction_B0_map = outputs["prediction_B0_map"]
        prediction_phi_map = outputs["prediction_phi_map"]
        acceleration = outputs["acceleration"]

        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            prediction_R2star_map,
            prediction_S0_map,
            prediction_B0_map,
            prediction_phi_map,
            prediction_reconstruction,
            R2star_map_target,
            S0_map_target,
            B0_map_target,
            phi_map_target,
            target_reconstruction,
            anatomy_mask,
            attrs,  # type: ignore
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,  # type: ignore
        )

        if self.accumulate_predictions:
            if self.use_reconstruction_module:
                while isinstance(prediction_reconstruction, list):
                    prediction_reconstruction = prediction_reconstruction[-1]

            while isinstance(prediction_R2star_map, list):
                prediction_R2star_map = prediction_R2star_map[-1]
            while isinstance(prediction_S0_map, list):
                prediction_S0_map = prediction_S0_map[-1]
            while isinstance(prediction_B0_map, list):
                prediction_B0_map = prediction_B0_map[-1]
            while isinstance(prediction_phi_map, list):
                prediction_phi_map = prediction_phi_map[-1]

        if self.use_reconstruction_module:
            # If "16" or "16-mixed" fp is used, ensure complex type will be supported when saving the predictions.
            prediction_reconstruction = (
                torch.view_as_complex(torch.view_as_real(prediction_reconstruction).type(torch.float32))
                .detach()
                .cpu()
                .numpy()
            )

        prediction_qmaps = (
            torch.stack([prediction_R2star_map, prediction_S0_map, prediction_B0_map, prediction_phi_map], dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        predictions = (
            (prediction_qmaps, prediction_reconstruction)
            if self.use_reconstruction_module
            else (prediction_qmaps, prediction_qmaps)
        )

        self.test_step_outputs.append([str(fname[0]), slice_idx, predictions])  # type: ignore

    def on_validation_epoch_end(self):  # noqa: MC0001
        """Called at the end of validation epoch to aggregate outputs.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        self.log("val_loss", torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean())

        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals_R2star = defaultdict(dict)
        nmse_vals_R2star = defaultdict(dict)
        ssim_vals_R2star = defaultdict(dict)
        psnr_vals_R2star = defaultdict(dict)
        haarpsi_vals_R2star = defaultdict(dict)

        mse_vals_S0 = defaultdict(dict)
        nmse_vals_S0 = defaultdict(dict)
        ssim_vals_S0 = defaultdict(dict)
        psnr_vals_S0 = defaultdict(dict)
        haarpsi_vals_S0 = defaultdict(dict)

        mse_vals_B0 = defaultdict(dict)
        nmse_vals_B0 = defaultdict(dict)
        ssim_vals_B0 = defaultdict(dict)
        psnr_vals_B0 = defaultdict(dict)
        haarpsi_vals_B0 = defaultdict(dict)

        mse_vals_phi = defaultdict(dict)
        nmse_vals_phi = defaultdict(dict)
        ssim_vals_phi = defaultdict(dict)
        psnr_vals_phi = defaultdict(dict)
        haarpsi_vals_phi = defaultdict(dict)

        for k, v in self.mse_vals_R2star.items():
            mse_vals_R2star[k].update(v)
        for k, v in self.nmse_vals_R2star.items():
            nmse_vals_R2star[k].update(v)
        for k, v in self.ssim_vals_R2star.items():
            ssim_vals_R2star[k].update(v)
        for k, v in self.psnr_vals_R2star.items():
            psnr_vals_R2star[k].update(v)
        for k, v in self.haarpsi_vals_R2star.items():
            haarpsi_vals_R2star[k].update(v)

        for k, v in self.mse_vals_S0.items():
            mse_vals_S0[k].update(v)
        for k, v in self.nmse_vals_S0.items():
            nmse_vals_S0[k].update(v)
        for k, v in self.ssim_vals_S0.items():
            ssim_vals_S0[k].update(v)
        for k, v in self.psnr_vals_S0.items():
            psnr_vals_S0[k].update(v)
        for k, v in self.haarpsi_vals_S0.items():
            haarpsi_vals_S0[k].update(v)

        for k, v in self.mse_vals_B0.items():
            mse_vals_B0[k].update(v)
        for k, v in self.nmse_vals_B0.items():
            nmse_vals_B0[k].update(v)
        for k, v in self.ssim_vals_B0.items():
            ssim_vals_B0[k].update(v)
        for k, v in self.psnr_vals_B0.items():
            psnr_vals_B0[k].update(v)
        for k, v in self.haarpsi_vals_B0.items():
            haarpsi_vals_B0[k].update(v)

        for k, v in self.mse_vals_phi.items():
            mse_vals_phi[k].update(v)
        for k, v in self.nmse_vals_phi.items():
            nmse_vals_phi[k].update(v)
        for k, v in self.ssim_vals_phi.items():
            ssim_vals_phi[k].update(v)
        for k, v in self.psnr_vals_phi.items():
            psnr_vals_phi[k].update(v)
        for k, v in self.haarpsi_vals_phi.items():
            haarpsi_vals_phi[k].update(v)

        if self.use_reconstruction_module:
            mse_vals_reconstruction = defaultdict(dict)
            nmse_vals_reconstruction = defaultdict(dict)
            ssim_vals_reconstruction = defaultdict(dict)
            psnr_vals_reconstruction = defaultdict(dict)
            haarpsi_vals_reconstruction = defaultdict(dict)

            for k, v in self.mse_vals_reconstruction.items():
                mse_vals_reconstruction[k].update(v)
            for k, v in self.nmse_vals_reconstruction.items():
                nmse_vals_reconstruction[k].update(v)
            for k, v in self.ssim_vals_reconstruction.items():
                ssim_vals_reconstruction[k].update(v)
            for k, v in self.psnr_vals_reconstruction.items():
                psnr_vals_reconstruction[k].update(v)
            for k, v in self.haarpsi_vals_reconstruction.items():
                haarpsi_vals_reconstruction[k].update(v)

        # apply means across image volumes
        metrics = {
            "MSE": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
            "NMSE": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
            "SSIM": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
            "PSNR": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
            "HAARPSI": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
        }
        local_examples = 0
        for fname in mse_vals_R2star:
            local_examples += 1
            metrics["MSE"]["R2star"] = metrics["MSE"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_R2star[fname].items()])
            )
            metrics["NMSE"]["R2star"] = metrics["NMSE"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals_R2star[fname].items()])
            )
            metrics["SSIM"]["R2star"] = metrics["SSIM"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_R2star[fname].items()])
            )
            metrics["PSNR"]["R2star"] = metrics["PSNR"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals_R2star[fname].items()])
            )
            metrics["HAARPSI"]["R2star"] = metrics["HAARPSI"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals_R2star[fname].items()])
            )

            metrics["MSE"]["S0"] = metrics["MSE"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_S0[fname].items()])
            )
            metrics["NMSE"]["S0"] = metrics["NMSE"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals_S0[fname].items()])
            )
            metrics["SSIM"]["S0"] = metrics["SSIM"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_S0[fname].items()])
            )
            metrics["PSNR"]["S0"] = metrics["PSNR"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals_S0[fname].items()])
            )
            metrics["HAARPSI"]["S0"] = metrics["HAARPSI"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals_S0[fname].items()])
            )

            metrics["MSE"]["B0"] = metrics["MSE"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_B0[fname].items()])
            )
            metrics["NMSE"]["B0"] = metrics["NMSE"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals_B0[fname].items()])
            )
            metrics["SSIM"]["B0"] = metrics["SSIM"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_B0[fname].items()])
            )
            metrics["PSNR"]["B0"] = metrics["PSNR"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals_B0[fname].items()])
            )
            metrics["HAARPSI"]["B0"] = metrics["HAARPSI"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals_B0[fname].items()])
            )

            metrics["MSE"]["phi"] = metrics["MSE"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_phi[fname].items()])
            )
            metrics["NMSE"]["phi"] = metrics["NMSE"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals_phi[fname].items()])
            )
            metrics["SSIM"]["phi"] = metrics["SSIM"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_phi[fname].items()])
            )
            metrics["PSNR"]["phi"] = metrics["PSNR"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals_phi[fname].items()])
            )
            metrics["HAARPSI"]["phi"] = metrics["HAARPSI"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals_phi[fname].items()])
            )

            if self.use_reconstruction_module:
                metrics["MSE"]["reconstruction"] = metrics["MSE"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in mse_vals_reconstruction[fname].items()])
                )
                metrics["NMSE"]["reconstruction"] = metrics["NMSE"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in nmse_vals_reconstruction[fname].items()])
                )
                metrics["SSIM"]["reconstruction"] = metrics["SSIM"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in ssim_vals_reconstruction[fname].items()])
                )
                metrics["PSNR"]["reconstruction"] = metrics["PSNR"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in psnr_vals_reconstruction[fname].items()])
                )
                metrics["HAARPSI"]["reconstruction"] = metrics["HAARPSI"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in haarpsi_vals_reconstruction[fname].items()])
                )

        # reduce across ddp via sum
        metrics["MSE"]["R2star"] = self.MSE(metrics["MSE"]["R2star"])
        metrics["NMSE"]["R2star"] = self.NMSE(metrics["NMSE"]["R2star"])
        metrics["SSIM"]["R2star"] = self.SSIM(metrics["SSIM"]["R2star"])
        metrics["PSNR"]["R2star"] = self.PSNR(metrics["PSNR"]["R2star"])
        metrics["HAARPSI"]["R2star"] = self.HAARPSI(metrics["HAARPSI"]["R2star"])

        metrics["MSE"]["S0"] = self.MSE(metrics["MSE"]["S0"])
        metrics["NMSE"]["S0"] = self.NMSE(metrics["NMSE"]["S0"])
        metrics["SSIM"]["S0"] = self.SSIM(metrics["SSIM"]["S0"])
        metrics["PSNR"]["S0"] = self.PSNR(metrics["PSNR"]["S0"])
        metrics["HAARPSI"]["S0"] = self.HAARPSI(metrics["HAARPSI"]["S0"])

        metrics["MSE"]["B0"] = self.MSE(metrics["MSE"]["B0"])
        metrics["NMSE"]["B0"] = self.NMSE(metrics["NMSE"]["B0"])
        metrics["SSIM"]["B0"] = self.SSIM(metrics["SSIM"]["B0"])
        metrics["PSNR"]["B0"] = self.PSNR(metrics["PSNR"]["B0"])
        metrics["HAARPSI"]["B0"] = self.HAARPSI(metrics["HAARPSI"]["B0"])

        metrics["MSE"]["phi"] = self.MSE(metrics["MSE"]["phi"])
        metrics["NMSE"]["phi"] = self.NMSE(metrics["NMSE"]["phi"])
        metrics["SSIM"]["phi"] = self.SSIM(metrics["SSIM"]["phi"])
        metrics["PSNR"]["phi"] = self.PSNR(metrics["PSNR"]["phi"])
        metrics["HAARPSI"]["phi"] = self.HAARPSI(metrics["HAARPSI"]["phi"])

        if self.use_reconstruction_module:
            metrics["MSE"]["reconstruction"] = self.MSE(metrics["MSE"]["reconstruction"])
            metrics["NMSE"]["reconstruction"] = self.NMSE(metrics["NMSE"]["reconstruction"])
            metrics["SSIM"]["reconstruction"] = self.SSIM(metrics["SSIM"]["reconstruction"])
            metrics["PSNR"]["reconstruction"] = self.PSNR(metrics["PSNR"]["reconstruction"])
            metrics["HAARPSI"]["reconstruction"] = self.HAARPSI(metrics["HAARPSI"]["reconstruction"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}_R2star", value["R2star"] / tot_examples, prog_bar=True, sync_dist=True)
            self.log(f"val_metrics/{metric}_S0", value["S0"] / tot_examples, prog_bar=True, sync_dist=True)
            self.log(f"val_metrics/{metric}_B0", value["B0"] / tot_examples, prog_bar=True, sync_dist=True)
            self.log(f"val_metrics/{metric}_phi", value["phi"] / tot_examples, prog_bar=True, sync_dist=True)
            if self.use_reconstruction_module:
                self.log(
                    f"val_metrics/{metric}_Reconstruction",
                    value["reconstruction"] / tot_examples,
                    prog_bar=True,
                    sync_dist=True,
                )

    def on_test_epoch_end(self):  # noqa: MC0001
        """Called at the end of test epoch to aggregate outputs, log metrics and save predictions.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals_R2star = defaultdict(dict)
        nmse_vals_R2star = defaultdict(dict)
        ssim_vals_R2star = defaultdict(dict)
        psnr_vals_R2star = defaultdict(dict)
        haarpsi_vals_R2star = defaultdict(dict)

        mse_vals_S0 = defaultdict(dict)
        nmse_vals_S0 = defaultdict(dict)
        ssim_vals_S0 = defaultdict(dict)
        psnr_vals_S0 = defaultdict(dict)
        haarpsi_vals_S0 = defaultdict(dict)

        mse_vals_B0 = defaultdict(dict)
        nmse_vals_B0 = defaultdict(dict)
        ssim_vals_B0 = defaultdict(dict)
        psnr_vals_B0 = defaultdict(dict)
        haarpsi_vals_B0 = defaultdict(dict)

        mse_vals_phi = defaultdict(dict)
        nmse_vals_phi = defaultdict(dict)
        ssim_vals_phi = defaultdict(dict)
        psnr_vals_phi = defaultdict(dict)
        haarpsi_vals_phi = defaultdict(dict)

        for k, v in self.mse_vals_R2star.items():
            mse_vals_R2star[k].update(v)
        for k, v in self.nmse_vals_R2star.items():
            nmse_vals_R2star[k].update(v)
        for k, v in self.ssim_vals_R2star.items():
            ssim_vals_R2star[k].update(v)
        for k, v in self.psnr_vals_R2star.items():
            psnr_vals_R2star[k].update(v)
        for k, v in self.haarpsi_vals_R2star.items():
            haarpsi_vals_R2star[k].update(v)

        for k, v in self.mse_vals_S0.items():
            mse_vals_S0[k].update(v)
        for k, v in self.nmse_vals_S0.items():
            nmse_vals_S0[k].update(v)
        for k, v in self.ssim_vals_S0.items():
            ssim_vals_S0[k].update(v)
        for k, v in self.psnr_vals_S0.items():
            psnr_vals_S0[k].update(v)
        for k, v in self.haarpsi_vals_S0.items():
            haarpsi_vals_S0[k].update(v)

        for k, v in self.mse_vals_B0.items():
            mse_vals_B0[k].update(v)
        for k, v in self.nmse_vals_B0.items():
            nmse_vals_B0[k].update(v)
        for k, v in self.ssim_vals_B0.items():
            ssim_vals_B0[k].update(v)
        for k, v in self.psnr_vals_B0.items():
            psnr_vals_B0[k].update(v)
        for k, v in self.haarpsi_vals_B0.items():
            haarpsi_vals_B0[k].update(v)

        for k, v in self.mse_vals_phi.items():
            mse_vals_phi[k].update(v)
        for k, v in self.nmse_vals_phi.items():
            nmse_vals_phi[k].update(v)
        for k, v in self.ssim_vals_phi.items():
            ssim_vals_phi[k].update(v)
        for k, v in self.psnr_vals_phi.items():
            psnr_vals_phi[k].update(v)
        for k, v in self.haarpsi_vals_phi.items():
            haarpsi_vals_phi[k].update(v)

        if self.use_reconstruction_module:
            mse_vals_reconstruction = defaultdict(dict)
            nmse_vals_reconstruction = defaultdict(dict)
            ssim_vals_reconstruction = defaultdict(dict)
            psnr_vals_reconstruction = defaultdict(dict)
            haarpsi_vals_reconstruction = defaultdict(dict)

            for k, v in self.mse_vals_reconstruction.items():
                mse_vals_reconstruction[k].update(v)
            for k, v in self.nmse_vals_reconstruction.items():
                nmse_vals_reconstruction[k].update(v)
            for k, v in self.ssim_vals_reconstruction.items():
                ssim_vals_reconstruction[k].update(v)
            for k, v in self.psnr_vals_reconstruction.items():
                psnr_vals_reconstruction[k].update(v)
            for k, v in self.haarpsi_vals_reconstruction.items():
                haarpsi_vals_reconstruction[k].update(v)

        # apply means across image volumes
        metrics = {
            "MSE": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
            "NMSE": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
            "SSIM": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
            "PSNR": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
            "HAARPSI": {"R2star": 0, "S0": 0, "B0": 0, "phi": 0, "reconstruction": 0},
        }
        local_examples = 0
        for fname in mse_vals_R2star:
            local_examples += 1
            metrics["MSE"]["R2star"] = metrics["MSE"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_R2star[fname].items()])
            )
            metrics["NMSE"]["R2star"] = metrics["NMSE"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals_R2star[fname].items()])
            )
            metrics["SSIM"]["R2star"] = metrics["SSIM"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_R2star[fname].items()])
            )
            metrics["PSNR"]["R2star"] = metrics["PSNR"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals_R2star[fname].items()])
            )
            metrics["HAARPSI"]["R2star"] = metrics["HAARPSI"]["R2star"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals_R2star[fname].items()])
            )

            metrics["MSE"]["S0"] = metrics["MSE"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_S0[fname].items()])
            )
            metrics["NMSE"]["S0"] = metrics["NMSE"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals_S0[fname].items()])
            )
            metrics["SSIM"]["S0"] = metrics["SSIM"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_S0[fname].items()])
            )
            metrics["PSNR"]["S0"] = metrics["PSNR"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals_S0[fname].items()])
            )
            metrics["HAARPSI"]["S0"] = metrics["HAARPSI"]["S0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals_S0[fname].items()])
            )

            metrics["MSE"]["B0"] = metrics["MSE"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_B0[fname].items()])
            )
            metrics["NMSE"]["B0"] = metrics["NMSE"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals_B0[fname].items()])
            )
            metrics["SSIM"]["B0"] = metrics["SSIM"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_B0[fname].items()])
            )
            metrics["PSNR"]["B0"] = metrics["PSNR"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals_B0[fname].items()])
            )
            metrics["HAARPSI"]["B0"] = metrics["HAARPSI"]["B0"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals_B0[fname].items()])
            )

            metrics["MSE"]["phi"] = metrics["MSE"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_phi[fname].items()])
            )
            metrics["NMSE"]["phi"] = metrics["NMSE"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals_phi[fname].items()])
            )
            metrics["SSIM"]["phi"] = metrics["SSIM"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_phi[fname].items()])
            )
            metrics["PSNR"]["phi"] = metrics["PSNR"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals_phi[fname].items()])
            )
            metrics["HAARPSI"]["phi"] = metrics["HAARPSI"]["phi"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals_phi[fname].items()])
            )

            if self.use_reconstruction_module:
                metrics["MSE"]["reconstruction"] = metrics["MSE"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in mse_vals_reconstruction[fname].items()])
                )
                metrics["NMSE"]["reconstruction"] = metrics["NMSE"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in nmse_vals_reconstruction[fname].items()])
                )
                metrics["SSIM"]["reconstruction"] = metrics["SSIM"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in ssim_vals_reconstruction[fname].items()])
                )
                metrics["PSNR"]["reconstruction"] = metrics["PSNR"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in psnr_vals_reconstruction[fname].items()])
                )
                metrics["HAARPSI"]["reconstruction"] = metrics["HAARPSI"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in haarpsi_vals_reconstruction[fname].items()])
                )

        # reduce across ddp via sum
        metrics["MSE"]["R2star"] = self.MSE(metrics["MSE"]["R2star"])
        metrics["NMSE"]["R2star"] = self.NMSE(metrics["NMSE"]["R2star"])
        metrics["SSIM"]["R2star"] = self.SSIM(metrics["SSIM"]["R2star"])
        metrics["PSNR"]["R2star"] = self.PSNR(metrics["PSNR"]["R2star"])
        metrics["HAARPSI"]["R2star"] = self.HAARPSI(metrics["HAARPSI"]["R2star"])

        metrics["MSE"]["S0"] = self.MSE(metrics["MSE"]["S0"])
        metrics["NMSE"]["S0"] = self.NMSE(metrics["NMSE"]["S0"])
        metrics["SSIM"]["S0"] = self.SSIM(metrics["SSIM"]["S0"])
        metrics["PSNR"]["S0"] = self.PSNR(metrics["PSNR"]["S0"])
        metrics["HAARPSI"]["S0"] = self.HAARPSI(metrics["HAARPSI"]["S0"])

        metrics["MSE"]["B0"] = self.MSE(metrics["MSE"]["B0"])
        metrics["NMSE"]["B0"] = self.NMSE(metrics["NMSE"]["B0"])
        metrics["SSIM"]["B0"] = self.SSIM(metrics["SSIM"]["B0"])
        metrics["PSNR"]["B0"] = self.PSNR(metrics["PSNR"]["B0"])
        metrics["HAARPSI"]["B0"] = self.HAARPSI(metrics["HAARPSI"]["B0"])

        metrics["MSE"]["phi"] = self.MSE(metrics["MSE"]["phi"])
        metrics["NMSE"]["phi"] = self.NMSE(metrics["NMSE"]["phi"])
        metrics["SSIM"]["phi"] = self.SSIM(metrics["SSIM"]["phi"])
        metrics["PSNR"]["phi"] = self.PSNR(metrics["PSNR"]["phi"])
        metrics["HAARPSI"]["phi"] = self.HAARPSI(metrics["HAARPSI"]["phi"])

        if self.use_reconstruction_module:
            metrics["MSE"]["reconstruction"] = self.MSE(metrics["MSE"]["reconstruction"])
            metrics["NMSE"]["reconstruction"] = self.NMSE(metrics["NMSE"]["reconstruction"])
            metrics["SSIM"]["reconstruction"] = self.SSIM(metrics["SSIM"]["reconstruction"])
            metrics["PSNR"]["reconstruction"] = self.PSNR(metrics["PSNR"]["reconstruction"])
            metrics["HAARPSI"]["reconstruction"] = self.HAARPSI(metrics["HAARPSI"]["reconstruction"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"test_metrics/{metric}_R2star", value["R2star"] / tot_examples, prog_bar=True, sync_dist=True)
            self.log(f"test_metrics/{metric}_S0", value["S0"] / tot_examples, prog_bar=True, sync_dist=True)
            self.log(f"test_metrics/{metric}_B0", value["B0"] / tot_examples, prog_bar=True, sync_dist=True)
            self.log(f"test_metrics/{metric}_phi", value["phi"] / tot_examples, prog_bar=True, sync_dist=True)
            if self.use_reconstruction_module:
                self.log(
                    f"test_metrics/{metric}_Reconstruction",
                    value["reconstruction"] / tot_examples,
                    prog_bar=True,
                    sync_dist=True,
                )

        qmaps = defaultdict(list)
        for fname, slice_num, output in self.test_step_outputs:
            qmaps_pred, _ = output
            qmaps[fname].append((slice_num, qmaps_pred))

        for fname in qmaps:
            qmaps[fname] = np.stack([out for _, out in sorted(qmaps[fname])])

        if self.consecutive_slices > 1:
            # iterate over the slices and always keep the middle slice
            for fname in qmaps:
                qmaps[fname] = qmaps[fname][:, self.consecutive_slices // 2]

        if self.use_reconstruction_module:
            reconstructions = defaultdict(list)
            for fname, slice_num, output in self.test_step_outputs:
                _, reconstructions_pred = output
                reconstructions[fname].append((slice_num, reconstructions_pred))

            for fname in reconstructions:
                reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])

            if self.consecutive_slices > 1:
                # iterate over the slices and always keep the middle slice
                for fname in reconstructions:
                    reconstructions[fname] = reconstructions[fname][:, self.consecutive_slices // 2]
        else:
            reconstructions = None

        if "wandb" in self.logger.__module__.lower():
            out_dir = Path(os.path.join(self.logger.save_dir, "predictions"))
        else:
            out_dir = Path(os.path.join(self.logger.log_dir, "predictions"))
        out_dir.mkdir(exist_ok=True, parents=True)

        if reconstructions is not None:
            for (fname, qmaps_pred), (_, reconstructions_pred) in zip(qmaps.items(), reconstructions.items()):
                with h5py.File(out_dir / fname, "w") as hf:
                    hf.create_dataset("qmaps", data=qmaps_pred)
                    hf.create_dataset("reconstruction", data=reconstructions_pred)
        else:
            for fname, qmaps_pred in qmaps.items():
                with h5py.File(out_dir / fname, "w") as hf:
                    hf.create_dataset("qmaps", data=qmaps_pred)

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

        if utils.is_none(mask_root) and not utils.is_none(mask_type):
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

        dataset_format = cfg.get("dataset_format", None)
        if dataset_format.lower() == "ahead":
            dataloader = AHEADqMRIDataset
        else:
            raise ValueError(
                f"Dataset format {dataset_format} not supported. "
                "At the moment only the AHEAD is supported for quantitative MRI."
            )

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
            transform=qMRIDataTransforms(
                TEs=cfg.get("TEs"),
                precompute_quantitative_maps=cfg.get("precompute_quantitative_maps"),
                qmaps_scaling_factor=cfg.get("qmaps_scaling_factor"),
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
            sequence=cfg.get("sequence", None),
            segmentation_mask_root=cfg.get("segmentation_mask_path", None),
            kspace_scaling_factor=cfg.get("kspace_scaling_factor", 1),
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


class SignalForwardModel:
    """Defines a signal forward model based on sequence."""

    def __init__(self, sequence: Union[str, None] = None):
        """Inits :class:`SignalForwardModel`.

        Parameters
        ----------
        sequence : str
            Sequence name.
        """
        super().__init__()
        self.sequence = sequence.lower() if isinstance(sequence, str) else None
        self.scaling = 1e-3

    def __call__(
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        B0_map: torch.Tensor,
        phi_map: torch.Tensor,
        TEs: Optional[List] = None,
    ):
        """Calls :class:`SignalForwardModel`.

        Parameters
        ----------
        R2star_map : torch.Tensor
            R2* map of shape [batch_size, n_x, n_y].
        S0_map : torch.Tensor
            S0 map of shape [batch_size, n_x, n_y].
        B0_map : torch.Tensor
            B0 map of shape [batch_size, n_x, n_y].
        phi_map : torch.Tensor
            phi map of shape [batch_size, n_x, n_y].
        TEs : list of float, optional
            List of echo times.
        """
        if TEs is None:
            TEs = torch.Tensor([3.0, 11.5, 20.0, 28.5])
        if self.sequence == "megre":
            return self.MEGRESignalModel(R2star_map, S0_map, B0_map, phi_map, TEs)
        if self.sequence == "megre_no_phase":
            return self.MEGRENoPhaseSignalModel(R2star_map, S0_map, TEs)
        raise ValueError(
            "Only MEGRE and MEGRE no phase are supported are signal forward model at the moment. "
            f"Found {self.sequence}"
        )

    def MEGRESignalModel(
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        B0_map: torch.Tensor,
        phi_map: torch.Tensor,
        TEs: Optional[List] = None,
    ):
        """MEGRE forward model.

        Parameters
        ----------
        R2star_map : torch.Tensor
            R2* map of shape [batch_size, n_x, n_y].
        S0_map : torch.Tensor
            S0 map of shape [batch_size, n_x, n_y].
        B0_map : torch.Tensor
            B0 map of shape [batch_size, n_x, n_y].
        phi_map : torch.Tensor
            phi map of shape [batch_size, n_x, n_y].
        TEs : list of float, optional
            List of echo times.
        """
        S0_map_real = S0_map
        S0_map_imag = phi_map

        def first_term(i):
            """First term of the MEGRE signal model."""
            return torch.exp(-TEs[i] * self.scaling * R2star_map)

        def second_term(i):
            """Second term of the MEGRE signal model."""
            return torch.cos(B0_map * self.scaling * -TEs[i])

        def third_term(i):
            """Third term of the MEGRE signal model."""
            return torch.sin(B0_map * self.scaling * -TEs[i])

        pred = torch.stack(
            [
                torch.stack(
                    (
                        S0_map_real * first_term(i) * second_term(i) - S0_map_imag * first_term(i) * third_term(i),
                        S0_map_real * first_term(i) * third_term(i) + S0_map_imag * first_term(i) * second_term(i),
                    ),
                    -1,
                )
                for i in range(len(TEs))  # type: ignore
            ],
            1,
        )
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        return torch.view_as_real(pred[..., 0] + 1j * pred[..., 1])

    def MEGRENoPhaseSignalModel(
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        TEs: Optional[List] = None,
    ):
        """MEGRE no phase forward model.

        Parameters
        ----------
        R2star_map : torch.Tensor
            R2* map of shape [batch_size, n_x, n_y].
        S0_map : torch.Tensor
            S0 map of shape [batch_size, n_x, n_y].
        TEs : list of float, optional
            List of echo times.
        """
        pred = torch.stack(
            [
                torch.stack(
                    (
                        S0_map * torch.exp(-TEs[i] * self.scaling * R2star_map),  # type: ignore
                        S0_map * torch.exp(-TEs[i] * self.scaling * R2star_map),  # type: ignore
                    ),
                    -1,
                )
                for i in range(len(TEs))  # type: ignore
            ],
            1,
        )
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        return torch.view_as_real(pred[..., 0] + 1j * pred[..., 1])
