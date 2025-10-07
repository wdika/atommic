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

from atommic.collections.common.data.subsample import create_masker
from atommic.collections.common.losses import VALID_RECONSTRUCTION_LOSSES, AggregatorLoss, SinkhornDistance
from atommic.collections.common.nn.base import BaseMRIModel, BaseSensitivityModel, DistributedMetricSum
from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import (
    check_stacked_complex,
    coil_combination_method,
    complex_abs,
    complex_abs_sq,
    expand_op,
    is_none,
    parse_list_and_keep_last,
    unnormalize,
)
from atommic.collections.quantitative.data import AHEADqMRIDataset
from atommic.collections.reconstruction.data.mri_reconstruction_loader import (
    CC359ReconstructionMRIDataset,
    ReconstructionMRIDataset,
    SKMTEAReconstructionMRIDataset,
    StanfordKneesReconstructionMRIDataset,
)
from atommic.collections.reconstruction.losses.na import NoiseAwareLoss
from atommic.collections.reconstruction.losses.ssim import SSIMLoss
from atommic.collections.reconstruction.losses.haarpsi import HaarPSILoss
from atommic.collections.reconstruction.metrics import mse, nmse, psnr, ssim, haarpsi
from atommic.collections.reconstruction.parts.transforms import ReconstructionMRIDataTransforms

__all__ = ["BaseMRIReconstructionModel"]


class BaseMRIReconstructionModel(BaseMRIModel, ABC):
    """Base class of all MRI reconstruction models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`BaseMRIReconstructionModel`.

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
        self.num_echoes = cfg_dict.get("num_echoes", 0)

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
                    warnings.warn(f"The weight of reconstruction loss {k} is set to 0.0. This loss will not be used.")
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
                elif name == "haarpsi":
                    self.reconstruction_losses[name] = HaarPSILoss()
        # replace losses names by 'loss_1', 'loss_2', etc. to properly iterate in the aggregator loss
        self.reconstruction_losses = {f"loss_{i+1}": v for i, v in enumerate(self.reconstruction_losses.values())}
        self.total_reconstruction_losses = len(self.reconstruction_losses)
        self.total_reconstruction_loss_weight = cfg_dict.get("total_reconstruction_loss_weight", 1.0)

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
            self.coil_sensitivity_maps_nn = BaseSensitivityModel(
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
        self.total_reconstruction_loss = AggregatorLoss(
            num_inputs=self.total_reconstruction_losses, weights=list(reconstruction_losses_.values())
        )

        # Set distributed metrics
        self.MSE = DistributedMetricSum()
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.HAARPSI = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()

        # Set evaluation metrics dictionaries
        self.mse_vals: Dict = defaultdict(dict)
        self.nmse_vals: Dict = defaultdict(dict)
        self.ssim_vals: Dict = defaultdict(dict)
        self.psnr_vals: Dict = defaultdict(dict)
        self.HAARPSI = DistributedMetricSum()

    def __abs_output__(self, x: torch.Tensor) -> torch.Tensor:
        """Converts the input to absolute value."""
        if isinstance(x, list):
            while isinstance(x, list):
                x = x[-1]
        if x.shape[-1] == 2 or torch.is_complex(x):
            if torch.is_complex(x):
                x = torch.view_as_real(x)
            if self.complex_valued_type == "stacked":
                x = torch.abs(check_stacked_complex(x))
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
        """Unnormalizes the data for computing the loss or logging.

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
            prediction = unnormalize(
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
            min_val = attrs["target_min"] if "target_min" in attrs else attrs[f"target_min_{r}"]
            max_val = attrs["target_max"] if "target_max" in attrs else attrs[f"target_max_{r}"]
            mean_val = attrs["target_mean"] if "target_mean" in attrs else attrs[f"target_mean_{r}"]
            std_val = attrs["target_std"] if "target_std" in attrs else attrs[f"target_std_{r}"]
            if isinstance(min_val, list):
                min_val = min_val[batch_idx]
            if isinstance(max_val, list):
                max_val = max_val[batch_idx]
            if isinstance(mean_val, list):
                mean_val = mean_val[batch_idx]
            if isinstance(std_val, list):
                std_val = std_val[batch_idx]

            target = unnormalize(
                target, {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}, self.normalization_type
            )

            min_val = attrs["prediction_min"] if "prediction_min" in attrs else attrs[f"prediction_min_{r}"]
            max_val = attrs["prediction_max"] if "prediction_max" in attrs else attrs[f"prediction_max_{r}"]
            mean_val = attrs["prediction_mean"] if "prediction_mean" in attrs else attrs[f"prediction_mean_{r}"]
            std_val = attrs["prediction_std"] if "prediction_std" in attrs else attrs[f"prediction_std_{r}"]
            if isinstance(min_val, list):
                min_val = min_val[batch_idx]
            if isinstance(max_val, list):
                max_val = max_val[batch_idx]
            if isinstance(mean_val, list):
                mean_val = mean_val[batch_idx]
            if isinstance(std_val, list):
                std_val = std_val[batch_idx]

            prediction = unnormalize(
                prediction, {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}, self.normalization_type
            )

        if sensitivity_maps is not None and "sensitivity_maps_min" in attrs:
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
            target = self.__abs_output__(target)
        elif not self.unnormalize_loss_inputs:
            # Ensure loss inputs are both viewed in the same way.
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

            if "haarpsi" in str(loss_func).lower():
                p = torch.abs(p / torch.max(torch.abs(p)))
                t = torch.abs(t / torch.max(torch.abs(t)))

                return loss_func(
                    p,
                    t,
                    data_range=torch.tensor([max(torch.max(t).item(), torch.max(p).item())]).unsqueeze(dim=0).to(t),
                )

            return loss_func(t, p)

        if self.num_echoes > 0:
            return torch.mean(
                torch.stack(
                    [
                        compute_reconstruction_loss(
                            target[echo].unsqueeze(0), prediction[echo].unsqueeze(0), sensitivity_maps
                        )
                        for echo in range(target.shape[0])
                    ]
                )
            ).to(target)

        return compute_reconstruction_loss(target, prediction, sensitivity_maps)

    def __compute_loss__(
        self,
        target: torch.Tensor,
        predictions: Union[list, torch.Tensor],
        predictions_n2r: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        ssdu_loss_mask: torch.Tensor,
        attrs: Union[Dict, torch.Tensor],
        r: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the reconstruction loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        predictions : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        predictions_n2r : Union[list, torch.Tensor]
            Noise-to-Recon prediction(s) of shape [batch_size, n_x, n_y, 2], if Noise-to-Recon is used.
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        ssdu_loss_mask : torch.Tensor
            SSDU loss mask of shape [batch_size, 1, n_x, n_y, 1]. It will be used if self.ssdu is True, to enforce
            data consistency on the prediction.
        attrs : Union[Dict, torch.Tensor]
            Attributes of the data with pre normalization values.
        r : Union[int, torch.Tensor]
            The selected acceleration factor.

        Returns
        -------
        loss: torch.FloatTensor
            Reconstruction loss.
        """
        if predictions_n2r is not None and not attrs["n2r_supervised"]:
            # Noise-to-Recon with/without SSDU
            target = predictions
            predictions = predictions_n2r
            weight = self.n2r_loss_weight
        else:
            # Supervised learning or Noise-to-Recon with SSDU
            weight = 1.0
        losses = {}
        for name, loss_func in self.reconstruction_losses.items():
            losses[name] = (
                self.process_reconstruction_loss(
                    target, predictions, sensitivity_maps, ssdu_loss_mask, attrs, r, loss_func=loss_func
                )
                * weight
            )
        return self.total_reconstruction_loss(**losses) * self.total_reconstruction_loss_weight

    def __compute_and_log_metrics_and_outputs__(
        self,
        target: torch.Tensor,
        predictions: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        attrs: Union[Dict, torch.Tensor],
        r: Union[int, torch.Tensor],
        fname: Union[str, torch.Tensor],
        slice_idx: Union[int, torch.Tensor],
        acceleration: Union[float, torch.Tensor],
    ):
        """Computes the metrics and logs the outputs.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y].
        predictions : Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]
            Prediction data of shape [batch_size, n_x, n_y, 2]. It can be a list or list of lists if iterative and/or
            cascading reconstruction methods are used.
        attrs : Union[Dict, torch.Tensor]
            Attributes of the data with pre normalization values.
        r : Union[int, torch.Tensor]
            The selected acceleration factor.
        fname : Union[str, torch.Tensor]
            File name.
        slice_idx : Union[int, torch.Tensor]
            Slice index.
        acceleration : Union[float, torch.Tensor]
            Acceleration factor.
        """
        while isinstance(predictions, list):
            predictions = predictions[-1]

        # Ensure loss inputs are both viewed in the same way.
        target = self.__abs_output__(target)
        predictions = self.__abs_output__(predictions)

        # Check if multiple echoes are used.
        if self.num_echoes > 1:
            # find the batch size
            batch_size = target.shape[0] / self.num_echoes
            # reshape to [batch_size, num_echoes, n_x, n_y]
            target = target.reshape((int(batch_size), self.num_echoes, *target.shape[1:]))
            predictions = predictions.reshape((int(batch_size), self.num_echoes, *predictions.shape[1:]))
            # concatenate the echoes in the last dim
            target = torch.cat([target[:, i, ...] for i in range(self.num_echoes)], dim=-1)
            predictions = torch.cat([predictions[:, i, ...] for i in range(self.num_echoes)], dim=-1)

        # Add dummy dimensions to target and predictions for logging.
        target = target.unsqueeze(1)
        predictions = predictions.unsqueeze(1)

        # Iterate over the batch and log the target and predictions.
        for _batch_idx_ in range(target.shape[0]):
            output_target = target[_batch_idx_]
            output_predictions = predictions[_batch_idx_]

            if self.unnormalize_log_outputs:
                # Unnormalize target and predictions with pre normalization values. This is only for logging purposes.
                # For the loss computation, the self.unnormalize_loss_inputs flag is used.
                output_target, output_predictions, _ = self.__unnormalize_for_loss_or_log__(
                    output_target, output_predictions, None, attrs, r, _batch_idx_
                )

            # Normalize target and predictions to [0, 1] for logging.
            if torch.is_complex(output_target) and output_target.shape[-1] != 2:
                output_target = torch.view_as_real(output_target)
            if output_target.shape[-1] == 2:
                output_target = torch.view_as_complex(output_target)
            output_target = torch.abs(output_target / torch.max(torch.abs(output_target))).detach().cpu()

            if torch.is_complex(output_predictions) and output_predictions.shape[-1] != 2:
                output_predictions = torch.view_as_real(output_predictions)
            if output_predictions.shape[-1] == 2:
                output_predictions = torch.view_as_complex(output_predictions)
            output_predictions = (
                torch.abs(output_predictions / torch.max(torch.abs(output_predictions))).detach().cpu()
            )

            # Log target and predictions, if log_image is True for this slice.
            if attrs["log_image"][_batch_idx_]:
                # if consecutive slices, select the middle slice
                if self.consecutive_slices > 1:
                    output_target = output_target[self.consecutive_slices // 2]
                    output_predictions = output_predictions[self.consecutive_slices // 2]

                key = f"{fname[_batch_idx_]}_slice_{int(slice_idx[_batch_idx_])}-Acc={acceleration}x"  # type: ignore
                self.log_image(f"{key}/target", output_target)
                self.log_image(f"{key}/reconstruction", output_predictions)
                self.log_image(f"{key}/error", torch.abs(output_target - output_predictions))

            # Compute metrics and log them.
            output_target = output_target.numpy()
            output_predictions = output_predictions.numpy()
            self.mse_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                mse(output_target, output_predictions)
            ).view(1)
            self.nmse_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                nmse(output_target, output_predictions)
            ).view(1)

            max_value = max(np.max(output_target), np.max(output_predictions)) - min(
                np.min(output_target), np.min(output_predictions)
            )
            max_value = max(max_value, attrs['max']) if 'max' in attrs else max_value

            self.ssim_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                ssim(output_target, output_predictions, maxval=max_value)
            ).view(1)
            self.psnr_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                psnr(output_target, output_predictions, maxval=max_value)
            ).view(1)
            max_value = max(np.max(output_target), np.max(output_predictions))
            self.haarpsi_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                haarpsi(output_target, output_predictions, maxval=max_value)
            ).view(1)

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
        elif self.ssdu and not self.n2r and len(mask) == 2:  # SSDU without Noise-to-Recon.
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
            Full k-space data of length n_accelerations or shape [batch_size, n_coils, n_x, n_y, 2].
        y : Union[List, torch.Tensor]
            Subsampled k-space data of length n_accelerations or shape [batch_size, n_coils, n_x, n_y, 2].
        mask : Union[List, torch.Tensor]
            Sampling mask of length n_accelerations or shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction : Union[List, torch.Tensor]
            Initial prediction of length n_accelerations or shape [batch_size, n_coils, n_x, n_y, 2].
        target : Union[List, torch.Tensor]
            Target data of length n_accelerations or shape [batch_size, n_x, n_y, 2].

        Returns
        -------
        kspace : torch.Tensor
            Full k-space data of shape [batch_size, n_coils, n_x, n_y, 2].
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction : torch.Tensor
            Initial prediction of shape [batch_size, n_coils, n_x, n_y, 2].
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

    def inference_step(
        self,
        kspace: torch.Tensor,
        y: Union[List[torch.Tensor], torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: Union[List[torch.Tensor], torch.Tensor],
        initial_prediction: Union[List, torch.Tensor],
        target: torch.Tensor,
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
        initial_prediction : Union[List, torch.Tensor]
            Initial prediction. If multiple accelerations are used, then it is a list of torch.Tensor.
            Shape [batch_size, n_x, n_y, 2].
        target : torch.Tensor
            Target data. Shape [batch_size, n_x, n_y, 2].
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
                'predictions' : Union[List[torch.Tensor], torch.Tensor]
                    Model's predictions. If accumulate predictions is True, then it is a list of torch.Tensor.
                    Shape [batch_size, n_x, n_y, 2].
                'predictions_n2r' : Union[List[torch.Tensor], torch.Tensor]
                    Model's predictions for Noise-to-Recon, if Noise-to-Recon is used. If accumulate predictions is
                    True, then it is a list of torch.Tensor. Shape [batch_size, n_x, n_y, 2].
                'target' : torch.Tensor
                    Target data. Shape [batch_size, n_x, n_y, 2].
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
        y, mask, initial_prediction, n2r_y, n2r_mask, n2r_initial_prediction = self.__check_noise_to_recon_inputs__(
            y, mask, initial_prediction, attrs
        )

        # Process inputs to randomly select one acceleration factor, in case multiple accelerations are used.
        kspace, y, mask, initial_prediction, target, r = self.__process_inputs__(
            kspace, y, mask, initial_prediction, target
        )

        # Process inputs if Noise-to-Recon and/or SSDU are used.
        n2r_y, n2r_mask, n2r_initial_prediction, mask, loss_mask = self.__process_unsupervised_inputs__(
            n2r_y, mask, n2r_mask, n2r_initial_prediction, attrs, r
        )

        # Check if multiple echoes are used.
        if self.num_echoes > 1:
            # stack the echoes along the batch dimension
            kspace = kspace.view(-1, *kspace.shape[2:])
            y = y.view(-1, *y.shape[2:])
            mask = mask.view(-1, *mask.shape[2:])
            initial_prediction = initial_prediction.view(-1, *initial_prediction.shape[2:])
            target = target.view(-1, *target.shape[2:])
            sensitivity_maps = torch.repeat_interleave(sensitivity_maps, repeats=kspace.shape[0], dim=0)

        # Check if a network is used for coil sensitivity maps estimation.
        if self.estimate_coil_sensitivity_maps_with_nn:
            # Estimate coil sensitivity maps with a network.
            sensitivity_maps = self.coil_sensitivity_maps_nn(kspace, mask, sensitivity_maps)
            # (Re-)compute the initial prediction with the estimated sensitivity maps. This also means that the
            # self.coil_combination_method is set to "SENSE", since in "RSS" the sensitivity maps are not used.
            initial_prediction = coil_combination_method(
                ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )
            if n2r_initial_prediction is not None:
                n2r_initial_prediction = coil_combination_method(
                    ifft2(n2r_y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                    sensitivity_maps,
                    self.coil_combination_method,
                    self.coil_dim,
                )

        # Forward pass
        predictions = self.forward(y, sensitivity_maps, mask, initial_prediction, attrs["noise"])

        # Noise-to-Recon forward pass, if Noise-to-Recon is used.
        predictions_n2r = None
        if self.n2r and n2r_mask is not None:
            predictions_n2r = self.forward(n2r_y, sensitivity_maps, n2r_mask, n2r_initial_prediction, attrs["noise"])

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
            "predictions": predictions,
            "predictions_n2r": predictions_n2r,
            "target": target,
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
        kspace, y, sensitivity_maps, mask, initial_prediction, target, fname, slice_idx, acceleration, attrs = batch

        outputs = self.inference_step(
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction,
            target,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        target = outputs["target"]

        # Compute loss
        train_loss = self.__compute_loss__(
            target,
            outputs["predictions"],
            outputs["predictions_n2r"],
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

        # In case of Noise-to-Recon or SSDU, the target is a list.
        if isinstance(target, list):
            while isinstance(target, list):
                target = target[-1]

        # Log train loss.
        self.log(
            "reconstruction_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=target.shape[0],  # type: ignore
            sync_dist=True,
        )

        return {"loss": train_loss, "log": logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys:
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : Union[torch.Tensor, None]
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor.. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
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
        kspace, y, sensitivity_maps, mask, initial_prediction, target, fname, slice_idx, acceleration, attrs = batch

        outputs = self.inference_step(
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction,
            target,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        fname = outputs["fname"]
        slice_idx = outputs["slice_idx"]
        acceleration = outputs["acceleration"]
        target = outputs["target"]
        predictions = outputs["predictions"]
        attrs = outputs["attrs"]
        r = outputs["r"]

        # Compute loss
        val_loss = self.__compute_loss__(
            target,
            predictions,
            outputs["predictions_n2r"],
            outputs["sensitivity_maps"],
            outputs["loss_mask"],
            attrs,
            r,
        )
        self.validation_step_outputs.append({"val_loss": val_loss})

        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            target,
            predictions,
            attrs,
            r,
            fname,
            slice_idx,
            acceleration,
        )

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a test step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : Union[torch.Tensor, None]
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor.. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
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
        kspace, y, sensitivity_maps, mask, initial_prediction, target, fname, slice_idx, acceleration, attrs = batch

        outputs = self.inference_step(
            kspace,
            y,
            sensitivity_maps,
            mask,
            initial_prediction,
            target,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            acceleration,
            attrs,  # type: ignore
        )

        fname = outputs["fname"]
        slice_idx = outputs["slice_idx"]
        acceleration = outputs["acceleration"]
        target = outputs["target"]
        predictions = outputs["predictions"]
        attrs = outputs["attrs"]
        r = outputs["r"]

        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            target,
            predictions,
            attrs,
            r,
            fname,
            slice_idx,
            acceleration,
        )

        if self.accumulate_predictions:
            predictions = parse_list_and_keep_last(predictions)

        # If "16" or "16-mixed" fp is used, ensure complex type will be supported when saving the predictions.
        if predictions.shape[-1] == 2:
            predictions = torch.view_as_complex(predictions.type(torch.float32))
        else:
            predictions = torch.view_as_complex(torch.view_as_real(predictions).type(torch.float32))
        predictions = predictions.detach().cpu().numpy()

        self.test_step_outputs.append([fname, slice_idx, predictions])

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to aggregate outputs."""
        self.log("val_loss", torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean(), sync_dist=True)

        # Initialize metrics.
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        haarpsi_vals = defaultdict(dict)

        for k, v in self.mse_vals.items():
            mse_vals[k].update(v)
        for k, v in self.nmse_vals.items():
            nmse_vals[k].update(v)
        for k, v in self.ssim_vals.items():
            ssim_vals[k].update(v)
        for k, v in self.psnr_vals.items():
            psnr_vals[k].update(v)
        for k, v in self.haarpsi_vals.items():
            haarpsi_vals[k].update(v)

        # Parse metrics and log them.
        metrics = {
            "MSE": 0,
            "NMSE": 0,
            "SSIM": 0,
            "PSNR": 0,
            "HaarPSI": 0,
        }
        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )
            metrics["HaarPSI"] = metrics["HaarPSI"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])
        metrics["HaarPSI"] = self.HAARPSI(metrics["HaarPSI"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        """Called at the end of test epoch to aggregate outputs, log metrics and save predictions."""
        # Initialize metrics.
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        haarpsi_vals = defaultdict(dict)

        for k, v in self.mse_vals.items():
            mse_vals[k].update(v)
        for k, v in self.nmse_vals.items():
            nmse_vals[k].update(v)
        for k, v in self.ssim_vals.items():
            ssim_vals[k].update(v)
        for k, v in self.psnr_vals.items():
            psnr_vals[k].update(v)
        for k, v in self.haarpsi_vals.items():
            haarpsi_vals[k].update(v)

        # apply means across image volumes
        metrics = {
            "MSE": 0,
            "NMSE": 0,
            "SSIM": 0,
            "PSNR": 0,
            "HaarPSI": 0,
        }

        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )
            metrics["HaarPSI"] = metrics["HaarPSI"] + torch.mean(
                torch.cat([v.view(-1) for _, v in haarpsi_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])
        metrics["HaarPSI"] = self.PSNR(metrics["HaarPSI"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"test_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)

        # Save predictions.
        reconstructions = defaultdict(list)
        for fname, slice_num, output in self.test_step_outputs:
            reconstructions[fname].append((slice_num, output))

        for fname in reconstructions:
            reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])

        if self.consecutive_slices > 1:
            # iterate over the slices and always keep the middle slice
            for fname in reconstructions:
                reconstructions[fname] = reconstructions[fname][:, self.consecutive_slices // 2]

        if "wandb" in self.logger.__module__.lower():
            out_dir = Path(os.path.join(self.logger.save_dir, "reconstructions"))
        else:
            out_dir = Path(os.path.join(self.logger.log_dir, "reconstructions"))
        out_dir.mkdir(exist_ok=True, parents=True)

        for fname, recons in reconstructions.items():
            with h5py.File(out_dir / fname[0], "w") as hf:
                hf.create_dataset("reconstruction", data=recons)

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
        # Get mask parameters.
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

        dataset_format = cfg.get("dataset_format", None)
        if dataset_format.lower() == "cc359":
            dataloader = CC359ReconstructionMRIDataset
        elif dataset_format.lower() == "stanford_knees":
            dataloader = StanfordKneesReconstructionMRIDataset
        elif dataset_format.lower() in (
            "skm-tea-echo1",
            "skm-tea-echo2",
            "skm-tea-echo1+echo2",
            "skm-tea-echo1+echo2-mc",
        ):
            dataloader = SKMTEAReconstructionMRIDataset
        elif dataset_format.lower() == "ahead":
            dataloader = AHEADqMRIDataset
        else:
            dataloader = ReconstructionMRIDataset

        # Get dataset.
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
            transform=ReconstructionMRIDataTransforms(
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
                mask_func=mask_func,  # type: ignore
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
