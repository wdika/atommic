# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from skimage.restoration import unwrap_phase
from torch.nn import functional as F

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.transforms import (
    N2R,
    SSDU,
    Composer,
    Cropper,
    EstimateCoilSensitivityMaps,
    GeometricDecompositionCoilCompression,
    Masker,
    NoisePreWhitening,
    Normalizer,
    ZeroFillingPadding,
)
from atommic.collections.common.parts.utils import add_coil_dim_if_singlecoil
from atommic.collections.common.parts.utils import coil_combination_method as coil_combination_method_func
from atommic.collections.common.parts.utils import is_none, sense, to_tensor
from atommic.collections.motioncorrection.parts.motionsimulation import MotionSimulation

__all__ = ["qMRIDataTransforms"]


class qMRIDataTransforms:
    """Data transforms for quantitative MRI.

    Returns
    -------
    qMRIDataTransforms
        Preprocessed data for quantitative MRI.
    """

    def __init__(
        self,
        TEs: Optional[List[float]],
        precompute_quantitative_maps: bool = True,
        qmaps_scaling_factor: float = 1.0,
        dataset_format: str = None,
        apply_prewhitening: bool = False,
        find_patch_size: bool = True,
        prewhitening_scale_factor: float = 1.0,
        prewhitening_patch_start: int = 10,
        prewhitening_patch_length: int = 30,
        apply_gcc: bool = False,
        gcc_virtual_coils: int = 10,
        gcc_calib_lines: int = 24,
        gcc_align_data: bool = True,
        apply_random_motion: bool = False,
        random_motion_type: str = "gaussian",
        random_motion_percentage: Sequence[int] = (10, 10),
        random_motion_angle: int = 10,
        random_motion_translation: int = 10,
        random_motion_center_percentage: float = 0.02,
        random_motion_num_segments: int = 8,
        random_motion_random_num_segments: bool = True,
        random_motion_non_uniform: bool = False,
        estimate_coil_sensitivity_maps: bool = False,
        coil_sensitivity_maps_type: str = "ESPIRiT",
        coil_sensitivity_maps_gaussian_sigma: float = 0.0,
        coil_sensitivity_maps_espirit_threshold: float = 0.05,
        coil_sensitivity_maps_espirit_kernel_size: int = 6,
        coil_sensitivity_maps_espirit_crop: float = 0.95,
        coil_sensitivity_maps_espirit_max_iters: int = 30,
        coil_combination_method: str = "SENSE",
        dimensionality: int = 2,
        mask_func: Optional[List] = None,
        shift_mask: bool = False,
        mask_center_scale: Optional[float] = 0.02,
        partial_fourier_percentage: float = 0.0,
        remask: bool = False,
        ssdu: bool = False,
        ssdu_mask_type: str = "Gaussian",
        ssdu_rho: float = 0.4,
        ssdu_acs_block_size: Sequence[int] = (4, 4),
        ssdu_gaussian_std_scaling_factor: float = 4.0,
        ssdu_outer_kspace_fraction: float = 0.0,
        ssdu_export_and_reuse_masks: bool = False,
        n2r: bool = False,
        n2r_supervised_rate: float = 0.0,
        n2r_probability: float = 0.0,
        n2r_std_devs: Tuple[float, float] = None,
        n2r_rhos: Tuple[float, float] = None,
        n2r_use_mask: bool = False,
        unsupervised_masked_target: bool = False,
        crop_size: Optional[Tuple[int, int]] = None,
        kspace_crop: bool = False,
        crop_before_masking: bool = True,
        kspace_zero_filling_size: Optional[Tuple] = None,
        normalize_inputs: bool = True,
        normalization_type: str = "max",
        kspace_normalization: bool = False,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = None,
        coil_dim: int = 0,
        consecutive_slices: int = 1,  # pylint: disable=unused-argument
        use_seed: bool = True,
    ):
        """Inits :class:`qMRIDataTransforms`.

        Parameters
        ----------
        TEs : Optional[List[float]]
            Echo times.
        precompute_quantitative_maps : bool, optional
            Precompute quantitative maps. Default is ``True``.
        qmaps_scaling_factor : float, optional
            Quantitative maps scaling factor. Default is ``1e-3``.
        dataset_format : str, optional
            The format of the dataset. For example, ``'custom_dataset'`` or ``'public_dataset_name'``.
            Default is ``None``.
        apply_prewhitening : bool, optional
            Apply prewhitening. If ``True`` then the prewhitening arguments are used. Default is ``False``.
        find_patch_size : bool, optional
            Find optimal patch size (automatically) to calculate psi. If False, patch_size must be defined.
            Default is ``True``.
        prewhitening_scale_factor : float, optional
            Prewhitening scale factor. Default is ``1.0``.
        prewhitening_patch_start : int, optional
            Prewhitening patch start. Default is ``10``.
        prewhitening_patch_length : int, optional
            Prewhitening patch length. Default is ``30``.
        apply_gcc : bool, optional
            Apply Geometric Decomposition Coil Compression. If ``True`` then the GCC arguments are used.
            Default is ``False``.
        gcc_virtual_coils : int, optional
            GCC virtual coils. Default is ``10``.
        gcc_calib_lines : int, optional
            GCC calibration lines. Default is ``24``.
        gcc_align_data : bool, optional
            GCC align data. Default is ``True``.
        apply_random_motion : bool, optional
            Simulate random motion in k-space. Default is ``False``.
        random_motion_type : str, optional
            Random motion type. It can be one of the following: ``piecewise_transient``, ``piecewise_constant``,
            ``gaussian``. Default is ``gaussian``.
        random_motion_percentage : Sequence[int], optional
            Random motion percentage. For example, 10%-20% motion can be defined as ``(10, 20)``.
            Default is ``(10, 10)``.
        random_motion_angle : float, optional
            Random motion angle. Default is ``10.0``.
        random_motion_translation : float, optional
            Random motion translation. Default is ``10.0``.
        random_motion_center_percentage : float, optional
            Random motion center percentage. Default is ``0.0``.
        random_motion_num_segments : int, optional
            Random motion number of segments to partition the k-space. Default is ``8``.
        random_motion_random_num_segments : bool, optional
            Whether to randomly generate the number of segments. Default is ``True``.
        random_motion_non_uniform : bool, optional
            Random motion non-uniform sampling. Default is ``False``.
        estimate_coil_sensitivity_maps : bool, optional
            Automatically estimate coil sensitivity maps. Default is ``False``. If ``True`` then the coil sensitivity
            maps arguments are used. Note that this is different from the ``estimate_coil_sensitivity_maps_with_nn``
            argument, which uses a neural network to estimate the coil sensitivity maps. The
            ``estimate_coil_sensitivity_maps`` estimates the coil sensitivity maps with methods such as ``ESPIRiT``,
            ``RSS`` or ``UNit``. ``ESPIRiT`` is the ``Eigenvalue to Self-Consistent Parallel Imaging Reconstruction
            Technique`` method. ``RSS`` is the ``Root Sum of Squares``  method. ``UNit`` returns a uniform coil
            sensitivity map.
        coil_sensitivity_maps_type : str, optional
            Coil sensitivity maps type. It can be one of the following: ``ESPIRiT``, ``RSS`` or ``UNit``. Default is
            ``ESPIRiT``.
        coil_sensitivity_maps_gaussian_sigma : float, optional
            Coil sensitivity maps Gaussian sigma. Default is ``0.0``.
        coil_sensitivity_maps_espirit_threshold : float, optional
            Coil sensitivity maps ESPRIT threshold. Default is ``0.05``.
        coil_sensitivity_maps_espirit_kernel_size : int, optional
            Coil sensitivity maps ESPRIT kernel size. Default is ``6``.
        coil_sensitivity_maps_espirit_crop : float, optional
            Coil sensitivity maps ESPRIT crop. Default is ``0.95``.
        coil_sensitivity_maps_espirit_max_iters : int, optional
            Coil sensitivity maps ESPRIT max iterations. Default is ``30``.
        coil_combination_method : str, optional
            Coil combination method. Default is ``"SENSE"``.
        dimensionality : int, optional
            Dimensionality. Default is ``2``.
        mask_func : Optional[List["MaskFunc"]], optional
            Mask function to retrospectively undersample the k-space. Default is ``None``.
        shift_mask : bool, optional
            Whether to shift the mask. This needs to be set alongside with the ``fft_centered`` argument.
            Default is ``False``.
        mask_center_scale : Optional[float], optional
            Center scale of the mask. This defines how much densely sampled will be the center of k-space.
            Default is ``0.02``.
        partial_fourier_percentage : float, optional
            Whether to simulate a half scan. Default is ``0.0``.
        remask : bool, optional
            Use the same mask. Default is ``False``.
        ssdu : bool, optional
            Whether to apply Self-Supervised Data Undersampling (SSDU) masks. Default is ``False``.
        ssdu_mask_type: str, optional
            Mask type. It can be one of the following:
            - "Gaussian": Gaussian sampling.
            - "Uniform": Uniform sampling.
            Default is "Gaussian".
        ssdu_rho: float, optional
            Split ratio for training and loss masks. Default is ``0.4``.
        ssdu_acs_block_size: tuple, optional
            Keeps a small acs region fully-sampled for training masks, if there is no acs region. The small acs block
            should be set to zero. Default is ``(4, 4)``.
        ssdu_gaussian_std_scaling_factor: float, optional
            Scaling factor for standard deviation of the Gaussian noise. If Uniform is select this factor is ignored.
            Default is ``4.0``.
        ssdu_outer_kspace_fraction: float, optional
            Fraction of the outer k-space to be kept/unmasked. Default is ``0.0``.
        ssdu_export_and_reuse_masks: bool, optional
            Whether to export and reuse the masks. Default is ``False``.
        n2r : bool, optional
            Whether to apply Noise to Reconstruction (N2R) masks. Default is ``False``.
        n2r_supervised_rate : Optional[float], optional
            A float between 0 and 1. This controls what fraction of the subjects should be loaded for Noise to
            Reconstruction (N2R) supervised loss, if N2R is enabled. Default is ``0.0``.
        n2r_probability : float, optional
            Probability of applying N2R. Default is ``0.0``.
        n2r_std_devs : Tuple[float, float], optional
            Standard deviations for the noise. Default is ``(0.0, 0.0)``.
        n2r_rhos : Tuple[float, float], optional
            Rho values for the noise. Default is ``(0.0, 0.0)``.
        n2r_use_mask : bool, optional
            Whether to use a mask for N2R. Default is ``False``.
        unsupervised_masked_target : bool, optional
            Whether to use the masked initial estimation for unsupervised learning. Default is ``False``.
        crop_size : Optional[Tuple[int, int]], optional
            Center crop size. It applies cropping in image space. Default is ``None``.
        kspace_crop : bool, optional
            Whether to crop in k-space. Default is ``False``.
        crop_before_masking : bool, optional
            Whether to crop before masking. Default is ``True``.
        kspace_zero_filling_size : Optional[Tuple], optional
            Whether to apply zero filling in k-space. Default is ``None``.
        normalize_inputs : bool, optional
            Whether to normalize the inputs. Default is ``True``.
        normalization_type : str, optional
            Normalization type. Can be ``max`` or ``mean`` or ``minmax``. Default is ``max``.
        kspace_normalization : bool, optional
            Whether to normalize the k-space. Default is ``False``.
        fft_centered : bool, optional
            Whether to center the FFT. Default is ``False``.
        fft_normalization : str, optional
            FFT normalization. Default is ``"backward"``.
        spatial_dims : Sequence[int], optional
            Spatial dimensions. Default is ``None``.
        coil_dim : int, optional
            Coil dimension. Default is ``0``, meaning that the coil dimension is the first dimension before applying
            batch.
        consecutive_slices : int, optional
            Consecutive slices. Default is ``1``.
        use_seed : bool, optional
            Whether to use seed. Default is ``True``.
        """
        super().__init__()

        if not precompute_quantitative_maps:
            raise ValueError(
                "Loading quantitative maps from disk is not supported yet. "
                "Please set precompute_quantitative_maps to True."
            )
        self.precompute_quantitative_maps = precompute_quantitative_maps

        if TEs is None:
            raise ValueError("Please specify echo times (TEs).")
        self.TEs = TEs
        self.qmaps_scaling_factor = qmaps_scaling_factor

        self.dataset_format = dataset_format

        self.coil_combination_method = coil_combination_method
        self.kspace_crop = kspace_crop
        self.crop_before_masking = crop_before_masking

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim - 1 if dimensionality == 2 else coil_dim

        self.prewhitening = (
            NoisePreWhitening(
                find_patch_size=find_patch_size,
                patch_size=[
                    prewhitening_patch_start,
                    prewhitening_patch_length + prewhitening_patch_start,
                    prewhitening_patch_start,
                    prewhitening_patch_length + prewhitening_patch_start,
                ],
                scale_factor=prewhitening_scale_factor,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            if apply_prewhitening
            else None
        )

        self.gcc = (
            GeometricDecompositionCoilCompression(
                virtual_coils=gcc_virtual_coils,
                calib_lines=gcc_calib_lines,
                align_data=gcc_align_data,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            if apply_gcc
            else None
        )

        self.random_motion = (
            MotionSimulation(
                motion_type=random_motion_type,
                angle=random_motion_angle,
                translation=random_motion_translation,
                center_percentage=random_motion_center_percentage,
                motion_percentage=random_motion_percentage,
                num_segments=random_motion_num_segments,
                random_num_segments=random_motion_random_num_segments,
                non_uniform=random_motion_non_uniform,
                spatial_dims=self.spatial_dims,
            )
            if apply_random_motion
            else None
        )

        self.coil_sensitivity_maps_estimator = (
            EstimateCoilSensitivityMaps(
                coil_sensitivity_maps_type=coil_sensitivity_maps_type.lower(),
                gaussian_sigma=coil_sensitivity_maps_gaussian_sigma,
                espirit_threshold=coil_sensitivity_maps_espirit_threshold,
                espirit_kernel_size=coil_sensitivity_maps_espirit_kernel_size,
                espirit_crop=coil_sensitivity_maps_espirit_crop,
                espirit_max_iters=coil_sensitivity_maps_espirit_max_iters,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
                coil_dim=self.coil_dim,
            )
            if estimate_coil_sensitivity_maps
            else None
        )

        self.kspace_zero_filling = (
            ZeroFillingPadding(
                zero_filling_size=kspace_zero_filling_size,  # type: ignore
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            if not is_none(kspace_zero_filling_size)
            else None
        )

        self.shift_mask = shift_mask
        self.masking = Masker(
            mask_func=mask_func,  # type: ignore
            spatial_dims=self.spatial_dims,
            shift_mask=self.shift_mask,
            partial_fourier_percentage=partial_fourier_percentage,
            center_scale=mask_center_scale,  # type: ignore
            dimensionality=dimensionality,
            remask=remask,
            dataset_format=self.dataset_format,
        )

        self.ssdu = ssdu
        self.ssdu_masking = (
            SSDU(
                mask_type=ssdu_mask_type,
                rho=ssdu_rho,
                acs_block_size=ssdu_acs_block_size,
                gaussian_std_scaling_factor=ssdu_gaussian_std_scaling_factor,
                outer_kspace_fraction=ssdu_outer_kspace_fraction,
                export_and_reuse_masks=ssdu_export_and_reuse_masks,
            )
            if self.ssdu
            else None
        )

        self.n2r = n2r
        self.n2r_supervised_rate = n2r_supervised_rate
        self.n2r_masking = (
            N2R(
                probability=n2r_probability,
                std_devs=n2r_std_devs,  # type: ignore
                rhos=n2r_rhos,  # type: ignore
                use_mask=n2r_use_mask,
            )
            if self.n2r
            else None
        )

        self.unsupervised_masked_target = unsupervised_masked_target

        self.cropping = (
            Cropper(
                cropping_size=crop_size,  # type: ignore
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            if not is_none(crop_size)
            else None
        )

        self.normalization_type = normalization_type
        self.normalization = (
            Normalizer(
                normalization_type=self.normalization_type,
                kspace_normalization=kspace_normalization,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            if normalize_inputs
            else None
        )

        self.prewhitening = Composer([self.prewhitening])  # type: ignore
        self.coils_shape_transforms = Composer(
            [
                self.gcc,  # type: ignore
                self.kspace_zero_filling,  # type: ignore
            ]
        )
        self.crop_normalize = Composer(
            [
                self.cropping,  # type: ignore
                self.normalization,  # type: ignore
            ]
        )
        self.cropping = Composer([self.cropping])  # type: ignore
        self.random_motion = Composer([self.random_motion])  # type: ignore
        self.normalization = Composer([self.normalization])  # type: ignore

        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        initial_prediction: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_idx: int,
    ) -> Tuple[
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        Union[List[torch.Tensor], torch.Tensor],
        str,
        int,
        List[Union[float, torch.Tensor, Any]],
        Dict,
    ]:
        """Calls :class:`qMRIDataTransforms`.

        Parameters
        ----------
        kspace : np.ndarray
            The fully-sampled kspace, if exists. Otherwise, the subsampled kspace.
        sensitivity_map : np.ndarray
            The coil sensitivity map.
        mask : np.ndarray
            The subsampling mask, if exists, meaning that the data are either prospectively undersampled or the mask is
            stored and loaded. It can be a list of masks, with the subsampling, the brain, and the head mask.
        initial_prediction : np.ndarray
            The initial prediction, if exists. Otherwise, it will be estimated with the chosen coil combination method.
        target : np.ndarray
            The target, if exists. Otherwise, it will be estimated with the chosen coil combination method.
        attrs : Dict
            The attributes, if stored in the data.
        fname : str
            The file name.
        slice_idx : int
            The slice index.

        Returns
        -------
        The transformed data.
        """
        mask, anatomy_mask = mask

        if mask.ndim <= 1:
            mask = None

        kspace, masked_kspace, mask, kspace_pre_normalization_vars, acc = self.__process_kspace__(  # type: ignore
            kspace, mask, attrs, fname
        )
        sensitivity_map, sensitivity_pre_normalization_vars = self.__process_coil_sensitivities_map__(
            sensitivity_map, kspace
        )

        if self.n2r and len(masked_kspace) > 1:  # type: ignore
            prediction, prediction_pre_normalization_vars = self.__initialize_prediction__(
                initial_prediction, masked_kspace[0], sensitivity_map  # type: ignore
            )
            if isinstance(masked_kspace, list) and not masked_kspace[1][0].dim() < 2:  # type: ignore
                noise_prediction, noise_prediction_pre_normalization_vars = self.__initialize_prediction__(
                    None, masked_kspace[1], sensitivity_map  # type: ignore
                )
            else:
                noise_prediction = torch.tensor([])
                noise_prediction_pre_normalization_vars = None
            prediction = [prediction, noise_prediction]
        else:
            prediction, prediction_pre_normalization_vars = self.__initialize_prediction__(
                initial_prediction, masked_kspace, sensitivity_map  # type: ignore
            )
            noise_prediction_pre_normalization_vars = None

        if self.unsupervised_masked_target:
            target, target_pre_normalization_vars = prediction, prediction_pre_normalization_vars
        else:
            target, target_pre_normalization_vars = self.__initialize_prediction__(
                None if self.ssdu else target, kspace, sensitivity_map
            )

        if anatomy_mask.ndim != 0:
            anatomy_mask = self.cropping(torch.from_numpy(anatomy_mask))  # type: ignore

        (
            R2star_map_target,
            R2star_map_target_pre_normalization_vars,
            S0_map_target,
            S0_map_target_pre_normalization_vars,
            B0_map_target,
            B0_map_target_pre_normalization_vars,
            phi_map_target,
            phi_map_target_pre_normalization_vars,
        ) = self.__compute_quantitative_maps__(
            kspace, sensitivity_map, None, anatomy_mask
        )  # type: ignore
        (
            R2star_map_init,
            R2star_map_init_pre_normalization_vars,
            S0_map_init,
            S0_map_init_pre_normalization_vars,
            B0_map_init,
            B0_map_init_pre_normalization_vars,
            phi_map_init,
            phi_map_init_pre_normalization_vars,
        ) = self.__compute_quantitative_maps__(  # type: ignore
            masked_kspace, sensitivity_map, prediction, anatomy_mask  # type: ignore
        )

        attrs.update(
            self.__parse_normalization_vars__(
                kspace_pre_normalization_vars,  # type: ignore
                sensitivity_pre_normalization_vars,
                prediction_pre_normalization_vars,
                noise_prediction_pre_normalization_vars,
                target_pre_normalization_vars,
                R2star_map_init_pre_normalization_vars,  # type: ignore
                R2star_map_target_pre_normalization_vars,
                S0_map_init_pre_normalization_vars,  # type: ignore
                S0_map_target_pre_normalization_vars,
                B0_map_init_pre_normalization_vars,  # type: ignore
                B0_map_target_pre_normalization_vars,
                phi_map_init_pre_normalization_vars,  # type: ignore
                phi_map_target_pre_normalization_vars,
            )
        )
        attrs["fname"] = fname
        attrs["slice_idx"] = slice_idx

        return (
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            torch.tensor(self.TEs),
            kspace,
            masked_kspace,  # type: ignore
            sensitivity_map,
            mask,
            anatomy_mask,
            prediction,
            target,
            fname,
            slice_idx,
            acc,  # type: ignore
            attrs,
        )

    def __repr__(self) -> str:
        """Representation of :class:`qMRIDataTransforms`."""
        return (
            f"Preprocessing transforms initialized for {self.__class__.__name__}: "
            f"precompute_quantitative_maps = {self.precompute_quantitative_maps}, "
            f"prewhitening = {self.prewhitening}, "
            f"masking = {self.masking}, "
            f"SSDU masking = {self.ssdu_masking}, "
            f"kspace zero-filling = {self.kspace_zero_filling}, "
            f"cropping = {self.cropping}, "
            f"normalization = {self.normalization}, "
        )

    def __str__(self) -> str:
        """String representation of :class:`qMRIDataTransforms`."""
        return self.__repr__()

    def __process_kspace__(  # noqa: MC0001
        self, kspace: np.ndarray, mask: Union[np.ndarray, None], attrs: Dict, fname: str
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], torch.Tensor], Union[List[torch.Tensor], torch.Tensor], int]:
        """Apply the preprocessing transforms to the kspace.

        Parameters
        ----------
        kspace : torch.Tensor
           The kspace.
        mask : torch.Tensor
            The mask, if None, the mask is generated.
        attrs : Dict
            The attributes, if stored in the file.
        fname : str
            The file name.

        Returns
        -------
        Tuple[torch.Tensor, Union[List[torch.Tensor], torch.Tensor], Union[List[torch.Tensor], torch.Tensor], int]
            The transformed (fully-sampled) kspace, the masked kspace, the mask, the attributes and the acceleration
            factor.
        """
        kspace = to_tensor(kspace)
        kspace = add_coil_dim_if_singlecoil(kspace, dim=self.coil_dim)

        kspace_echoes = []
        for ke in range(kspace.shape[0]):
            kspace_echo = kspace[ke]
            kspace_echo = self.coils_shape_transforms(kspace_echo, apply_backward_transform=True)
            kspace_echo = self.prewhitening(kspace_echo)  # type: ignore
            kspace_echoes.append(kspace_echo)
        kspace = torch.stack(kspace_echoes, dim=0)

        if self.crop_before_masking:
            kspace = self.cropping(kspace, apply_backward_transform=not self.kspace_crop)  # type: ignore

        kspace = torch.stack([self.random_motion(kspace[ke]) for ke in range(kspace.shape[0])], dim=0)  # type: ignore

        masked_kspace, mask, acc = self.masking(
            kspace,
            mask,
            (
                attrs["padding_left"] if "padding_left" in attrs else 0,
                attrs["padding_right"] if "padding_right" in attrs else 0,
            ),
            tuple(map(ord, fname)) if self.use_seed else None,  # type: ignore
        )

        if not self.crop_before_masking:
            kspace = self.cropping(kspace, apply_backward_transform=not self.kspace_crop)  # type: ignore
            masked_kspace = self.cropping(masked_kspace, apply_backward_transform=not self.kspace_crop)  # type: ignore
            mask = self.cropping(mask)  # type: ignore

        init_kspace = kspace
        init_masked_kspace = masked_kspace
        init_mask = mask

        if isinstance(kspace, list):
            kspaces = []
            pre_normalization_vars = []
            for i in range(len(kspace)):  # pylint: disable=consider-using-enumerate
                if not is_none(self.normalization.__repr__()):
                    _kspace, _pre_normalization_vars = self.normalization(  # type: ignore
                        kspace[i], apply_backward_transform=True
                    )
                else:
                    _kspace = kspace[i]
                    is_complex = _kspace.shape[-1] == 2
                    if is_complex:
                        _kspace = torch.view_as_complex(_kspace)
                    _pre_normalization_vars = {
                        "min": torch.min(torch.abs(_kspace)),
                        "max": torch.max(torch.abs(_kspace)),
                        "mean": torch.mean(torch.abs(_kspace)),
                        "std": torch.std(torch.abs(_kspace)),
                        "var": torch.var(torch.abs(_kspace)),
                    }
                    if is_complex:
                        _kspace = torch.view_as_real(_kspace)
                kspaces.append(_kspace)
                pre_normalization_vars.append(_pre_normalization_vars)
            kspace = kspaces
        else:
            if not is_none(self.normalization.__repr__()):
                kspace, pre_normalization_vars = self.normalization(  # type: ignore
                    kspace, apply_backward_transform=True
                )
            else:
                is_complex = kspace.shape[-1] == 2
                if is_complex:
                    kspace = torch.view_as_complex(kspace)
                pre_normalization_vars = {  # type: ignore
                    "min": torch.min(torch.abs(kspace)),
                    "max": torch.max(torch.abs(kspace)),
                    "mean": torch.mean(torch.abs(kspace)),
                    "std": torch.std(torch.abs(kspace)),
                    "var": torch.var(torch.abs(kspace)),
                }
                if is_complex:
                    kspace = torch.view_as_real(kspace)

        if isinstance(masked_kspace, list):
            masked_kspaces = []
            masked_pre_normalization_vars = []
            for i in range(len(masked_kspace)):  # pylint: disable=consider-using-enumerate
                if not is_none(self.normalization.__repr__()):
                    _masked_kspace, _masked_pre_normalization_vars = self.normalization(  # type: ignore
                        masked_kspace[i], apply_backward_transform=True
                    )
                else:
                    _masked_kspace = masked_kspace[i]
                    is_complex = _masked_kspace.shape[-1] == 2
                    if is_complex:
                        _masked_kspace = torch.view_as_complex(_masked_kspace)
                    _masked_pre_normalization_vars = {
                        "min": torch.min(torch.abs(_masked_kspace)),
                        "max": torch.max(torch.abs(_masked_kspace)),
                        "mean": torch.mean(torch.abs(_masked_kspace)),
                        "std": torch.std(torch.abs(_masked_kspace)),
                        "var": torch.var(torch.abs(_masked_kspace)),
                    }
                    if is_complex:
                        _masked_kspace = torch.view_as_real(_masked_kspace)
                masked_kspaces.append(_masked_kspace)
                masked_pre_normalization_vars.append(_masked_pre_normalization_vars)
            masked_kspace = masked_kspaces
        else:
            if not is_none(self.normalization.__repr__()):
                masked_kspace, masked_pre_normalization_vars = self.normalization(
                    masked_kspace, apply_backward_transform=True
                )
            else:
                is_complex = masked_kspace.shape[-1] == 2
                if is_complex:
                    masked_kspace = torch.view_as_complex(masked_kspace)
                masked_pre_normalization_vars = {
                    "min": torch.min(torch.abs(masked_kspace)),
                    "max": torch.max(torch.abs(masked_kspace)),
                    "mean": torch.mean(torch.abs(masked_kspace)),
                    "std": torch.std(torch.abs(masked_kspace)),
                    "var": torch.var(torch.abs(masked_kspace)),
                }
                if is_complex:
                    masked_kspace = torch.view_as_real(masked_kspace)

        if self.ssdu:
            kspace, masked_kspace, mask = self.__self_supervised_data_undersampling__(  # type: ignore
                kspace, masked_kspace, mask, fname
            )

        n2r_pre_normalization_vars = None
        if self.n2r and (not attrs["n2r_supervised"] or self.ssdu):
            n2r_masked_kspace, n2r_mask = self.__noise_to_reconstruction__(init_kspace, init_masked_kspace, init_mask)

            if self.ssdu:
                if isinstance(mask, list):
                    for i in range(len(mask)):  # pylint: disable=consider-using-enumerate
                        if init_mask[i].dim() != mask[i][0].dim():  # type: ignore
                            # find dimensions == 1 in mask[i][0] and add them to init_mask
                            unitary_dims = [j for j in range(mask[i][0].dim()) if mask[i][0].shape[j] == 1]
                            # unsqueeze init_mask to the index of the unitary dimensions
                            for j in unitary_dims:
                                init_mask[i] = init_mask[i].unsqueeze(j)  # type: ignore
                        masked_kspace[i] = init_masked_kspace[i]
                        mask[i][0] = init_mask[i]
                else:
                    if init_mask.dim() != mask[0].dim():  # type: ignore
                        # find dimensions == 1 in mask[0] and add them to init_mask
                        unitary_dims = [j for j in range(mask[0].dim()) if mask[0].shape[j] == 1]
                        # unsqueeze init_mask to the index of the unitary dimensions
                        for j in unitary_dims:
                            init_mask = init_mask.unsqueeze(j)  # type: ignore
                    masked_kspace = init_masked_kspace
                    mask[0] = init_mask

            if "None" not in self.normalization.__repr__():
                if isinstance(masked_kspace, list):
                    masked_kspaces = []
                    masked_pre_normalization_vars = []
                    for i in range(len(masked_kspace)):  # pylint: disable=consider-using-enumerate
                        _masked_kspace, _masked_pre_normalization_vars = self.normalization(  # type: ignore
                            masked_kspace[i], apply_backward_transform=True
                        )
                        masked_kspaces.append(_masked_kspace)
                        masked_pre_normalization_vars.append(_masked_pre_normalization_vars)
                    masked_kspace = masked_kspaces
                else:
                    masked_kspace, masked_pre_normalization_vars = self.normalization(  # type: ignore
                        masked_kspace, apply_backward_transform=True
                    )
                if isinstance(n2r_masked_kspace, list):
                    n2r_masked_kspaces = []
                    n2r_pre_normalization_vars = []
                    for i in range(len(n2r_masked_kspace)):  # pylint: disable=consider-using-enumerate
                        _n2r_masked_kspace, _n2r_pre_normalization_vars = self.normalization(  # type: ignore
                            n2r_masked_kspace[i], apply_backward_transform=True
                        )
                        n2r_masked_kspaces.append(_n2r_masked_kspace)
                        n2r_pre_normalization_vars.append(_n2r_pre_normalization_vars)
                    n2r_masked_kspace = n2r_masked_kspaces
                else:
                    n2r_masked_kspace, n2r_pre_normalization_vars = self.normalization(  # type: ignore
                        n2r_masked_kspace, apply_backward_transform=True
                    )
            else:
                masked_pre_normalization_vars = None  # type: ignore
                n2r_pre_normalization_vars = None  # type: ignore

            masked_kspace = [masked_kspace, n2r_masked_kspace]
            mask = [mask, n2r_mask]

        if self.normalization_type == "grayscale":
            if isinstance(mask, list):
                masks = []
                for i in range(len(mask)):  # pylint: disable=consider-using-enumerate
                    _mask, _ = self.normalization(mask[i], apply_backward_transform=False)  # type: ignore
                    masks.append(_mask)
                mask = masks
            else:
                mask, _ = self.normalization(mask, apply_backward_transform=False)  # type: ignore

        pre_normalization_vars = {  # type: ignore
            "kspace_pre_normalization_vars": pre_normalization_vars,
            "masked_kspace_pre_normalization_vars": masked_pre_normalization_vars,
            "noise_masked_kspace_pre_normalization_vars": n2r_pre_normalization_vars,
        }

        return kspace, masked_kspace, mask, pre_normalization_vars, acc  # type: ignore

    def __noise_to_reconstruction__(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: Union[List, torch.Tensor],
    ) -> Tuple[Union[List, torch.Tensor], Union[List, torch.Tensor]]:
        """Apply the noise-to-reconstruction transform.

        Parameters
        ----------
        kspace : torch.Tensor
            The fully-sampled kspace.
        masked_kspace : torch.Tensor
            The undersampled kspace.
        mask : Union[List, torch.Tensor]
            The undersampling mask.

        Returns
        -------
        n2r_masked_kspace : Union[List, torch.Tensor]
            The noise-to-reconstruction undersampled kspace.
        n2r_mask : Union[List, torch.Tensor]
            The noise-to-reconstruction mask.
        """
        if isinstance(mask, list):
            n2r_masked_kspaces = []
            n2r_masks = []
            for i in range(len(mask)):  # pylint: disable=consider-using-enumerate
                n2r_mask = self.n2r_masking(kspace, mask[i])  # type: ignore  # pylint: disable=not-callable
                n2r_masks.append(n2r_mask)
                n2r_masked_kspaces.append(masked_kspace[i] * n2r_mask + 0.0)
            n2r_mask = n2r_masks
            n2r_masked_kspace = n2r_masked_kspaces
        else:
            n2r_mask = self.n2r_masking(kspace, mask)  # type: ignore  # pylint: disable=not-callable
            n2r_masked_kspace = masked_kspace * n2r_mask + 0.0
        return n2r_masked_kspace, n2r_mask

    def __self_supervised_data_undersampling__(  # noqa: MC0001
        self,
        kspace: torch.Tensor,
        masked_kspace: Union[List, torch.Tensor],
        mask: Union[List, torch.Tensor],
        fname: str,
    ) -> Tuple[
        List[float | Any] | float | Any,
        List[float | Any] | float | Any,
        List[List[torch.Tensor | Any]] | List[torch.Tensor | Any],
    ]:
        """Self-supervised data undersampling.

        Parameters
        ----------
        kspace : torch.Tensor
            The fully-sampled kspace.
        masked_kspace : Union[List, torch.Tensor]
            The undersampled kspace.
        mask : Union[List, torch.Tensor]
            The undersampling mask.
        fname : str
            The filename of the current sample.

        Returns
        -------
        kspace : torch.Tensor
            The kspace with the loss mask applied.
        masked_kspace : torch.Tensor
            The kspace with the train mask applied.
        mask : list, [torch.Tensor, torch.Tensor]
            The train and loss masks.
        """
        if isinstance(mask, list):
            kspaces = []
            masked_kspaces = []
            masks = []
            for i in range(len(mask)):  # pylint: disable=consider-using-enumerate
                is_1d = mask[i].squeeze().dim() == 1
                if self.shift_mask:
                    mask[i] = torch.fft.fftshift(mask[i].squeeze(-1), dim=(-2, -1)).unsqueeze(-1)
                mask[i] = mask[i].squeeze()
                if is_1d:
                    mask[i] = mask[i].unsqueeze(0).repeat_interleave(kspace.shape[1], dim=0)
                train_mask, loss_mask = self.ssdu_masking(  # type: ignore  # pylint: disable=not-callable
                    kspace, mask[i], fname
                )
                if self.shift_mask:
                    train_mask = torch.fft.fftshift(train_mask, dim=(0, 1))
                    loss_mask = torch.fft.fftshift(loss_mask, dim=(0, 1))
                if is_1d:
                    train_mask = train_mask.unsqueeze(0).unsqueeze(-1)
                    loss_mask = loss_mask.unsqueeze(0).unsqueeze(-1)
                else:
                    # find unitary dims in mask
                    dims = [i for i, x in enumerate(mask[i].shape) if x == 1]
                    # unsqueeze to broadcast
                    for d in dims:
                        train_mask = train_mask.unsqueeze(d)
                        loss_mask = loss_mask.unsqueeze(d)
                if train_mask.dim() != kspace.dim():
                    # find dims != to any train_mask dim
                    dims = [i for i, x in enumerate(kspace.shape) if x not in train_mask.shape]
                    # unsqueeze to broadcast
                    for d in dims:
                        train_mask = train_mask.unsqueeze(d)
                        loss_mask = loss_mask.unsqueeze(d)
                kspaces.append(kspace * loss_mask + 0.0)
                masked_kspaces.append(masked_kspace[i] * train_mask + 0.0)
                masks.append([train_mask, loss_mask])
            kspace = kspaces
            masked_kspace = masked_kspaces
            mask = masks
        else:
            is_1d = mask.squeeze().dim() == 1
            if self.shift_mask:
                mask = torch.fft.fftshift(mask.squeeze(-1), dim=(-2, -1)).unsqueeze(-1)
            mask = mask.squeeze()
            if is_1d:
                mask = mask.unsqueeze(0).repeat_interleave(kspace.shape[1], dim=0)
            train_mask, loss_mask = self.ssdu_masking(  # type: ignore  # pylint: disable=not-callable
                kspace, mask, fname
            )
            if self.shift_mask:
                train_mask = torch.fft.fftshift(train_mask, dim=(0, 1))
                loss_mask = torch.fft.fftshift(loss_mask, dim=(0, 1))
            if is_1d:
                train_mask = train_mask.unsqueeze(0).unsqueeze(-1)
                loss_mask = loss_mask.unsqueeze(0).unsqueeze(-1)
            else:
                # find unitary dims in mask
                dims = [i for i, x in enumerate(mask.shape) if x == 1]
                # unsqueeze to broadcast
                for d in dims:
                    train_mask = train_mask.unsqueeze(d)
                    loss_mask = loss_mask.unsqueeze(d)
            if train_mask.dim() != kspace.dim():
                # find dims != to any train_mask dim
                dims = [i for i, x in enumerate(kspace.shape) if x not in train_mask.shape]
                # unsqueeze to broadcast
                for d in dims:
                    train_mask = train_mask.unsqueeze(d)
                    loss_mask = loss_mask.unsqueeze(d)
            kspace = kspace * loss_mask + 0.0
            masked_kspace = masked_kspace * train_mask + 0.0
            mask = [train_mask, loss_mask]
        return kspace, masked_kspace, mask

    def __process_coil_sensitivities_map__(self, sensitivity_map: np.ndarray, kspace: torch.Tensor) -> torch.Tensor:
        """Preprocesses the coil sensitivities map.

        Parameters
        ----------
        sensitivity_map : np.ndarray
            The coil sensitivities map.
        kspace : torch.Tensor
            The kspace.

        Returns
        -------
        torch.Tensor
            The preprocessed coil sensitivities map.
        """
        # This condition is necessary in case of auto estimation of sense maps.
        if self.coil_sensitivity_maps_estimator is not None:
            sensitivity_map = self.coil_sensitivity_maps_estimator(kspace)
        elif sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = to_tensor(sensitivity_map)
            sensitivity_map = self.coils_shape_transforms(sensitivity_map, apply_forward_transform=True)
            sensitivity_map = self.cropping(sensitivity_map, apply_forward_transform=self.kspace_crop)  # type: ignore
        else:
            # If no sensitivity map is provided, either the data is singlecoil or the sense net is used.
            # Initialize the sensitivity map to 1 to assure for the singlecoil case.
            sensitivity_map = torch.ones_like(kspace) if not isinstance(kspace, list) else torch.ones_like(kspace[0])
        if not is_none(self.normalization.__repr__()):
            sensitivity_map, pre_normalization_vars = self.normalization(  # type: ignore
                sensitivity_map, apply_forward_transform=self.kspace_crop
            )
        else:
            is_complex = sensitivity_map.shape[-1] == 2
            if is_complex:
                sensitivity_map = torch.view_as_complex(sensitivity_map)
            pre_normalization_vars = {
                "min": torch.min(torch.abs(sensitivity_map)),
                "max": torch.max(torch.abs(sensitivity_map)),
                "mean": torch.mean(torch.abs(sensitivity_map)),
                "std": torch.std(torch.abs(sensitivity_map)),
                "var": torch.var(torch.abs(sensitivity_map)),
            }
            if is_complex:
                sensitivity_map = torch.view_as_real(sensitivity_map)
        return sensitivity_map, pre_normalization_vars

    def __compute_quantitative_maps__(
        self,
        kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        prediction: Union[torch.Tensor, None],
        anatomy_mask: torch.Tensor,
    ) -> Tuple[
        List[Any] | Any,
        List[Dict[str, torch.Tensor]],
        List[Any] | Any,
        List[Dict[str, torch.Tensor]],
        List[Any] | Any,
        List[Dict[str, torch.Tensor]],
        List[Any] | Any,
        List[Dict[str, torch.Tensor]],
    ]:
        """Compute quantitative maps from the masked kspace data.

        Parameters
        ----------
        kspace: torch.Tensor
            Kspace data.
        sensitivity_map: torch.Tensor
            Sensitivity maps.
        prediction: Union[torch.Tensor, None]
            Initial prediction or None.
        anatomy_mask: torch.Tensor
            Brain mask.

        Returns
        -------
        R2star_map: torch.Tensor
            Computed R2* map.
        R2star_map_pre_normalization_vars: Dict[str, torch.Tensor]
            Computed R2* map pre-normalization variables.
        S0_map: torch.Tensor
            Computed S0 map.
        S0_map_pre_normalization_vars: Dict[str, torch.Tensor]
            Computed S0 map pre-normalization variables.
        B0_map: torch.Tensor
            Computed B0 map.
        B0_map_pre_normalization_vars: Dict[str, torch.Tensor]
            Computed B0 map pre-normalization variables.
        phi_map: torch.Tensor
            Computed phi map.
        phi_map_pre_normalization_vars: Dict[str, torch.Tensor]
            Computed phi map pre-normalization variables.
        """
        R2star_maps = []
        R2star_map_pre_normalization_vars = []
        S0_maps = []
        S0_map_pre_normalization_vars = []
        B0_maps = []
        B0_map_pre_normalization_vars = []
        phi_maps = []
        phi_map_pre_normalization_vars = []
        if isinstance(kspace, list):
            for i, _ in enumerate(kspace):
                R2star_map, S0_map, B0_map, phi_map = R2star_B0_S0_phi_mapping(
                    prediction[i] if isinstance(prediction, list) else prediction,
                    self.TEs,
                    anatomy_mask,
                    scaling_factor=self.qmaps_scaling_factor,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

                R2star_maps.append(R2star_map)
                R2star_map_pre_normalization_vars.append(
                    {
                        "min": torch.min(torch.abs(R2star_map)),
                        "max": torch.max(torch.abs(R2star_map)),
                        "mean": torch.mean(torch.abs(R2star_map)),
                        "std": torch.std(torch.abs(R2star_map)),
                        "var": torch.var(torch.abs(R2star_map)),
                    }
                )
                S0_maps.append(S0_map)
                S0_map_pre_normalization_vars.append(
                    {
                        "min": torch.min(torch.abs(S0_map)),
                        "max": torch.max(torch.abs(S0_map)),
                        "mean": torch.mean(torch.abs(S0_map)),
                        "std": torch.std(torch.abs(S0_map)),
                        "var": torch.var(torch.abs(S0_map)),
                    }
                )
                B0_maps.append(B0_map)
                B0_map_pre_normalization_vars.append(
                    {
                        "min": torch.min(torch.abs(B0_map)),
                        "max": torch.max(torch.abs(B0_map)),
                        "mean": torch.mean(torch.abs(B0_map)),
                        "std": torch.std(torch.abs(B0_map)),
                        "var": torch.var(torch.abs(B0_map)),
                    }
                )
                phi_maps.append(phi_map)
                phi_map_pre_normalization_vars.append(
                    {
                        "min": torch.min(torch.abs(phi_map)),
                        "max": torch.max(torch.abs(phi_map)),
                        "mean": torch.mean(torch.abs(phi_map)),
                        "std": torch.std(torch.abs(phi_map)),
                        "var": torch.var(torch.abs(phi_map)),
                    }
                )

            R2star_map = R2star_maps
            S0_map = S0_maps
            B0_map = B0_maps
            phi_map = phi_maps
        else:
            if prediction is None:
                prediction = sense(
                    ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_map.unsqueeze(0),
                    dim=self.coil_dim,
                )

            R2star_map, S0_map, B0_map, phi_map = R2star_B0_S0_phi_mapping(
                prediction,
                self.TEs,
                anatomy_mask,
                scaling_factor=self.qmaps_scaling_factor,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

            R2star_map_pre_normalization_vars = {  # type: ignore
                "min": torch.min(torch.abs(R2star_map)),
                "max": torch.max(torch.abs(R2star_map)),
                "mean": torch.mean(torch.abs(R2star_map)),
                "std": torch.std(torch.abs(R2star_map)),
                "var": torch.var(torch.abs(R2star_map)),
            }
            S0_map_pre_normalization_vars = {  # type: ignore
                "min": torch.min(torch.abs(S0_map)),
                "max": torch.max(torch.abs(S0_map)),
                "mean": torch.mean(torch.abs(S0_map)),
                "std": torch.std(torch.abs(S0_map)),
                "var": torch.var(torch.abs(S0_map)),
            }
            B0_map_pre_normalization_vars = {  # type: ignore
                "min": torch.min(torch.abs(B0_map)),
                "max": torch.max(torch.abs(B0_map)),
                "mean": torch.mean(torch.abs(B0_map)),
                "std": torch.std(torch.abs(B0_map)),
                "var": torch.var(torch.abs(B0_map)),
            }
            phi_map_pre_normalization_vars = {  # type: ignore
                "min": torch.min(torch.abs(phi_map)),
                "max": torch.max(torch.abs(phi_map)),
                "mean": torch.mean(torch.abs(phi_map)),
                "std": torch.std(torch.abs(phi_map)),
                "var": torch.var(torch.abs(phi_map)),
            }

        return (
            R2star_map,
            R2star_map_pre_normalization_vars,
            S0_map,
            S0_map_pre_normalization_vars,
            B0_map,
            B0_map_pre_normalization_vars,
            phi_map,
            phi_map_pre_normalization_vars,
        )

    def __initialize_prediction__(
        self, prediction: Union[torch.Tensor, np.ndarray, None], kspace: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """Predicts a coil-combined image.

        Parameters
        ----------
        prediction : np.ndarray
            The initial estimation, if None, the prediction is initialized.
        kspace : torch.Tensor
            The kspace.
        sensitivity_map : torch.Tensor
            The sensitivity map.

        Returns
        -------
        Union[List[torch.Tensor], torch.Tensor]
            The initialized prediction, either a list of coil-combined images or a single coil-combined image.
        """
        if is_none(prediction) or prediction.ndim < 2:  # type: ignore
            if isinstance(kspace, list):
                prediction = []
                pre_normalization_vars = []
                for y in kspace:
                    pred = coil_combination_method_func(
                        ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                        sensitivity_map,
                        method=self.coil_combination_method,
                        dim=self.coil_dim,
                    )
                    pred = self.cropping(pred, apply_forward_transform=self.kspace_crop)  # type: ignore
                    if not is_none(self.normalization.__repr__()):
                        pred, _pre_normalization_vars = self.normalization(  # type: ignore
                            pred, apply_forward_transform=self.kspace_crop
                        )
                    else:
                        if pred.shape[-1] == 2:
                            pred = torch.view_as_complex(pred)
                        _pre_normalization_vars = {
                            "min": torch.min(torch.abs(pred)),
                            "max": torch.max(torch.abs(pred)),
                            "mean": torch.mean(torch.abs(pred)),
                            "std": torch.std(torch.abs(pred)),
                            "var": torch.var(torch.abs(pred)),
                        }
                    prediction.append(pred)
                    pre_normalization_vars.append(_pre_normalization_vars)
                if prediction[0].shape[-1] != 2 and torch.is_complex(prediction[0]):
                    prediction = [torch.view_as_real(x) for x in prediction]
            else:
                prediction = coil_combination_method_func(
                    ifft2(kspace, self.fft_centered, self.fft_normalization, self.spatial_dims),
                    sensitivity_map,
                    method=self.coil_combination_method,
                    dim=self.coil_dim,
                )
                prediction = self.cropping(prediction, apply_forward_transform=self.kspace_crop)  # type: ignore
                if not is_none(self.normalization.__repr__()):
                    prediction, pre_normalization_vars = self.normalization(  # type: ignore
                        prediction, apply_forward_transform=self.kspace_crop
                    )
                else:
                    if prediction.shape[-1] == 2:
                        prediction = torch.view_as_complex(prediction)
                    pre_normalization_vars = {  # type: ignore
                        "min": torch.min(torch.abs(prediction)),
                        "max": torch.max(torch.abs(prediction)),
                        "mean": torch.mean(torch.abs(prediction)),
                        "std": torch.std(torch.abs(prediction)),
                        "var": torch.var(torch.abs(prediction)),
                    }
                if prediction.shape[-1] != 2 and torch.is_complex(prediction):
                    prediction = torch.view_as_real(prediction)
        else:
            if isinstance(prediction, np.ndarray):
                prediction = to_tensor(prediction)
            prediction = self.cropping(prediction, apply_forward_transform=self.kspace_crop)  # type: ignore
            if not is_none(self.normalization.__repr__()):
                prediction, pre_normalization_vars = self.normalization(  # type: ignore
                    prediction, apply_forward_transform=self.kspace_crop
                )
            else:
                if prediction.shape[-1] == 2:  # type: ignore
                    prediction = torch.view_as_complex(prediction)
                pre_normalization_vars = {  # type: ignore
                    "min": torch.min(torch.abs(prediction)),
                    "max": torch.max(torch.abs(prediction)),
                    "mean": torch.mean(torch.abs(prediction)),
                    "std": torch.std(torch.abs(prediction)),
                    "var": torch.var(torch.abs(prediction)),
                }
            if prediction.shape[-1] != 2 and torch.is_complex(prediction):
                prediction = torch.view_as_real(prediction)

        return prediction, pre_normalization_vars

    @staticmethod
    def __parse_normalization_vars__(  # noqa: MC0001
        kspace_vars,
        sensitivity_vars,
        prediction_vars,
        noise_prediction_vars,
        target_vars,
        R2star_maps_init_vars,
        R2star_map_target_vars,
        S0_map_init_vars,
        S0_map_target_vars,
        B0_map_init_vars,
        B0_map_target_vars,
        phi_map_init_vars,
        phi_map_target_vars,
    ) -> Dict:
        """Parses the normalization variables and returns a unified dictionary.

        Parameters
        ----------
        kspace_vars : Dict
            The kspace normalization variables.
        sensitivity_vars : Dict
            The sensitivity map normalization variables.
        prediction_vars : Dict
            The prediction normalization variables.
        noise_prediction_vars : Union[Dict, None]
            The noise prediction normalization variables.
        target_vars : Dict
            The target normalization variables.
        R2star_maps_init_vars : Dict
            The R2* maps initialization normalization variables.
        R2star_map_target_vars : Dict
            The R2* maps target normalization variables.
        S0_map_init_vars : Dict
            The S0 maps initialization normalization variables.
        S0_map_target_vars : Dict
            The S0 maps target normalization variables.
        B0_map_init_vars : Dict
            The B0 maps initialization normalization variables.
        B0_map_target_vars : Dict
            The B0 maps target normalization variables.
        phi_map_init_vars : Dict
            The phi maps initialization normalization variables.
        phi_map_target_vars : Dict
            The phi maps target normalization variables.

        Returns
        -------
        Dict
            The normalization variables.
        """
        normalization_vars = {}

        masked_kspace_vars = kspace_vars["masked_kspace_pre_normalization_vars"]
        if isinstance(masked_kspace_vars, list):
            if masked_kspace_vars[0] is not None:
                for i, masked_kspace_var in enumerate(masked_kspace_vars):
                    normalization_vars[f"masked_kspace_min_{i}"] = masked_kspace_var["min"]
                    normalization_vars[f"masked_kspace_max_{i}"] = masked_kspace_var["max"]
                    normalization_vars[f"masked_kspace_mean_{i}"] = masked_kspace_var["mean"]
                    normalization_vars[f"masked_kspace_std_{i}"] = masked_kspace_var["std"]
                    normalization_vars[f"masked_kspace_var_{i}"] = masked_kspace_var["var"]
        else:
            if masked_kspace_vars is not None:
                normalization_vars["masked_kspace_min"] = masked_kspace_vars["min"]
                normalization_vars["masked_kspace_max"] = masked_kspace_vars["max"]
                normalization_vars["masked_kspace_mean"] = masked_kspace_vars["mean"]
                normalization_vars["masked_kspace_std"] = masked_kspace_vars["std"]
                normalization_vars["masked_kspace_var"] = masked_kspace_vars["var"]

        noise_masked_kspace_vars = kspace_vars["noise_masked_kspace_pre_normalization_vars"]
        if noise_masked_kspace_vars is not None:
            if isinstance(noise_masked_kspace_vars, list):
                if noise_masked_kspace_vars[0] is not None:
                    for i, noise_masked_kspace_var in enumerate(noise_masked_kspace_vars):
                        normalization_vars[f"noise_masked_kspace_min_{i}"] = noise_masked_kspace_var["min"]
                        normalization_vars[f"noise_masked_kspace_max_{i}"] = noise_masked_kspace_var["max"]
                        normalization_vars[f"noise_masked_kspace_mean_{i}"] = noise_masked_kspace_var["mean"]
                        normalization_vars[f"noise_masked_kspace_std_{i}"] = noise_masked_kspace_var["std"]
                        normalization_vars[f"noise_masked_kspace_var_{i}"] = noise_masked_kspace_var["var"]
            else:
                if noise_masked_kspace_vars is not None:
                    normalization_vars["noise_masked_kspace_min"] = noise_masked_kspace_vars["min"]
                    normalization_vars["noise_masked_kspace_max"] = noise_masked_kspace_vars["max"]
                    normalization_vars["noise_masked_kspace_mean"] = noise_masked_kspace_vars["mean"]
                    normalization_vars["noise_masked_kspace_std"] = noise_masked_kspace_vars["std"]
                    normalization_vars["noise_masked_kspace_var"] = noise_masked_kspace_vars["var"]

        kspace_vars = kspace_vars["kspace_pre_normalization_vars"]
        if isinstance(kspace_vars, list):
            if kspace_vars[0] is not None:
                for i, kspace_var in enumerate(kspace_vars):
                    normalization_vars[f"kspace_min_{i}"] = kspace_var["min"]
                    normalization_vars[f"kspace_max_{i}"] = kspace_var["max"]
                    normalization_vars[f"kspace_mean_{i}"] = kspace_var["mean"]
                    normalization_vars[f"kspace_std_{i}"] = kspace_var["std"]
                    normalization_vars[f"kspace_var_{i}"] = kspace_var["var"]
        else:
            if kspace_vars is not None:
                normalization_vars["kspace_min"] = kspace_vars["min"]
                normalization_vars["kspace_max"] = kspace_vars["max"]
                normalization_vars["kspace_mean"] = kspace_vars["mean"]
                normalization_vars["kspace_std"] = kspace_vars["std"]
                normalization_vars["kspace_var"] = kspace_vars["var"]

        if sensitivity_vars is not None:
            normalization_vars["sensitivity_maps_min"] = sensitivity_vars["min"]
            normalization_vars["sensitivity_maps_max"] = sensitivity_vars["max"]
            normalization_vars["sensitivity_maps_mean"] = sensitivity_vars["mean"]
            normalization_vars["sensitivity_maps_std"] = sensitivity_vars["std"]
            normalization_vars["sensitivity_maps_var"] = sensitivity_vars["var"]

        if isinstance(prediction_vars, list):
            if prediction_vars[0] is not None:
                for i, prediction_var in enumerate(prediction_vars):
                    normalization_vars[f"prediction_min_{i}"] = prediction_var["min"]
                    normalization_vars[f"prediction_max_{i}"] = prediction_var["max"]
                    normalization_vars[f"prediction_mean_{i}"] = prediction_var["mean"]
                    normalization_vars[f"prediction_std_{i}"] = prediction_var["std"]
                    normalization_vars[f"prediction_var_{i}"] = prediction_var["var"]
        else:
            if prediction_vars is not None:
                normalization_vars["prediction_min"] = prediction_vars["min"]
                normalization_vars["prediction_max"] = prediction_vars["max"]
                normalization_vars["prediction_mean"] = prediction_vars["mean"]
                normalization_vars["prediction_std"] = prediction_vars["std"]
                normalization_vars["prediction_var"] = prediction_vars["var"]

        if noise_prediction_vars is not None:
            if isinstance(noise_prediction_vars, list):
                for i, noise_prediction_var in enumerate(noise_prediction_vars):
                    normalization_vars[f"noise_prediction_min_{i}"] = noise_prediction_var["min"]
                    normalization_vars[f"noise_prediction_max_{i}"] = noise_prediction_var["max"]
                    normalization_vars[f"noise_prediction_mean_{i}"] = noise_prediction_var["mean"]
                    normalization_vars[f"noise_prediction_std_{i}"] = noise_prediction_var["std"]
                    normalization_vars[f"noise_prediction_var_{i}"] = noise_prediction_var["var"]
            else:
                normalization_vars["noise_prediction_min"] = noise_prediction_vars["min"]
                normalization_vars["noise_prediction_max"] = noise_prediction_vars["max"]
                normalization_vars["noise_prediction_mean"] = noise_prediction_vars["mean"]
                normalization_vars["noise_prediction_std"] = noise_prediction_vars["std"]
                normalization_vars["noise_prediction_var"] = noise_prediction_vars["var"]

        if isinstance(target_vars, list):
            if target_vars[0] is not None:
                for i, target_var in enumerate(target_vars):
                    normalization_vars[f"target_min_{i}"] = target_var["min"]
                    normalization_vars[f"target_max_{i}"] = target_var["max"]
                    normalization_vars[f"target_mean_{i}"] = target_var["mean"]
                    normalization_vars[f"target_std_{i}"] = target_var["std"]
                    normalization_vars[f"target_var_{i}"] = target_var["var"]
        else:
            if target_vars is not None:
                normalization_vars["target_min"] = target_vars["min"]
                normalization_vars["target_max"] = target_vars["max"]
                normalization_vars["target_mean"] = target_vars["mean"]
                normalization_vars["target_std"] = target_vars["std"]
                normalization_vars["target_var"] = target_vars["var"]

        if isinstance(R2star_maps_init_vars, list):
            if R2star_maps_init_vars[0] is not None:
                for i, R2star_map_init_var in enumerate(R2star_maps_init_vars):
                    normalization_vars[f"R2star_map_init_min_{i}"] = R2star_map_init_var["min"]
                    normalization_vars[f"R2star_map_init_max_{i}"] = R2star_map_init_var["max"]
                    normalization_vars[f"R2star_map_init_mean_{i}"] = R2star_map_init_var["mean"]
                    normalization_vars[f"R2star_map_init_std_{i}"] = R2star_map_init_var["std"]
                    normalization_vars[f"R2star_map_init_var_{i}"] = R2star_map_init_var["var"]
        else:
            if R2star_maps_init_vars is not None:
                normalization_vars["R2star_map_init_min"] = R2star_maps_init_vars["min"]
                normalization_vars["R2star_map_init_max"] = R2star_maps_init_vars["max"]
                normalization_vars["R2star_map_init_mean"] = R2star_maps_init_vars["mean"]
                normalization_vars["R2star_map_init_std"] = R2star_maps_init_vars["std"]
                normalization_vars["R2star_map_init_var"] = R2star_maps_init_vars["var"]

        if isinstance(R2star_map_target_vars, list):
            if R2star_map_target_vars[0] is not None:
                for i, R2star_map_target_var in enumerate(R2star_map_target_vars):
                    normalization_vars[f"R2star_map_target_min_{i}"] = R2star_map_target_var["min"]
                    normalization_vars[f"R2star_map_target_max_{i}"] = R2star_map_target_var["max"]
                    normalization_vars[f"R2star_map_target_mean_{i}"] = R2star_map_target_var["mean"]
                    normalization_vars[f"R2star_map_target_std_{i}"] = R2star_map_target_var["std"]
                    normalization_vars[f"R2star_map_target_var_{i}"] = R2star_map_target_var["var"]
        else:
            if R2star_map_target_vars is not None:
                normalization_vars["R2star_map_target_min"] = R2star_map_target_vars["min"]
                normalization_vars["R2star_map_target_max"] = R2star_map_target_vars["max"]
                normalization_vars["R2star_map_target_mean"] = R2star_map_target_vars["mean"]
                normalization_vars["R2star_map_target_std"] = R2star_map_target_vars["std"]
                normalization_vars["R2star_map_target_var"] = R2star_map_target_vars["var"]

        if isinstance(S0_map_init_vars, list):
            if S0_map_init_vars[0] is not None:
                for i, S0_map_init_var in enumerate(S0_map_init_vars):
                    normalization_vars[f"S0_map_init_min_{i}"] = S0_map_init_var["min"]
                    normalization_vars[f"S0_map_init_max_{i}"] = S0_map_init_var["max"]
                    normalization_vars[f"S0_map_init_mean_{i}"] = S0_map_init_var["mean"]
                    normalization_vars[f"S0_map_init_std_{i}"] = S0_map_init_var["std"]
                    normalization_vars[f"S0_map_init_var_{i}"] = S0_map_init_var["var"]
        else:
            if S0_map_init_vars is not None:
                normalization_vars["S0_map_init_min"] = S0_map_init_vars["min"]
                normalization_vars["S0_map_init_max"] = S0_map_init_vars["max"]
                normalization_vars["S0_map_init_mean"] = S0_map_init_vars["mean"]
                normalization_vars["S0_map_init_std"] = S0_map_init_vars["std"]
                normalization_vars["S0_map_init_var"] = S0_map_init_vars["var"]

        if isinstance(S0_map_target_vars, list):
            if S0_map_target_vars[0] is not None:
                for i, S0_map_target_var in enumerate(S0_map_target_vars):
                    normalization_vars[f"S0_map_target_min_{i}"] = S0_map_target_var["min"]
                    normalization_vars[f"S0_map_target_max_{i}"] = S0_map_target_var["max"]
                    normalization_vars[f"S0_map_target_mean_{i}"] = S0_map_target_var["mean"]
                    normalization_vars[f"S0_map_target_std_{i}"] = S0_map_target_var["std"]
                    normalization_vars[f"S0_map_target_var_{i}"] = S0_map_target_var["var"]
        else:
            if S0_map_target_vars is not None:
                normalization_vars["S0_map_target_min"] = S0_map_target_vars["min"]
                normalization_vars["S0_map_target_max"] = S0_map_target_vars["max"]
                normalization_vars["S0_map_target_mean"] = S0_map_target_vars["mean"]
                normalization_vars["S0_map_target_std"] = S0_map_target_vars["std"]
                normalization_vars["S0_map_target_var"] = S0_map_target_vars["var"]

        if isinstance(B0_map_init_vars, list):
            if B0_map_init_vars[0] is not None:
                for i, B0_map_init_var in enumerate(B0_map_init_vars):
                    normalization_vars[f"B0_map_init_min_{i}"] = B0_map_init_var["min"]
                    normalization_vars[f"B0_map_init_max_{i}"] = B0_map_init_var["max"]
                    normalization_vars[f"B0_map_init_mean_{i}"] = B0_map_init_var["mean"]
                    normalization_vars[f"B0_map_init_std_{i}"] = B0_map_init_var["std"]
                    normalization_vars[f"B0_map_init_var_{i}"] = B0_map_init_var["var"]
        else:
            if B0_map_init_vars is not None:
                normalization_vars["B0_map_init_min"] = B0_map_init_vars["min"]
                normalization_vars["B0_map_init_max"] = B0_map_init_vars["max"]
                normalization_vars["B0_map_init_mean"] = B0_map_init_vars["mean"]
                normalization_vars["B0_map_init_std"] = B0_map_init_vars["std"]
                normalization_vars["B0_map_init_var"] = B0_map_init_vars["var"]

        if isinstance(B0_map_target_vars, list):
            if B0_map_target_vars[0] is not None:
                for i, B0_map_target_var in enumerate(B0_map_target_vars):
                    normalization_vars[f"B0_map_target_min_{i}"] = B0_map_target_var["min"]
                    normalization_vars[f"B0_map_target_max_{i}"] = B0_map_target_var["max"]
                    normalization_vars[f"B0_map_target_mean_{i}"] = B0_map_target_var["mean"]
                    normalization_vars[f"B0_map_target_std_{i}"] = B0_map_target_var["std"]
                    normalization_vars[f"B0_map_target_var_{i}"] = B0_map_target_var["var"]
        else:
            if B0_map_target_vars is not None:
                normalization_vars["B0_map_target_min"] = B0_map_target_vars["min"]
                normalization_vars["B0_map_target_max"] = B0_map_target_vars["max"]
                normalization_vars["B0_map_target_mean"] = B0_map_target_vars["mean"]
                normalization_vars["B0_map_target_std"] = B0_map_target_vars["std"]
                normalization_vars["B0_map_target_var"] = B0_map_target_vars["var"]

        if isinstance(phi_map_init_vars, list):
            if phi_map_init_vars[0] is not None:
                for i, phi_map_init_var in enumerate(phi_map_init_vars):
                    normalization_vars[f"phi_map_init_min_{i}"] = phi_map_init_var["min"]
                    normalization_vars[f"phi_map_init_max_{i}"] = phi_map_init_var["max"]
                    normalization_vars[f"phi_map_init_mean_{i}"] = phi_map_init_var["mean"]
                    normalization_vars[f"phi_map_init_std_{i}"] = phi_map_init_var["std"]
                    normalization_vars[f"phi_map_init_var_{i}"] = phi_map_init_var["var"]
        else:
            if phi_map_init_vars is not None:
                normalization_vars["phi_map_init_min"] = phi_map_init_vars["min"]
                normalization_vars["phi_map_init_max"] = phi_map_init_vars["max"]
                normalization_vars["phi_map_init_mean"] = phi_map_init_vars["mean"]
                normalization_vars["phi_map_init_std"] = phi_map_init_vars["std"]
                normalization_vars["phi_map_init_var"] = phi_map_init_vars["var"]

        if isinstance(phi_map_target_vars, list):
            if phi_map_target_vars[0] is not None:
                for i, phi_map_target_var in enumerate(phi_map_target_vars):
                    normalization_vars[f"phi_map_target_min_{i}"] = phi_map_target_var["min"]
                    normalization_vars[f"phi_map_target_max_{i}"] = phi_map_target_var["max"]
                    normalization_vars[f"phi_map_target_mean_{i}"] = phi_map_target_var["mean"]
                    normalization_vars[f"phi_map_target_std_{i}"] = phi_map_target_var["std"]
                    normalization_vars[f"phi_map_target_var_{i}"] = phi_map_target_var["var"]
        else:
            if phi_map_target_vars is not None:
                normalization_vars["phi_map_target_min"] = phi_map_target_vars["min"]
                normalization_vars["phi_map_target_max"] = phi_map_target_vars["max"]
                normalization_vars["phi_map_target_mean"] = phi_map_target_vars["mean"]
                normalization_vars["phi_map_target_std"] = phi_map_target_vars["std"]
                normalization_vars["phi_map_target_var"] = phi_map_target_vars["var"]

        return normalization_vars


class GaussianSmoothing(torch.nn.Module):
    """Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed separately for each channel in the
    input using a depthwise convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Union[List[int], int],
        sigma: float,
        dim: int = 2,
        shift: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = None,
    ):
        """Inits :class:`GaussianSmoothing`.

        Parameters
        ----------
        channels : int
            Number of channels in the input tensor.
        kernel_size : Union[Optional[List[int]], int]
            Gaussian kernel size.
        sigma : float
            Gaussian kernel standard deviation.
        dim : int
            Number of dimensions in the input tensor.
        shift : bool
            If True, the gaussian kernel is centered at (kernel_size - 1) / 2.
        fft_centered : bool
            Whether to center the FFT for a real- or complex-valued input.
        fft_normalization : str
            Whether to normalize the FFT output (None, "ortho", "backward", "forward", "none").
        spatial_dims : Sequence[int]
            Spatial dimensions to keep in the FFT.
        """
        super().__init__()

        self.shift = shift
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim

        if isinstance(sigma, float):
            sigma = [sigma] * dim  # type: ignore

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        for size, std, mgrid in zip(
            kernel_size,
            sigma,
            torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij"),
        ):  # type: ignore
            tmp_kernel = 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - (size - 1) / 2) / std) ** 2) / 2)
            kernel = kernel * tmp_kernel

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())  # type: ignore
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))  # type: ignore

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(f"Only 1, 2 and 3 dimensions are supported. Received {dim}.")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`GaussianSmoothing`.

        Parameters
        ----------
        data : torch.Tensor
            Input to apply gaussian filter on.

        Returns
        -------
        torch.Tensor
            Filtered output.
        """
        if self.shift:
            data = data.permute(0, 2, 3, 1)
            data = ifft2(
                torch.fft.fftshift(
                    fft2(
                        torch.view_as_real(data[..., 0] + 1j * data[..., 1]),
                        self.fft_centered,
                        self.fft_normalization,
                        self.spatial_dims,
                    ),
                ),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            ).permute(0, 3, 1, 2)

        x = self.conv(data, weight=self.weight.to(data), groups=self.groups).to(data).detach()

        if self.shift:
            x = x.permute(0, 2, 3, 1)
            x = ifft2(
                torch.fft.fftshift(
                    fft2(
                        torch.view_as_real(x[..., 0] + 1j * x[..., 1]),
                        self.fft_centered,
                        self.fft_normalization,
                        self.spatial_dims,
                    ),
                ),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            ).permute(0, 3, 1, 2)

        return x


class LeastSquaresFitting:
    """Differentiable least square fitting in PyTorch."""

    def __init__(self, device):
        """Inits :class:`LeastSquaresFitting`."""
        super().__init__()
        self.device = device

    @staticmethod
    def lsqrt(A: torch.Tensor, Y: torch.Tensor, reg_factor: float = 0.0) -> torch.Tensor:
        """Differentiable least square solution.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix.
        Y : torch.Tensor
            Echo times matrix.
        reg_factor : float
            Regularization parameter.

        Returns
        -------
        torch.Tensor
            Least square solution.
        """
        q, r = torch.qr(A)
        return torch.inverse(r) @ q.permute(0, 2, 1) @ Y + reg_factor

    @staticmethod
    def lsqrt_pinv(
        A: torch.Tensor, Y: torch.Tensor, reg_factor: float = 0.0  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Differentiable inverse least square solution.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix.
        Y : torch.Tensor
            Echo times matrix.
        reg_factor : float
            Regularization parameter.

        Returns
        -------
        torch.Tensor
            Inverse least square solution.
        """
        if Y.dim() == 2:
            return torch.matmul(torch.inverse(Y), A)
        return torch.bmm(
            torch.matmul(
                torch.inverse(torch.matmul(torch.conj(Y).permute(0, 2, 1), Y)), torch.conj(Y).permute(0, 2, 1)
            ),
            A,
        )[..., 0]


def R2star_B0_S0_phi_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    anatomy_mask: torch.Tensor,
    scaling_factor: float = 1e-3,
    fft_centered: bool = False,
    fft_normalization: str = "backward",
    spatial_dims: Sequence[int] = None,
):
    """Maps the prediction to R2*, B0, and S0 maps.

    Parameters
    ----------
    prediction : torch.Tensor
        The prediction of the model.
    TEs : Union[Optional[List[float]], float]
        The TEs of the images.
    anatomy_mask : torch.Tensor
        The anatomy mask of the images.
    scaling_factor : float
        The scaling factor to apply to the prediction.
    fft_centered : bool
        Whether to center the FFT for a real- or complex-valued input.
    fft_normalization : str
        Whether to normalize the FFT output (None, "ortho", "backward", "forward", "none").
    spatial_dims : Sequence[int]
        Spatial dimensions to keep in the FFT.

    Returns
    -------
    R2star : torch.Tensor
        The R2* map.
    B0 : torch.Tensor
        The B0 map.
    S0 : torch.Tensor
        The S0 map.
    phi : torch.Tensor
        The phi map.
    """
    R2star_map = R2star_mapping(prediction, TEs, scaling_factor=scaling_factor)
    B0_map = -B0_phi_mapping(
        prediction,
        TEs,
        anatomy_mask,
        scaling_factor=scaling_factor,
        fft_centered=fft_centered,
        fft_normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )[0]
    S0_map, phi_map = S0_mapping(
        prediction,
        TEs,
        R2star_map,
        B0_map,
        scaling_factor=scaling_factor,
        fft_centered=fft_centered,
        fft_normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )

    return R2star_map, S0_map, B0_map, phi_map


def R2star_mapping(prediction: torch.Tensor, TEs: Union[Optional[List[float]], float], scaling_factor: float = 1e-3):
    """R2* map and S0 map estimation for multi-echo GRE from stored magnitude image files acquired at multiple TEs.

    Parameters
    ----------
    prediction : torch.Tensor
        The prediction of the model.
    TEs : Union[Optional[List[float]], float]
        The TEs of the images.
    scaling_factor : float
        The scaling factor.

    Returns
    -------
    R2star : torch.Tensor
        The R2* map.
    S0 : torch.Tensor
        The S0 map.
    """
    prediction = torch.view_as_complex(prediction)
    prediction = torch.abs(prediction / torch.max(torch.abs(prediction))) + 1e-8

    prediction_flatten = torch.flatten(prediction, start_dim=1, end_dim=-1).detach().cpu()
    log_prediction_flatten = torch.log(prediction_flatten)
    sqrt_prediction_flatten = torch.sqrt(prediction_flatten)

    TEs = torch.tensor(TEs).to(prediction_flatten)
    TEs = TEs * scaling_factor  # type: ignore

    R2star_map = torch.zeros([prediction_flatten.shape[1]])
    for i in range(prediction_flatten.shape[1]):
        R2star_map[i], _ = torch.from_numpy(
            np.polyfit(TEs, log_prediction_flatten[:, i], 1, w=sqrt_prediction_flatten[:, i])
        ).to(prediction)

    R2star_map = torch.reshape(-R2star_map, prediction.shape[1:4])
    return R2star_map


def B0_phi_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    anatomy_mask: torch.Tensor,
    scaling_factor: float = 1e-3,
    fft_centered: bool = False,
    fft_normalization: str = "backward",
    spatial_dims: Sequence[int] = None,
):
    """B0 map and Phi map estimation for multi-echo GRE from stored magnitude image files acquired at multiple TEs.

    Parameters
    ----------
    prediction : torch.Tensor
        The prediction of the model.
    TEs : Union[Optional[List[float]], float]
        The TEs of the images.
    anatomy_mask : torch.Tensor
        The anatomy mask of the images.
    scaling_factor : float
        The scaling factor.
    fft_centered : bool
        Whether to center the FFT for a real- or complex-valued input.
    fft_normalization : str
        Whether to normalize the FFT output (None, "ortho", "backward", "forward", "none").
    spatial_dims : Sequence[int]
        Spatial dimensions to keep in the FFT.

    Returns
    -------
    B0 : torch.Tensor
        The B0 map.
    phi : torch.Tensor
        The phi map.
    """
    lsq = LeastSquaresFitting(device=prediction.device)

    TEnotused = 3  # if fully_sampled else 3
    TEs = torch.tensor(TEs)

    # brain_mask is used only for descale of phase difference (so that phase_diff is in between -2pi and 2pi)
    anatomy_mask_descale = anatomy_mask
    shape = prediction.shape

    # apply gaussian blur with radius r to
    smoothing = GaussianSmoothing(
        channels=2,
        kernel_size=9,
        sigma=scaling_factor,
        dim=2,
        fft_centered=fft_centered,
        fft_normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )
    prediction = prediction.unsqueeze(1).permute([0, 1, 4, 2, 3])  # add a dummy batch dimension
    for i in range(prediction.shape[0]):
        prediction[i] = smoothing(F.pad(prediction[i], (4, 4, 4, 4), mode="reflect"))
    prediction = prediction.permute([0, 1, 3, 4, 2]).squeeze(1)

    prediction = ifft2(
        torch.fft.fftshift(fft2(prediction, fft_centered, fft_normalization, spatial_dims), dim=(1, 2)),
        fft_centered,
        fft_normalization,
        spatial_dims,
    )

    phase = torch.angle(torch.view_as_complex(prediction))

    # unwrap phases
    phase_unwrapped = torch.zeros_like(phase)

    body_part_mask = anatomy_mask.clone()
    body_part_mask_np = np.invert(body_part_mask.cpu().detach().numpy() > 0.5)

    # loop over echo times
    for i in range(phase.shape[0]):
        phase_unwrapped[i] = torch.from_numpy(
            unwrap_phase(np.ma.array(phase[i].detach().cpu().numpy(), mask=body_part_mask_np)).data
        ).to(prediction)

    phase_diff_set = []
    TE_diff = []

    # obtain phase differences and TE differences
    for i in range(phase_unwrapped.shape[0] - TEnotused):
        phase_diff_set.append(torch.flatten(phase_unwrapped[i + 1] - phase_unwrapped[i]))
        phase_diff_set[i] = (
            phase_diff_set[i]
            - torch.round(
                torch.abs(
                    torch.sum(phase_diff_set[i] * torch.flatten(anatomy_mask_descale))
                    / torch.sum(anatomy_mask_descale)
                    / 2
                    / np.pi
                )
            )
            * 2
            * np.pi
        )
        TE_diff.append(TEs[i + 1] - TEs[i])  # type: ignore

    phase_diff_set = torch.stack(phase_diff_set, 0)
    TE_diff = torch.stack(TE_diff, 0).to(prediction)

    # least squares fitting to obtain phase map
    B0_map_tmp = lsq.lsqrt_pinv(
        phase_diff_set.unsqueeze(2).permute(1, 0, 2), TE_diff.unsqueeze(1) * scaling_factor  # type: ignore
    )
    B0_map = B0_map_tmp.reshape(shape[-3], shape[-2])
    B0_map = B0_map * torch.abs(body_part_mask)

    # obtain phi map
    phi_map = (phase_unwrapped[0] - scaling_factor * TEs[0] * B0_map).squeeze(0)  # type: ignore

    return B0_map.squeeze(0).to(prediction), phi_map.to(prediction)


def S0_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    R2star_map: torch.Tensor,
    B0_map: torch.Tensor,
    scaling_factor: float = 1e-3,
    fft_centered: bool = False,
    fft_normalization: str = "backward",
    spatial_dims: Sequence[int] = None,
):
    """Complex S0 mapping.

    Parameters
    ----------
    prediction : torch.Tensor
        The prediction of the model.
    TEs : Union[Optional[List[float]], float]
        The TEs of the images.
    R2star_map : torch.Tensor
        The R2* map.
    B0_map : torch.Tensor
        The B0 map.
    scaling_factor : float
        The scaling factor.
    fft_centered : bool
        Whether to center the FFT for a real- or complex-valued input.
    fft_normalization : str
        Whether to normalize the FFT output (None, "ortho", "backward", "forward", "none").
    spatial_dims : Sequence[int]
        Spatial dimensions to keep in the FFT.

    Returns
    -------
    S0 : torch.Tensor
        The S0 map.
    """
    lsq = LeastSquaresFitting(device=prediction.device)

    prediction = torch.view_as_complex(prediction)
    prediction_flatten = prediction.reshape(prediction.shape[0], -1)

    TEs = torch.tensor(TEs).to(prediction)

    R2star_B0_complex_map = R2star_map.to(prediction) + 1j * B0_map.to(prediction)
    R2star_B0_complex_map_flatten = R2star_B0_complex_map.flatten()

    TEs_r2 = TEs[0:4].unsqueeze(1) * -R2star_B0_complex_map_flatten  # type: ignore

    S0_map = lsq.lsqrt_pinv(
        prediction_flatten.permute(1, 0).unsqueeze(2), torch.exp(scaling_factor * TEs_r2.permute(1, 0).unsqueeze(2))
    )

    S0_map = torch.view_as_real(S0_map.reshape(prediction.shape[1:]))

    S0_map = ifft2(
        torch.fft.fftshift(fft2(S0_map, fft_centered, fft_normalization, spatial_dims), dim=(0, 1)),
        fft_centered,
        fft_normalization,
        spatial_dims,
    )

    S0_map = torch.view_as_complex(S0_map).squeeze(-1)

    return torch.abs(S0_map), torch.angle(S0_map)
