# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from atommic.collections.common.parts.fft import ifft2
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
from atommic.collections.common.parts.utils import is_none, to_tensor
from atommic.collections.motioncorrection.parts.motionsimulation import MotionSimulation

__all__ = ["RSMRIDataTransforms"]


class RSMRIDataTransforms:
    """Data transforms for accelerated-MRI reconstruction and MRI segmentation.

    Returns
    -------
    RSMRIDataTransforms
        Preprocessed data for accelerated-MRI reconstruction and MRI segmentation.
    """

    def __init__(
        self,
        complex_data: bool = True,
        segmentation_mode: str = "multilabel",
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
        consecutive_slices: int = 1,
        use_seed: bool = True,
    ):
        """Inits :class:`RSMRIDataTransforms`.

        Parameters
        ----------
        complex_data : bool, optional
            Whether to use complex data. If ``False`` the data are assumed to be magnitude only. Default is ``True``.
        segmentation_mode: str, optional
            Defines the segmentation labels model, either ``multiclass``or ``multilabel``. In ``multiclass`` mode, only
            one class is assigned per voxel. In ``multilabel`` mode, multiple (overlapping) classes are allowed per
            voxel. Default is ``multilabel``.
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
        self.complex_data = complex_data

        self.dataset_format = dataset_format

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim - 1 if dimensionality == 2 and not is_none(coil_dim) else coil_dim

        if not self.complex_data:
            if not is_none(coil_combination_method):
                raise ValueError("Coil combination method for non-complex data should be None.")
            if not is_none(mask_func):
                raise ValueError("Mask function for non-complex data should be None.")
            self.kspace_crop = kspace_crop
            if self.kspace_crop:
                raise ValueError("K-space crop for non-complex data should be False.")
            if not is_none(kspace_zero_filling_size):
                raise ValueError("K-space zero filling size for non-complex data should be None.")
            if not is_none(coil_dim):
                raise ValueError("Coil dimension for non-complex data should be None.")
            if apply_prewhitening:
                raise ValueError("Prewhitening for non-complex data cannot be applied.")
            if apply_gcc:
                raise ValueError("GCC for non-complex data cannot be applied.")
            if apply_random_motion:
                raise ValueError("Random motion for non-complex data cannot be applied.")
        else:
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
                shift_mask=shift_mask,
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

            self.kspace_crop = kspace_crop
            self.crop_before_masking = crop_before_masking

            self.coil_combination_method = coil_combination_method

            self.prewhitening = Composer([self.prewhitening])  # type: ignore
            self.coils_shape_transforms = Composer(
                [
                    self.gcc,  # type: ignore
                    self.kspace_zero_filling,  # type: ignore
                ]
            )
            self.random_motion = Composer([self.random_motion])  # type: ignore

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

        self.crop_normalize = Composer(
            [
                self.cropping,  # type: ignore
                self.normalization,  # type: ignore
            ]
        )
        self.consecutive_slices = consecutive_slices
        self.segmentation_mode = segmentation_mode

        self.cropping = Composer([self.cropping])  # type: ignore
        self.normalization = Composer([self.normalization])  # type: ignore

        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        imspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        initial_prediction_reconstruction: np.ndarray,
        segmentation_labels: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_idx: int,
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        Union[List[torch.Tensor], torch.Tensor],
        torch.tensor,
        torch.tensor,
        str,
        int,
        Union[List[Union[float, torch.Tensor, Any]]],
        Dict,
    ]:
        """Calls :class:`RSMRIDataTransforms`.

        Parameters
        ----------
        kspace : np.ndarray
            The fully-sampled kspace, if exists. Otherwise, the subsampled kspace.
        imspace : np.ndarray
            The image space for segmentation, if exists.
        sensitivity_map : np.ndarray
            The coil sensitivity map.
        mask : np.ndarray
            The subsampling mask, if exists, meaning that the data are either prospectively undersampled or the mask is
            stored and loaded.
        initial_prediction_reconstruction : np.ndarray
            The initial prediction, if exists. Otherwise, it will be estimated with the chosen coil combination method.
        segmentation_labels : np.ndarray
            The segmentation labels.
        attrs : Dict
            The attributes, if stored in the data.
        fname : str
            The file name.
        slice_idx : int
            The slice index.
        """
        initial_prediction_reconstruction = (
            to_tensor(initial_prediction_reconstruction)
            if initial_prediction_reconstruction is not None and initial_prediction_reconstruction.size != 0
            else torch.tensor([])
        )

        if not self.complex_data:
            kspace = torch.empty([])
            kspace_pre_normalization_vars = None
            sensitivity_map = torch.empty([])
            sensitivity_pre_normalization_vars = None
            masked_kspace = torch.empty([])
            mask = torch.empty([])
            acc = torch.empty([])
            (
                initial_prediction_reconstruction,
                initial_prediction_pre_normalization_vars,
            ) = self.__initialize_prediction__(imspace, kspace, sensitivity_map)

            if "min" in attrs:
                initial_prediction_pre_normalization_vars["min"] = attrs["min"]
            if "max" in attrs:
                initial_prediction_pre_normalization_vars["max"] = attrs["max"]
            if "mean" in attrs:
                initial_prediction_pre_normalization_vars["mean"] = attrs["mean"]
            if "std" in attrs:
                initial_prediction_pre_normalization_vars["std"] = attrs["std"]

            noise_prediction_pre_normalization_vars = None
            target_reconstruction = initial_prediction_reconstruction
            target_pre_normalization_vars = initial_prediction_pre_normalization_vars
        else:
            kspace, masked_kspace, mask, kspace_pre_normalization_vars, acc = self.__process_kspace__(  # type: ignore
                kspace, mask, attrs, fname
            )
            sensitivity_map, sensitivity_pre_normalization_vars = self.__process_coil_sensitivities_map__(
                sensitivity_map, kspace
            )
            target_reconstruction, target_pre_normalization_vars = self.__initialize_prediction__(
                torch.empty([]), kspace, sensitivity_map
            )
            target_prediction_pre_normalization_vars = None
            if self.n2r and len(masked_kspace) > 1:
                (
                    initial_prediction_reconstruction,
                    initial_prediction_pre_normalization_vars,
                ) = self.__initialize_prediction__(
                    initial_prediction_reconstruction, masked_kspace[0], sensitivity_map
                )
                if isinstance(masked_kspace, list) and not masked_kspace[1][0].dim() < 2:
                    noise_prediction, noise_prediction_pre_normalization_vars = self.__initialize_prediction__(
                        None, masked_kspace[1], sensitivity_map
                    )
                else:
                    noise_prediction = torch.tensor([])
                    noise_prediction_pre_normalization_vars = None
                initial_prediction_reconstruction = [initial_prediction_reconstruction, noise_prediction]
            else:
                (
                    initial_prediction_reconstruction,
                    initial_prediction_pre_normalization_vars,
                ) = self.__initialize_prediction__(initial_prediction_reconstruction, masked_kspace, sensitivity_map)
                noise_prediction_pre_normalization_vars = None

            if self.unsupervised_masked_target:
                target_reconstruction, target_prediction_pre_normalization_vars = (
                    initial_prediction_reconstruction,
                    noise_prediction_pre_normalization_vars,
                )
            else:
                target_reconstruction, target_prediction_pre_normalization_vars = self.__initialize_prediction__(
                    None if self.ssdu else target_prediction_pre_normalization_vars, kspace, sensitivity_map
                )

        if not is_none(segmentation_labels) and segmentation_labels.ndim > 1:
            segmentation_labels = self.cropping(torch.from_numpy(segmentation_labels))  # type: ignore
        else:
            segmentation_labels = torch.empty([])

        # if segmentation_labels is Bool type, convert to float
        if segmentation_labels.dtype == torch.bool:
            segmentation_labels = segmentation_labels.float()
        segmentation_labels = torch.abs(segmentation_labels)

        if self.segmentation_mode == "multiclass":
            # Ensures background class is explicitly added when performing multiclass segmentation -> final total
            # number of classes should be N + 1
            if self.consecutive_slices > 1 and not torch.all(torch.sum(segmentation_labels[0], dim=0) == 1):
                segmentation_labels_bg = torch.zeros(
                    (segmentation_labels.shape[0], segmentation_labels.shape[2], segmentation_labels.shape[3])
                )
                segmentation_labels_new = torch.zeros(
                    (
                        segmentation_labels.shape[0],
                        segmentation_labels.shape[1] + 1,
                        segmentation_labels.shape[2],
                        segmentation_labels.shape[3],
                    )
                )
                for i in range(target_reconstruction.shape[0]):
                    idx_background = torch.where(torch.sum(segmentation_labels[i], dim=0) == 0)
                    segmentation_labels_bg[i][idx_background] = 1
                    segmentation_labels_new[i] = torch.concat(
                        (segmentation_labels_bg[i].unsqueeze(0), segmentation_labels[i]), dim=0
                    )
                segmentation_labels = segmentation_labels_new
            elif not torch.all(torch.sum(segmentation_labels, dim=0) == 1):
                segmentation_labels_bg = torch.zeros((segmentation_labels.shape[-2], segmentation_labels.shape[-1]))
                idx_background = torch.where(torch.sum(segmentation_labels, dim=0) == 0)
                segmentation_labels_bg[idx_background] = 1
                segmentation_labels = torch.concat((segmentation_labels_bg.unsqueeze(0), segmentation_labels), dim=0)

        attrs.update(
            self.__parse_normalization_vars__(
                kspace_pre_normalization_vars,
                sensitivity_pre_normalization_vars,
                initial_prediction_pre_normalization_vars,
                noise_prediction_pre_normalization_vars,
                target_pre_normalization_vars,
            )
        )
        attrs["fname"] = fname
        attrs["slice_idx"] = slice_idx

        return (
            kspace,
            masked_kspace,
            sensitivity_map,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            segmentation_labels,
            fname,
            slice_idx,
            acc,
            attrs,
        )

    def __repr__(self) -> str:
        """Representation of :class:`RSMRIDataTransforms`."""
        return (
            f"Preprocessing transforms initialized for {self.__class__.__name__}: "
            f"prewhitening = {self.prewhitening}, "
            f"masking = {self.masking}, "
            f"SSDU masking = {self.ssdu_masking}, "
            f"kspace zero-filling = {self.kspace_zero_filling}, "
            f"cropping = {self.cropping}, "
            f"normalization = {self.normalization}, "
        )

    def __str__(self) -> str:
        """String representation of :class:`RSMRIDataTransforms`."""
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

        kspace = self.coils_shape_transforms(kspace, apply_backward_transform=True)
        kspace = self.prewhitening(kspace)  # type: ignore

        if self.crop_before_masking:
            kspace = self.cropping(kspace, apply_backward_transform=not self.kspace_crop)  # type: ignore

        masked_kspace, mask, acc = self.masking(
            self.random_motion(kspace),  # type: ignore
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

    def __process_coil_sensitivities_map__(
        self, sensitivity_map: np.ndarray, kspace: torch.Tensor
    ) -> Union[torch.Tensor, Dict]:
        """Preprocesses the coil sensitivities map.

        Parameters
        ----------
        sensitivity_map : np.ndarray
            The coil sensitivities map.
        kspace : torch.Tensor
            The kspace.

        Returns
        -------
        List[torch.Tensor, Dict]
            The preprocessed coil sensitivities map and the normalization variables.
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

    def __initialize_prediction__(
        self, prediction: Union[np.ndarray, None], kspace: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Dict]:
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
        Tuple[Union[List[torch.Tensor], torch.Tensor], Dict]
            The initialized prediction, either a list of coil-combined images or a single coil-combined image and the
            pre-normalization variables (min, max, mean, std).
        """
        if is_none(prediction) or prediction.ndim < 2 or isinstance(kspace, list):  # type: ignore
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
        return prediction, pre_normalization_vars  # type: ignore

    def __parse_normalization_vars__(  # noqa: MC0001
        self, kspace_vars, sensitivity_vars, prediction_vars, noise_prediction_vars, target_vars
    ) -> Dict:
        """
        Parses the normalization variables and returns a unified dictionary.

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

        Returns
        -------
        Dict
            The normalization variables.
        """
        normalization_vars = {}

        if self.complex_data:
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

        return normalization_vars
