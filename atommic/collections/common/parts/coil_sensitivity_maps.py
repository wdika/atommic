# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/algorithms/mri_algorithms.py

from abc import ABC
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from atommic.collections.common.parts.fft import ifft2
from atommic.collections.common.parts.utils import crop_to_acs


class MaximumEigenvaluePowerMethod(ABC):
    """A class for solving the maximum eigenvalue problem using the Power Method.

    The Power Method is an iterative algorithm that can be used to find the largest eigenvalue of a matrix. The
    algorithm is initialized with a random vector and iteratively updates the vector by multiplying it with the matrix
    and normalizing it. The algorithm converges to the eigenvector corresponding to the largest eigenvalue. The
    largest eigenvalue is then estimated by taking the dot product of the eigenvector with the matrix.
    """

    def __init__(
        self,
        forward_operator: Callable,
        norm_func: Optional[Callable] = None,
        max_iter: int = 30,
    ):
        """Inits :class:`MaximumEigenvaluePowerMethod`.

        Parameters
        ----------
        forward_operator : Callable
            The forward operator for the problem.
        norm_func : Callable, optional
            An optional function for normalizing the eigenvector. Default is ``None``.
        max_iter : int, optional
            Maximum number of iterations to run the algorithm. Default is ``30``.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.norm_func = norm_func
        self.max_iter = max_iter
        self.iter = 0

    def _update(self) -> None:
        """Perform a single update step of the maximum eigenvalue guess and corresponding eigenvector."""
        y = self.forward_operator(self.x)  # type: ignore
        if self.norm_func is None:
            self.max_eig = (y * self.x.conj()).sum() / (self.x * self.x.conj()).sum()  # type: ignore
        else:
            self.max_eig = self.norm_func(y)
        self.x = y / self.max_eig

    def _done(self) -> bool:
        """Check if the algorithm is done."""
        return self.iter >= self.max_iter

    def _fit(self, x: torch.Tensor) -> None:
        """Sets initial maximum eigenvector guess."""
        self.x = x

    def update(self) -> None:
        """Update the algorithm's parameters and increment the iteration count."""
        self._update()
        self.iter += 1

    def done(self) -> bool:
        """Check if the algorithm has converged."""
        return self._done()

    def fit(self, *args, **kwargs) -> None:
        """Fit the algorithm.

        Parameters
        ----------
        *args : tuple
            Tuple of arguments for `_fit` method.
        **kwargs : dict
            Keyword arguments for `_fit` method.
        """
        self._fit(*args, **kwargs)
        while not self.done():
            self.update()


class EspiritCalibration(ABC, torch.nn.Module):
    """Estimates sensitivity maps estimated with the ESPIRIT calibration method as described in [1]_.

    We adapted code for ESPIRIT method adapted from [2]_.

    References
    ----------

    .. [1] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. ESPIRiT--an eigenvalue
        approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magn Reson Med. 2014 Mar;71(3):990-1001.
        doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.
    .. [2] https://github.com/mikgroup/sigpy/blob/1817ff849d34d7cbbbcb503a1b310e7d8f95c242/sigpy/mri/app.py#L388-L491
    """

    def __init__(
        self,
        threshold: float = 0.05,
        kernel_size: int = 6,
        crop: float = 0.95,
        max_iter: int = 100,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = (-2, -1),
    ):
        """Inits :class:`EstimateSensitivityMap`.

        Parameters
        ----------
        threshold: float, optional
            Threshold for the calibration matrix. Default: 0.05.
        kernel_size: int, optional
            Kernel size for the calibration matrix. Default: 6.
        crop: float, optional
            Output eigenvalue cropping threshold. Default: 0.95.
        max_iter: int, optional
            Power method iterations. Default: 30.
        fft_centered: bool, optional
            Whether to center the FFT. Default is ``False``.
        fft_normalization: str, optional
            Normalization to apply to the FFT. Default is ``"backward"``.
        spatial_dims: Sequence[int], optional
            Spatial dimensions of the input. Default is ``(-2, -1)``.
        """
        super().__init__()

        self.threshold = threshold
        self.kernel_size = kernel_size
        self.crop = crop
        self.max_iter = max_iter
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def calculate_sensitivity_map(self, acs_mask: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
        """Calculates sensitivity map given as input the `acs_mask` and the `k-space`.

        Parameters
        ----------
        acs_mask : torch.Tensor
            Autocalibration mask.
        kspace : torch.Tensor
            K-space.

        Returns
        -------
        sensitivity_map : torch.Tensor
            Coil sensitivity maps.
        """
        ndim = kspace.ndim - 2
        spatial_size = kspace.shape[1:-1]

        # Used in case the k-space is padded (e.g. for batches)
        non_padded_dim = kspace.clone().sum(dim=tuple(range(1, kspace.ndim))).bool()

        num_coils = non_padded_dim.sum()
        acs_kspace_cropped = torch.view_as_complex(crop_to_acs(acs_mask.squeeze(), kspace[non_padded_dim]))

        # Get calibration matrix.
        calibration_matrix = (
            torch.nn.functional.unfold(acs_kspace_cropped.unsqueeze(0), kernel_size=self.kernel_size, stride=1)
            .transpose(1, 2)
            .to(acs_kspace_cropped.device)
            .reshape(
                num_coils,
                *(np.array(acs_kspace_cropped.shape[1:3]) - self.kernel_size + 1),
                *([self.kernel_size] * ndim),
            )
        )
        calibration_matrix = calibration_matrix.reshape(num_coils, -1, self.kernel_size**ndim)
        calibration_matrix = calibration_matrix.permute(1, 0, 2)
        calibration_matrix = calibration_matrix.reshape(-1, num_coils * self.kernel_size**ndim)

        _, s, vh = torch.linalg.svd(calibration_matrix, full_matrices=True)
        vh = torch.where(s > (self.threshold * s.max()), vh, torch.zeros_like(vh))

        # Get kernels
        num_kernels = vh.shape[0]
        kernels = vh.reshape([num_kernels, num_coils] + [self.kernel_size] * ndim)

        # Get covariance matrix in image domain
        covariance = torch.zeros(
            spatial_size[::-1] + (num_coils, num_coils),
            dtype=kernels.dtype,
            device=kernels.device,
        )
        for kernel in kernels:
            pad_h, pad_w = (
                spatial_size[0] - self.kernel_size,
                spatial_size[1] - self.kernel_size,
            )
            pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            kernel_padded = torch.nn.functional.pad(kernel, pad)

            img_kernel = ifft2(kernel_padded, self.fft_centered, self.fft_normalization, self.spatial_dims)
            img_kernel = torch.view_as_complex(img_kernel)

            aH = img_kernel.permute(*torch.arange(img_kernel.ndim - 1, -1, -1)).unsqueeze(-1)
            a = aH.transpose(-1, -2).conj()
            covariance = covariance + aH @ a

        covariance = covariance * (np.prod(spatial_size) / self.kernel_size**ndim)
        sensitivity_map = torch.ones(
            (*spatial_size[::-1], num_coils, 1),
            dtype=kernels.dtype,
            device=kernels.device,
        )

        def forward(x):
            return covariance @ x

        def normalize(x):
            return (x.abs() ** 2).sum(dim=-2, keepdims=True) ** 0.5

        power_method = MaximumEigenvaluePowerMethod(forward, max_iter=self.max_iter, norm_func=normalize)
        power_method.fit(x=sensitivity_map)

        temp_sensitivity_map = power_method.x.squeeze(-1)
        temp_sensitivity_map = temp_sensitivity_map.permute(
            *torch.arange(temp_sensitivity_map.ndim - 1, -1, -1)
        ).squeeze(-1)
        temp_sensitivity_map = temp_sensitivity_map * temp_sensitivity_map.conj() / temp_sensitivity_map.abs()

        max_eig = power_method.max_eig.squeeze()
        max_eig = max_eig.permute(*torch.arange(max_eig.ndim - 1, -1, -1))
        temp_sensitivity_map = temp_sensitivity_map * (max_eig > self.crop)

        sensitivity_map = torch.zeros_like(kspace, device=kspace.device, dtype=kspace.dtype)
        sensitivity_map[non_padded_dim] = torch.view_as_real(temp_sensitivity_map)
        return sensitivity_map

    def forward(self, acs_mask: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`EspiritCalibration`.

        Parameters
        ----------
        acs_mask : torch.Tensor
            Autocalibration mask.
        kspace : torch.Tensor
            K-space.

        Returns
        -------
        sensitivity_map : torch.Tensor
            Coil sensitivity maps.
        """
        return self.calculate_sensitivity_map(acs_mask, kspace)
