# coding=utf-8
__author__ = "Dimitris Karkalousos"

import contextlib
from typing import List, Optional, Sequence, Tuple, Union

import numba as nb
import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    """Temporarily set the seed of a numpy random number generator.

    Parameters
    ----------
    rng : np.random.Generator
        The numpy random number generator to modify.
    seed : Optional[Union[int, Tuple[int, ...]]], optional
        The seed to set, by default None.
    """
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """MaskFunc is an abstract base class for creating sub-sampling masks.

    This class is used to create a mask for MRI data that can be used to randomly under-sample the k-space data. The
    mask is created by retaining a specified fraction of the low-frequency columns and setting the rest to zero. The
    fraction of low-frequency columns to retain and the amount of under-sampling can be specified at initialization.

    Examples
    --------
    >>> from atommic.collections.common.data.subsample import MaskFunc
    >>> mask_func = MaskFunc(center_fractions=[0.08, 0.04], accelerations=[4, 8])
    >>> mask_func.choose_acceleration()
    (0.08, 4)
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """Inits :class:`MaskFunc`.

        Parameters
        ----------
        center_fractions : Sequence[float]
            Fraction of low-frequency columns to be retained. If multiple values are provided, then one of these
            numbers is chosen uniformly each time. For 2D setting this value corresponds to setting the
            Full-Width-Half-Maximum.
        accelerations : Sequence[int]
            Amount of under-sampling. This should have the same length as center_fractions. If multiple values are
            provided, then one of these is chosen uniformly each time.
        """
        super().__init__()
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        partial_fourier_percentage: Optional[float] = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Calls :class:`MaskFunc`.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the mask to be created. The shape should have at least 3 dimensions. Same as the shape of the
            input k-space data.
        seed : int or tuple of ints, optional
            Seed for the random number generator. Default is ``None``.
        partial_fourier_percentage : float, optional
            Percentage of the low-frequency columns to be retained. Default is ``0.0``.
        """
        raise NotImplementedError

    def choose_acceleration(self) -> Tuple[float, int]:
        """Chooses an acceleration factor and center fractions from a list of multiple values.

        Returns
        -------
        Tuple[float, int]
            A tuple of the center fraction and the acceleration factor.
        """
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]
        return center_fraction, acceleration


class Equispaced1DMaskFunc(MaskFunc):
    r"""Equispaced1DMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N columns, the mask
    picks out:

        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to low-frequencies.

        2. The other columns are selected with equal spacing at a proportion that reaches the desired acceleration
        rate taking into consideration the number of low frequencies. This ensures that the expected number of
        columns selected is equal to (N / acceleration).


    It is possible to use multiple center_fractions and accelerations, in which case one possible (center_fraction,
    acceleration) is chosen uniformly at random each time the Equispaced1DMaskFunc object is called.

    Note that this function may not give equispaced samples \
    (documented in https://github.com/facebookresearch/fastMRI/issues/54), which will require modifications to
    standard GRAPPA approaches. Nonetheless, this aspect of the function has been preserved to match the public
    multicoil data.

    Examples
    --------
    >>> import torch
    >>> from atommic.collections.common.data.subsample import Equispaced1DMaskFunc
    >>> mask_func = Equispaced1DMaskFunc(center_fractions=[0.08, 0.04], accelerations=[4, 8])
    >>> mask_func.choose_acceleration()
    (0.08, 4)
    >>> kspace = torch.randn(1, 1, 640, 368)
    >>> mask, acceleration = mask_func(kspace.shape)
    >>> mask.shape
    torch.Size([1, 1, 1, 368])
    >>> acceleration
    4
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        partial_fourier_percentage: Optional[float] = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Calls :class:`Equispaced1DMaskFunc`.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the mask to be created. The shape should have at least 3 dimensions. Same as the shape of the
            input k-space data.
        seed : int or tuple of ints, optional
            Seed for the random number generator. Default is ``None``.
        partial_fourier_percentage : float, optional
            Percentage of the low-frequency columns to be retained. Default is ``0.0``.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple of the generated mask and the acceleration factor.

        Raises
        ------
        ValueError
            If the `shape` parameter has less than 3 dimensions.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = torch.div((num_cols - num_low_freqs + 1), 2, rounding_mode="trunc").item()
            mask[pad : pad + num_low_freqs] = True

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

            if partial_fourier_percentage != 0:
                mask[:, : int(np.round(mask.shape[1] * partial_fourier_percentage))] = 0.0

        return mask, acceleration


class Equispaced2DMaskFunc(MaskFunc):
    """Same as Equispaced1DMaskFunc, but for 2D k-space data.

    .. note::
        See ..class::`atommic.collections.common.data.subsample.Equispaced1DMaskFunc` for more details.

    Examples
    --------
    >>> import torch
    >>> from atommic.collections.common.data.subsample import Equispaced2DMaskFunc
    >>> mask_func = Equispaced2DMaskFunc(center_fractions=[0.08, 0.04], accelerations=[4, 8])
    >>> mask_func.choose_acceleration()
    (0.08, 4)
    >>> kspace = torch.randn(1, 1, 640, 368)
    >>> mask, acceleration = mask_func(kspace.shape)
    >>> mask.shape
    torch.Size([1, 1, 640, 368])
    >>> acceleration
    4
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        partial_fourier_percentage: Optional[float] = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Calls :class:`Equispaced2DMaskFunc`.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the mask to be created. The shape should have at least 3 dimensions. Same as the shape of the
            input k-space data.
        seed : int or tuple of ints, optional
            Seed for the random number generator. Default is ``None``.
        partial_fourier_percentage : float, optional
            Percentage of the low-frequency columns to be retained. Default is ``0.0``.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple containing the mask and the acceleration factor.
            The mask is a `torch.Tensor` of the same shape as the input shape.
            The acceleration factor is an `int` that represents the number of samples taken in the k-space.

        Raises
        ------
        ValueError
            If the `shape` parameter has less than 3 dimensions.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()

            acceleration = int(acceleration / 2)
            center_fraction = center_fraction / 2

            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            num_rows = shape[-3]
            num_high_freqs = int(round(num_rows * center_fraction))

            # create the mask
            mask = np.zeros([num_rows, num_cols], dtype=np.float32)

            pad_cols = torch.div((num_cols - num_low_freqs + 1), 2, rounding_mode="trunc").item()
            pad_rows = torch.div((num_rows - num_high_freqs + 1), 2, rounding_mode="trunc").item()
            mask[pad_rows : pad_rows + num_high_freqs, pad_cols : pad_cols + num_low_freqs] = True

            for i in np.arange(0, num_rows, acceleration):
                for j in np.arange(0, num_cols, acceleration):
                    mask[int(i), int(j)] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask_shape[-3] = num_rows
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

            if partial_fourier_percentage != 0:
                mask[:, : int(np.round(mask.shape[1] * partial_fourier_percentage))] = 0.0

        return mask, acceleration * 2


class Gaussian1DMaskFunc(MaskFunc):
    """Same as Gaussian2DMaskFunc, but for 1D k-space data.

    .. note::
        See ..class::`atommic.collections.common.data.subsample.Gaussian2DMaskFunc` for more details.

    Examples
    --------
    >>> import torch
    >>> from atommic.collections.common.data.subsample import Gaussian1DMaskFunc
    >>> mask_func = Gaussian1DMaskFunc(center_fractions=[0.7, 0.7], accelerations=[4, 8])
    >>> mask_func.choose_acceleration()
    (0.7, 4)
    >>> kspace = torch.randn(1, 1, 640, 368)
    >>> mask, acceleration = mask_func(kspace.shape)
    >>> mask.shape
    torch.Size([1, 1, 1, 368])
    >>> acceleration
    4
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        partial_fourier_percentage: Optional[float] = 0.0,
        center_scale: Optional[float] = 0.02,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Calls :class:`Gaussian1DMaskFunc`.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the mask to be created. The shape should have at least 3 dimensions. Same as the shape of the
            input k-space data.
        seed : int or tuple of ints, optional
            Seed for the random number generator. Default is ``None``.
        partial_fourier_percentage : float, optional
            Percentage of the low-frequency columns to be retained. Default is ``0.0``.
        center_scale : float, optional
            For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse
            of which the half-axes will be set to the given `center_scale` percentage of the fully sampled region.
            Default is ``0.02``.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple of the generated mask and the acceleration factor.
        """
        with temp_seed(self.rng, seed):
            dims = [1 for _ in shape]
            self.shape = tuple(shape[-3:-1])
            self.shape = (self.shape[1], self.shape[0])
            dims[-2] = self.shape[-2]

            full_width_half_maximum, acceleration = self.choose_acceleration()

            self.full_width_half_maximum = full_width_half_maximum
            self.acceleration = acceleration
            self.center_scale = center_scale

            mask = self.gaussian_kspace()
            mask[tuple(self.gaussian_coordinates())] = 1.0

            mask = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(mask, axes=0), axes=0), axes=(0, 1))

            if partial_fourier_percentage != 0:
                mask[:, : int(np.round(mask.shape[1] * partial_fourier_percentage))] = 0.0

            mask = torch.from_numpy(np.transpose(mask, (1, 0))[0].reshape(*dims).astype(np.float32))

        return mask, acceleration

    def gaussian_kspace(self) -> np.ndarray:
        """Creates a Gaussian sampled k-space center."""
        scaled = int(self.shape[0] * self.center_scale)
        center = np.ones((scaled, self.shape[1]))
        top_scaled = torch.div((self.shape[0] - scaled), 2, rounding_mode="trunc").item()
        bottom_scaled = self.shape[0] - scaled - top_scaled
        top = np.zeros((top_scaled, self.shape[1]))
        btm = np.zeros((bottom_scaled, self.shape[1]))
        return np.concatenate((top, center, btm))

    def gaussian_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Returns Gaussian sampled k-space coordinates.

        Returns
        -------
        xsamples : np.ndarray
            A 1D numpy array of x-coordinates.
        ysamples : np.ndarray
            A 1D numpy array of y-coordinates.

        Notes
        -----
        The number of samples taken is determined by `n_sample` which is calculated as
        `self.shape[0] / self.acceleration`. The selection of the samples is based on the probabilities calculated
        from `gaussian_kernel`.
        """
        n_sample = int(self.shape[0] / self.acceleration)
        idxs = np.random.choice(range(self.shape[0]), size=n_sample, replace=False, p=self.gaussian_kernel())
        xsamples = np.concatenate([np.tile(i, self.shape[1]) for i in idxs])
        ysamples = np.concatenate([range(self.shape[1]) for _ in idxs])
        return xsamples, ysamples

    def gaussian_kernel(self) -> np.ndarray:
        r"""Creates a Gaussian sampled k-space kernel.

        .. note::
            The function calculates the Gaussian kernel by computing the sum of the exponential of the squared \
            x-values divided by 2 times the square of the standard deviation. The standard deviation is calculated \
            from the full width at half maximum (FWHM) of the Gaussian curve and is defined as the FWHM divided by \
            the square root of 8 times the natural logarithm of 2. The FWHM and the kern_len are obtained from the \
            `full_width_half_maximum` and `shape` attributes of the class respectively.

        Returns
        -------
        ndarray
            The Gaussian kernel.
        """
        kernel = 1
        for kern_len in self.shape:
            sigma = self.full_width_half_maximum / np.sqrt(8 * np.log(2))
            x = np.linspace(-1.0, 1.0, kern_len)
            g = np.exp(-(x**2 / (2 * sigma**2)))  # noqa: F841
            kernel = g
            break
        kernel = kernel / kernel.sum()  # type: ignore
        return kernel


class Gaussian2DMaskFunc(MaskFunc):
    """Creates a 2D sub-sampling mask of a given shape.

    The sub-sampling mask is generated in k-space, with data points near the k-space center being fully sampled within
    an ellipse. The half-axes of the ellipse are set to the `center_scale` percentage of the fully sampled region. The
    remaining points are sampled according to a Gaussian distribution.

    The center fractions act as Full-Width at Half-Maximum (FWHM) values.

    Examples
    --------
    >>> import torch
    >>> from atommic.collections.common.data.subsample import Gaussian2DMaskFunc
    >>> mask_func = Gaussian2DMaskFunc(center_fractions=[0.7, 0.7], accelerations=[4, 8])
    >>> mask_func.choose_acceleration()
    (0.7, 4)
    >>> kspace = torch.randn(1, 1, 640, 368)
    >>> mask, acceleration = mask_func(kspace.shape)
    >>> mask.shape
    torch.Size([1, 1, 640, 368])
    >>> acceleration
    4
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        partial_fourier_percentage: Optional[float] = 0.0,
        center_scale: Optional[float] = 0.02,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Calls :class:`Gaussian2DMaskFunc`.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the mask to be created. The shape should have at least 3 dimensions. Same as the shape of the
            input k-space data.
        seed : int or tuple of ints, optional
            Seed for the random number generator. Default is ``None``.
        partial_fourier_percentage : float, optional
            Percentage of the low-frequency columns to be retained. Default is ``0.0``.
        center_scale : float, optional
            For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse
            of which the half-axes will be set to the given `center_scale` percentage of the fully sampled region.
            Default is ``0.02``.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple of the generated mask and the acceleration factor.

        Raises
        ------
        ValueError
            If the `shape` parameter has less than 3 dimensions.
        """
        with temp_seed(self.rng, seed):
            dims = [1 for _ in shape]
            self.shape = tuple(shape[-3:-1])
            dims[-3:-1] = self.shape

            full_width_half_maximum, acceleration = self.choose_acceleration()

            self.full_width_half_maximum = full_width_half_maximum
            self.acceleration = acceleration
            self.center_scale = center_scale

            mask = self.gaussian_kspace()
            mask[tuple(self.gaussian_coordinates())] = 1.0

            if partial_fourier_percentage != 0:
                mask[: int(np.round(mask.shape[0] * partial_fourier_percentage)), :] = 0.0

            mask = torch.from_numpy(mask.reshape(dims).astype(np.float32))

        return mask, acceleration

    def gaussian_kspace(self) -> np.ndarray:
        """Creates a Gaussian sampled k-space center."""
        a, b = self.center_scale * self.shape[0], self.center_scale * self.shape[1]
        afocal, bfocal = self.shape[0] / 2, self.shape[1] / 2
        xx, yy = np.mgrid[: self.shape[0], : self.shape[1]]
        ellipse = np.power((xx - afocal) / a, 2) + np.power((yy - bfocal) / b, 2)
        return (ellipse < 1).astype(float)

    def gaussian_coordinates(self) -> List[Tuple[int, int]]:
        r"""Returns Gaussian sampled k-space coordinates.

        Returns
        -------
        xsamples : np.ndarray
            A 1D numpy array of x-coordinates.
        ysamples : np.ndarray
            A 1D numpy array of y-coordinates.

        Notes
        -----
        The number of samples taken is determined by `n_sample` which is calculated as \
        `self.shape[0] / self.acceleration`. The selection of the samples is based on the probabilities calculated \
        from `gaussian_kernel`.
        """
        n_sample = int(self.shape[0] * self.shape[1] / self.acceleration)
        cartesian_prod = list(np.ndindex(self.shape))
        kernel = self.gaussian_kernel()
        idxs = np.random.choice(range(len(cartesian_prod)), size=n_sample, replace=False, p=kernel.flatten())
        return list(zip(*list(map(cartesian_prod.__getitem__, idxs))))

    def gaussian_kernel(self) -> np.ndarray:
        r"""Creates a Gaussian sampled k-space kernel.

        .. note::
            The function calculates the Gaussian kernel by computing the sum of the exponential of the squared \
            x-values divided by 2 times the square of the standard deviation. The standard deviation is calculated \
            from the full width at half maximum (FWHM) of the Gaussian curve and is defined as the FWHM divided by \
            the square root of 8 times the natural logarithm of 2. The FWHM and the kern_len are obtained from the \
            `full_width_half_maximum` and `shape` attributes of the class respectively.

        Returns
        -------
        ndarray
            The Gaussian kernel.
        """
        kernels = []
        for kern_len in self.shape:
            sigma = self.full_width_half_maximum / np.sqrt(8 * np.log(2))
            x = np.linspace(-1.0, 1.0, kern_len)
            g = np.exp(-(x**2 / (2 * sigma**2)))
            kernels.append(g)
        kernel = np.sqrt(np.outer(kernels[0], kernels[1]))
        kernel = kernel / kernel.sum()
        return kernel


class Poisson2DMaskFunc(MaskFunc):
    r"""Generate variable-density Poisson-disc sampling pattern, as described in [1]_.

    The function generates a variable density Poisson-disc sampling mask with density proportional to
    :math:`1 / (1 + s |r|)`, where :math:`r` represents the k-space radius, and :math:`s` represents the slope. A
    binary search is performed on the slope :math:`s` such that the resulting acceleration factor is close to the
    prescribed acceleration factor `accel`. The parameter `tol` determines how much they can deviate.

    References
    ----------
    .. [1] Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions." SIGGRAPH sketches. 2007

    .. note::
        Taken and adapted from: https://github.com/mikgroup/sigpy/blob/master/sigpy/mri/samp.py
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        partial_fourier_percentage: Optional[float] = 0.0,
        center_scale: Optional[float] = 0.02,
        calib: Optional[Tuple[float, float]] = (0.0, 0.0),
        crop_corner: bool = True,
        max_attempts: int = 30,
        tol: float = 0.3,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Calls :class:`Poisson2DMaskFunc`.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the mask to be created. The shape should have at least 3 dimensions. Same as the shape of the
            input k-space data.
        seed : int or tuple of ints, optional
            Seed for the random number generator. Default is ``None``.
        partial_fourier_percentage : float, optional
            Percentage of the low-frequency columns to be retained. Default is ``0.0``.
        center_scale : float, optional
            For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse
            of which the half-axes will be set to the given `center_scale` percentage of the fully sampled region.
            Default is ``0.02``.
        calib : Optional[Tuple[float, float]], optional
            Defines the size of the calibration region, which is a square region in the center of k-space. The first
            value defines the percentage of the center that is sampled, and the second value defines the size of the
            calibration region in the center of k-space. Default is ``(0.0, 0.0)``.
        crop_corner : bool, optional
            If set to True, the center of the mask will be cropped to the size of the calibration region. Default is
            ``True``.
        max_attempts : int, optional
            Maximum number of attempts to generate a mask with the desired acceleration factor. Default is ``30``.
        tol : float, optional
            Tolerance for the acceleration factor. Default is ``0.3``.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple containing the mask and the acceleration factor.
            The mask is a `torch.Tensor` of the same shape as the input shape.
            The acceleration factor is an `int` that represents the number of samples taken in the k-space.
        """
        with temp_seed(self.rng, seed):
            self.shape = tuple(shape[-3:-1])
            self.center_scale = center_scale
            _, self.acceleration = self.choose_acceleration()

            ny, nx = self.shape
            y, x = np.mgrid[:ny, :nx]
            x = np.maximum(abs(x - nx / 2) - calib[-1] / 2, 0)  # type: ignore
            x /= x.max()
            y = np.maximum(abs(y - ny / 2) - calib[-2] / 2, 0)  # type: ignore
            y /= y.max()

            r = np.hypot(x, y)

            slope_max = 40.0
            slope_min = 0.0

            d = max(nx, ny)

            while slope_min < slope_max:
                slope = (slope_max + slope_min) / 2
                radius_x = np.clip((1 + r * slope) * nx / d, 1, None)
                radius_y = np.clip((1 + r * slope) * ny / d, 1, None)

                mask = self.generate_poisson_mask(nx, ny, max_attempts, radius_x, radius_y, calib)

                if crop_corner:
                    mask *= r < 1

                with np.errstate(divide="ignore", invalid="ignore"):
                    actual_acceleration = mask.size / np.sum(mask)

                if abs(actual_acceleration - self.acceleration) < tol:
                    break
                if actual_acceleration < self.acceleration:
                    slope_min = slope
                else:
                    slope_max = slope

            pattern1 = mask
            pattern2 = self.centered_circle()
            mask = np.logical_or(pattern1, pattern2)

            if abs(actual_acceleration - self.acceleration) >= tol:
                raise ValueError(f"Cannot generate mask to satisfy acceleration factor of {self.acceleration}.")

            if partial_fourier_percentage != 0:
                mask[:, : int(np.round(mask.shape[1] * partial_fourier_percentage))] = 0.0

            mask = torch.from_numpy(mask.reshape(self.shape).astype(np.float32)).unsqueeze(0).unsqueeze(-1)

        return mask, self.acceleration

    def centered_circle(self) -> np.ndarray:
        """Creates a boolean centered circle image using the center_scale as a radius.

        Returns
        -------
        np.ndarray
            A 2D array of type bool, where True values indicate the points inside the centered circle and False
            values indicate the points outside the centered circle. The circle has its center at the center of the
            input shape and its radius is determined by the `center_scale` attribute.
        """
        center_x = int((self.shape[0] - 1) / 2)
        center_y = int((self.shape[1] - 1) / 2)

        X, Y = np.indices(self.shape)
        radius = int(self.shape[0] * self.center_scale)
        radius_squared = radius**2
        return ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius_squared

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def generate_poisson_mask(
        nx: int,
        ny: int,
        max_attempts: int,
        radius_x: np.ndarray,
        radius_y: np.ndarray,
        calib: Tuple[float, float],
    ) -> np.ndarray:
        """Generates a Poisson mask of shape `(ny, nx)` by placing points on the grid according to a Poisson
        distribution.

        Parameters
        ----------
        nx : int
            The number of columns in the output mask.
        ny : int
            The number of rows in the output mask.
        max_attempts : int
            The maximum number of attempts to generate a mask with the desired acceleration factor.
        radius_x : np.ndarray
            An array of shape `(ny, nx)` representing the radius of the Poisson distribution in the x-direction.
        radius_y : np.ndarray
            An array of shape `(ny, nx)` representing the radius of the Poisson distribution in the y-direction.
        calib : Tuple[float, float]
            Defines the size of the calibration region. The calibration region is a square region in the center of
            k-space. The first value defines the percentage of the center that is sampled. The second value defines
            the size of the calibration region in the center of k-space.

        Returns
        -------
        np.ndarray
            A binary mask with points placed according to a Poisson distribution.
        """
        mask = np.zeros((ny, nx))

        # Add calibration region
        mask[
            int(ny / 2 - calib[-2] / 2) : int(ny / 2 + calib[-2] / 2),
            int(nx / 2 - calib[-1] / 2) : int(nx / 2 + calib[-1] / 2),
        ] = 1

        # initialize active list
        pxs = np.empty(nx * ny, np.int32)
        pys = np.empty(nx * ny, np.int32)
        pxs[0] = np.random.randint(0, nx)
        pys[0] = np.random.randint(0, ny)
        num_actives = 1
        while num_actives > 0:
            i = np.random.randint(0, num_actives)
            px = pxs[i]
            py = pys[i]
            rx = radius_x[py, px]
            ry = radius_y[py, px]

            # Attempt to generate point
            done = False
            k = 0
            while not done and k < max_attempts:
                # Generate point randomly from r and 2 * r
                v = (np.random.random() * 3 + 1) ** 0.5
                t = 2 * np.pi * np.random.random()
                qx = px + v * rx * np.cos(t)
                qy = py + v * ry * np.sin(t)

                # Reject if outside grid or close to other points
                if 0 <= qx < nx and 0 <= qy < ny:
                    startx = max(int(qx - rx), 0)
                    endx = min(int(qx + rx + 1), nx)
                    starty = max(int(qy - ry), 0)
                    endy = min(int(qy + ry + 1), ny)

                    done = True
                    for x in range(startx, endx):
                        for y in range(starty, endy):
                            if mask[y, x] == 1 and (
                                ((qx - x) / radius_x[y, x]) ** 2 + ((qy - y) / (radius_y[y, x])) ** 2 < 1
                            ):
                                done = False
                                break

                k += 1

            # Add point if done else remove from active list
            if done:
                pxs[num_actives] = qx
                pys[num_actives] = qy
                mask[int(qy), int(qx)] = 1
                num_actives += 1
            else:
                pxs[i] = pxs[num_actives - 1]
                pys[i] = pys[num_actives - 1]
                num_actives -= 1

        return mask


class Random1DMaskFunc(MaskFunc):
    r"""Random1DMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N columns, the mask
    picks out:

        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to low-frequencies.

        2. The other columns are selected uniformly at random with a probability equal to:
        prob = (N / acceleration - N_low_freqs) /  (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which case one possible (center_fraction,
    acceleration) is chosen uniformly at random each time the Random1DMaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there is a 50% probability that
    4-fold acceleration with 8% center fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        partial_fourier_percentage: Optional[float] = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Calls :class:`Random1DMaskFunc`.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the mask to be created. The shape should have at least 3 dimensions. Same as the shape of the
            input k-space data.
        seed : int or tuple of ints, optional
            Seed for the random number generator. Default is ``None``.
        partial_fourier_percentage : float, optional
            Percentage of the low-frequency columns to be retained. Default is ``0.0``.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple of the generated mask and the acceleration factor.

        Raises
        ------
        ValueError
            If the `shape` parameter has less than 3 dimensions.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
            mask = self.rng.uniform(size=num_cols) < prob
            pad = torch.div((num_cols - num_low_freqs + 1), 2, rounding_mode="trunc").item()
            mask[pad : pad + num_low_freqs] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

            if partial_fourier_percentage != 0:
                mask[:, : int(np.round(mask.shape[1] * partial_fourier_percentage))] = 0.0

        return mask, acceleration


def create_masker(
    mask_type_str: str, center_fractions: Union[Sequence[float], float], accelerations: Union[Sequence[int], int]
) -> MaskFunc:
    """Creates a MaskFunc object based on the specified mask type.

    Parameters
    ----------
    mask_type_str : str
        The string representation of the mask type. Must be one of the following:
        'equispaced1d', 'equispaced2d', 'gaussian1d', 'gaussian2d', 'poisson2d', 'random1d'.
    center_fractions : Sequence[float] or float
        The center fractions for the mask.
    accelerations : Sequence[int] or int
        The accelerations for the mask.

    Returns
    -------
    MaskFunc
        A MaskFunc object that corresponds to the specified mask type.

    Raises
    ------
    NotImplementedError
        If the specified `mask_type_str` is not supported.

    Examples
    --------
    >>> from atommic.collections.common.data.subsample import create_masker
    >>> create_masker("random1d", [0.5], [4])
    Random1DMaskFunc([0.5], [4])
    >>> create_masker("equispaced2d", [0.3, 0.7], [8, 6])
    Equispaced2DMaskFunc([0.3, 0.7], [8, 6])
    >>> create_masker("poisson2d", [0.3, 0.7], [8, 6])
    Poisson2DMaskFunc([0.3, 0.7], [8, 6])
    >>> create_masker("gaussian1d", [0.3, 0.7], [8, 6])
    Gaussian1DMaskFunc([0.3, 0.7], [8, 6])
    >>> create_masker("gaussian2d", [0.3, 0.7], [8, 6])
    Gaussian2DMaskFunc([0.3, 0.7], [8, 6])
    """
    if isinstance(center_fractions, float):
        center_fractions = [center_fractions]
    if isinstance(accelerations, int):
        accelerations = [accelerations]
    if mask_type_str == "random1d":
        return Random1DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "equispaced1d":
        return Equispaced1DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "equispaced2d":
        return Equispaced2DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "gaussian1d":
        return Gaussian1DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "gaussian2d":
        return Gaussian2DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "poisson2d":
        return Poisson2DMaskFunc(center_fractions, accelerations)
    raise NotImplementedError(f"{mask_type_str} not supported")
