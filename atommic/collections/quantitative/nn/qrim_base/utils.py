# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import List, Sequence

import torch

from atommic.collections.common.parts import fft, utils
from atommic.collections.quantitative.nn.base import SignalForwardModel


def expand_op(x: torch.Tensor, sensitivity_maps: torch.Tensor) -> torch.Tensor:
    """Expands a coil-combined image to multicoil.

    Parameters
    ----------
    x : torch.Tensor
        Coil-combined image.
    sensitivity_maps : torch.Tensor
        Coil sensitivity maps.

    Returns
    -------
    torch.Tensor
        Multicoil image.

    Examples
    --------
    >>> import torch
    >>> from atommic.collections.quantitative.nn.qrim_base.utils import expand_op
    >>> data = torch.randn(1, 1, 320, 320, 2)
    >>> coil_sensitivity_maps = torch.randn(1, 32, 320, 320, 2)
    >>> expand_op(data, coil_sensitivity_maps).shape
    torch.Size([1, 32, 320, 320, 2])
    """
    x = utils.complex_mul(x, sensitivity_maps)
    if torch.isnan(x).any():
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    return x


def analytical_log_likelihood_gradient(
    linear_forward_model: SignalForwardModel,
    prediction: torch.Tensor,
    TEs: List,
    sensitivity_maps: torch.Tensor,
    masked_kspace: torch.Tensor,
    sampling_mask: torch.Tensor,
    fft_centered: bool,
    fft_normalization: str,
    spatial_dims: Sequence[int],
    coil_dim: int,
    coil_combination_method: str = "SENSE",
    scaling: float = 1e-3,
) -> torch.Tensor:
    """Computes the analytical gradient of the log-likelihood function.

    Parameters
    ----------
    linear_forward_model: SignalForwardModel
        Signal forward model to use.
    prediction : torch.Tensor
        Current prediction of the quantitative maps.
    TEs : List
        List of echo times.
    sensitivity_maps : torch.Tensor
        Coil sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2].
    masked_kspace : torch.Tensor
        Data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
    sampling_mask : torch.Tensor
        Mask of the sampling of shape [batch_size, 1, n_x, n_y, 1].
    fft_centered : bool
        If True, the FFT is centered.
    fft_normalization : str
        Normalization of the FFT.
    spatial_dims : Sequence[int]
        Spatial dimensions of the input.
    coil_dim : int
        Coil dimension.
    coil_combination_method : str, optional
        Coil combination method. Default is ``"SENSE"``.
    scaling : float, optional
        Scaling factor. Default is ``1e-3``.

    Returns
    -------
    torch.Tensor
        Analytical gradient of the log-likelihood function.
    """
    nr_TEs = len(TEs)

    R2star_map, S0_map, B0_map, phi_map = prediction[0], prediction[1], prediction[2], prediction[3]

    R2star_map = R2star_map.unsqueeze(0)
    S0_map = S0_map.unsqueeze(0)
    B0_map = B0_map.unsqueeze(0)
    phi_map = phi_map.unsqueeze(0)

    pred = linear_forward_model(R2star_map, S0_map, B0_map, phi_map, TEs)

    S0_map_real = S0_map
    S0_map_imag = phi_map

    pred_kspace = fft.fft2(
        expand_op(pred.unsqueeze(coil_dim), sensitivity_maps.unsqueeze(0).unsqueeze(coil_dim - 1)),
        fft_centered,
        fft_normalization,
        spatial_dims,
    )

    diff_data = (pred_kspace - masked_kspace) * sampling_mask
    diff_data_inverse = utils.coil_combination_method(
        fft.ifft2(diff_data, fft_centered, fft_normalization, spatial_dims),
        sensitivity_maps.unsqueeze(0).unsqueeze(coil_dim - 1),
        method=coil_combination_method,
        dim=coil_dim,
    )

    def first_term(i):
        """First term of the gradient."""
        return torch.exp(-TEs[i] * scaling * R2star_map)

    def second_term(i):
        """Second term of the gradient."""
        return torch.cos(B0_map * scaling * -TEs[i])

    def third_term(i):
        """Third term of the gradient."""
        return torch.sin(B0_map * scaling * -TEs[i])

    S0_part_der = torch.stack(
        [torch.stack((first_term(i) * second_term(i), -first_term(i) * third_term(i)), -1) for i in range(nr_TEs)], 1
    )

    R2str_part_der = torch.stack(
        [
            torch.stack(
                (
                    -TEs[i] * scaling * first_term(i) * (S0_map_real * second_term(i) - S0_map_imag * third_term(i)),
                    -TEs[i] * scaling * first_term(i) * (-S0_map_real * third_term(i) - S0_map_imag * second_term(i)),
                ),
                -1,
            )
            for i in range(nr_TEs)
        ],
        1,
    )

    S0_map_real_grad = (
        diff_data_inverse[..., 0] * S0_part_der[..., 0] - diff_data_inverse[..., 1] * S0_part_der[..., 1]
    )
    S0_map_imag_grad = (
        diff_data_inverse[..., 0] * S0_part_der[..., 1] + diff_data_inverse[..., 1] * S0_part_der[..., 0]
    )
    R2star_map_real_grad = (
        diff_data_inverse[..., 0] * R2str_part_der[..., 0] - diff_data_inverse[..., 1] * R2str_part_der[..., 1]
    )
    R2star_map_imag_grad = (
        diff_data_inverse[..., 0] * R2str_part_der[..., 1] + diff_data_inverse[..., 1] * R2str_part_der[..., 0]
    )

    S0_map_grad = torch.stack([S0_map_real_grad, S0_map_imag_grad], -1).squeeze()
    S0_map_grad = torch.mean(S0_map_grad, 0)
    R2star_map_grad = torch.stack([R2star_map_real_grad, R2star_map_imag_grad], -1).squeeze()
    R2star_map_grad = torch.mean(R2star_map_grad, 0)

    return torch.stack([R2star_map_grad[..., 0], S0_map_grad[..., 0], R2star_map_grad[..., 1], S0_map_grad[..., 1]], 0)
