# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import List, Optional, Tuple

import torch

from atommic.collections.common.parts import utils
from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.quantitative.nn.base import SignalForwardModel


class qVarNetBlock(torch.nn.Module):
    """Implementation of the quantitative End-to-end Variational Network (qVN), as presented in [Zhang2022]_.

    References
    ----------
    .. [Zhang2022] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW.
        A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent
        inference machine. NeuroImage. 2022 Dec 1;264:119680.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        no_dc: bool = False,
        linear_forward_model=None,
    ):
        """Inits :class:`qVarNetBlock`.

        Parameters
        ----------
        model : torch.nn.Module
            Model to apply soft data consistency.
        fft_centered : bool, optional
            Whether to center the fft. Default is ``False``.
        fft_normalization : str, optional
            The normalization of the fft. Default is ``backward``.
        spatial_dims : tuple, optional
            The spatial dimensions of the data. Default is ``None``.
        coil_dim : int, optional
            The dimension of the coils. Default is ``1``.
        no_dc : bool, optional
            Whether to not apply the DC component. Default is ``False``.
        linear_forward_model : torch.nn.Module, optional
            Linear forward model. Default is ``None``.
        """
        super().__init__()

        self.linear_forward_model = (
            SignalForwardModel(sequence="MEGRE") if linear_forward_model is None else linear_forward_model
        )

        self.model = model
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim
        self.no_dc = no_dc
        self.dc_weight = torch.nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """Combines the sensitivity maps with coil-combined data to get multicoil data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        sens_maps : torch.Tensor
            Coil Sensitivity maps.

        Returns
        -------
        torch.Tensor
            Expanded multicoil data.
        """
        return fft2(
            utils.complex_mul(x, sens_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """Combines the sensitivity maps with multicoil data to get coil-combined data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        sens_maps : torch.Tensor
            Coil Sensitivity maps.

        Returns
        -------
        torch.Tensor
            SENSE coil-combined reconstruction.
        """
        x = ifft2(x, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims)
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(dim=self.coil_dim)

    def forward(
        self,
        prediction: torch.Tensor,
        masked_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        sampling_mask: torch.Tensor,
        TEs: List,
    ) -> torch.Tensor:
        """Forward pass of :class:`qVarNetBlock`.

        Parameters
        ----------
        prediction : torch.Tensor
            Initial prediction of the quantitative maps.
        masked_kspace : torch.Tensor
            Subsampled k-space of shape [batch_size, n_coils, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2].
        sampling_mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        TEs : List
            List of echo times.

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape [batch_size, n_coils, n_x, n_y, 2].
        """
        initial_prediction = self.linear_forward_model(
            prediction[:, 0, ...].unsqueeze(0),  # R2*
            prediction[:, 1, ...].unsqueeze(0),  # S0
            prediction[:, 2, ...].unsqueeze(0),  # B0
            prediction[:, 3, ...].unsqueeze(0),  # phi
            TEs,
        )
        initial_prediction_kspace = self.sens_expand(initial_prediction, sensitivity_maps.unsqueeze(self.coil_dim - 1))
        soft_dc = (initial_prediction_kspace - masked_kspace) * sampling_mask * self.dc_weight
        initial_prediction = self.sens_reduce(soft_dc, sensitivity_maps.unsqueeze(self.coil_dim - 1)).to(masked_kspace)

        prediction = torch.view_as_real(prediction + torch.view_as_complex(self.model(initial_prediction)))
        prediction_tmp = prediction[:, 0, ...]
        prediction_tmp[prediction_tmp < 0] = 0
        prediction[:, 0, ...] = prediction_tmp

        return torch.abs(torch.view_as_complex(prediction))
