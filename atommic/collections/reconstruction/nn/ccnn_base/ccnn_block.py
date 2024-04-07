# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Optional, Tuple

import torch

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import complex_conj, complex_mul


class Conv2d(torch.nn.Module):
    """Implementation of a simple cascade of 2D convolutions. If batchnorm is set to True, batch normalization layer is
    applied after each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_convs: int = 3,
        activation: torch.nn.Module = torch.nn.PReLU(),
        batchnorm: bool = False,
    ):
        """Inits :class:`Conv2d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        hidden_channels : int
            Number of hidden channels.
        n_convs : int, optional
            Number of convolutional layers. Default is ``3``.
        activation : torch.nn.Module, optional
            Activation function. Default is ``nn.PReLU()``.
        batchnorm : bool, optional
            If True a batch normalization layer is applied after every convolution. Default is ``False``.
        """
        super().__init__()

        self.conv = []
        for idx in range(n_convs):
            self.conv.append(
                torch.nn.Conv2d(
                    in_channels if idx == 0 else hidden_channels,
                    hidden_channels if idx != n_convs - 1 else out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if batchnorm:
                self.conv.append(
                    torch.nn.BatchNorm2d(hidden_channels if idx != n_convs - 1 else out_channels, eps=1e-4)
                )
            if idx != n_convs - 1:
                self.conv.append(activation)
        self.conv = torch.nn.Sequential(*self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`Conv2d`."""
        if x.dim() == 5:
            x = x.squeeze(1)
            if x.shape[-1] == 2:
                x = x.permute(0, 3, 1, 2)
        return self.conv(x)  # type: ignore


class CascadeNetBlock(torch.nn.Module):
    """Model block for CascadeNet & Convolution Recurrent Neural Network."""

    def __init__(
        self,
        model: torch.nn.Module,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        no_dc: bool = False,
    ):
        """Inits :class:`CascadeNetBlock`.

        Parameters
        ----------
        model : torch.nn.Module
            Model to apply soft data consistency.
        fft_centered : bool, optional
            Whether to center the FFT. Default is ``False``.
        fft_normalization : str, optional
            Whether to normalize the FFT. Default is ``"backward"``.
        spatial_dims : Tuple[int, int], optional
            Spatial dimensions of the input. Default is ``None``.
        coil_dim : int, optional
            Coil dimension. Default is ``1``.
        no_dc : bool, optional
            Flag to disable the soft data consistency. Default is ``False``.
        """
        super().__init__()

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
            complex_mul(x, sens_maps),
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
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=self.coil_dim)

    def forward(
        self,
        pred: torch.Tensor,
        ref_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of :class:`CascadeNetBlock`.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        ref_kspace : torch.Tensor
            Reference k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]

        Returns
        -------
        torch.Tensor
            Reconstructed image. Shape [batch_size, n_x, n_y, 2]
        """
        zero = torch.zeros(1, 1, 1, 1, 1).to(pred)
        soft_dc = torch.where(mask.bool(), pred - ref_kspace, zero) * self.dc_weight

        prediction = self.sens_reduce(pred, sensitivity_maps)
        prediction = self.model(prediction.squeeze(self.coil_dim).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if prediction.dim() < sensitivity_maps.dim():
            prediction = prediction.unsqueeze(1)
        prediction = self.sens_expand(prediction, sensitivity_maps)

        if not self.no_dc:
            prediction = pred - soft_dc - prediction

        return prediction
