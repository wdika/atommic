# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Any, Optional, Tuple, Union

import torch

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import coil_combination_method, complex_mul
from atommic.collections.reconstruction.nn.rim_base import conv_layers, rim_utils, rnn_cells


class RIMBlock(torch.nn.Module):
    """RIMBlock is a block of Recurrent Inference Machines (RIMs) as presented in [Lonning19]_.

    References
    ----------
    .. [Lonning19] Lonning19 K, Putzky P, Sonke JJ, Reneman L, Caan MW, Welling M. Recurrent inference machines for
        reconstructing heterogeneous MRI data. Medical image analysis. 2019 Apr 1;53:64-78.

    """

    def __init__(
        self,
        recurrent_layer=None,
        conv_filters=None,
        conv_kernels=None,
        conv_dilations=None,
        conv_bias=None,
        recurrent_filters=None,
        recurrent_kernels=None,
        recurrent_dilations=None,
        recurrent_bias=None,
        depth: int = 2,
        time_steps: int = 8,
        conv_dim: int = 2,
        no_dc: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        dimensionality: int = 2,
        consecutive_slices: int = 1,
        coil_combination_method: str = "SENSE",
    ):
        """Inits :class:`RIMBlock`.

        Parameters
        ----------
        recurrent_layer : torch.nn.Module
            Type of the recurrent layer. It can be ``GRU``, ``MGU``, ``IndRNN``. Check ``rnn_cells`` for more details.
        conv_filters : list of int
            Number of filters in the convolutional layers.
        conv_kernels : list of int
            Kernel size of the convolutional layers.
        conv_dilations : list of int
            Dilation of the convolutional layers.
        conv_bias : list of bool
            Bias of the convolutional layers.
        recurrent_filters : list of int
            Number of filters in the recurrent layers.
        recurrent_kernels : list of int
            Kernel size of the recurrent layers.
        recurrent_dilations : list of int
            Dilation of the recurrent layers.
        recurrent_bias : list of bool
            Bias of the recurrent layers.
        depth : int
            Number of sequence of convolutional and recurrent layers. Default is ``2``.
        time_steps : int
            Number of recurrent time steps. Default is ``8``.
        conv_dim : int
            Dimension of the convolutional layers. Default is ``2``.
        no_dc : bool
            If ``True`` the DC component is not used. Default is ``True``.
        fft_centered : bool
            If ``True`` the FFT is centered. Default is ``False``.
        fft_normalization : str
            Normalization of the FFT. Default is ``"backward"``.
        spatial_dims : tuple of int
            Spatial dimensions of the input. Default is ``None``.
        coil_dim : int
            Coil dimension of the input. Default is ``1``.
        dimensionality : int
            Dimensionality of the input. Default is ``2``.
        consecutive_slices : int
            Number of consecutive slices. Default is ``1``.
        coil_combination_method : str
            Coil combination method. Default is ``"SENSE"``.
        """
        super().__init__()

        self.input_size = depth * 2
        self.time_steps = time_steps

        self.layers = torch.nn.ModuleList()
        for (
            (conv_features, conv_k_size, conv_dilation, l_conv_bias, nonlinear),
            (rnn_features, rnn_k_size, rnn_dilation, rnn_bias, rnn_type),
        ) in zip(
            zip(conv_filters, conv_kernels, conv_dilations, conv_bias, ["relu", "relu", None]),
            zip(
                recurrent_filters,
                recurrent_kernels,
                recurrent_dilations,
                recurrent_bias,
                [recurrent_layer, recurrent_layer, None],
            ),
        ):
            conv_layer = None

            if conv_features != 0:
                conv_layer = conv_layers.ConvNonlinear(
                    self.input_size,
                    conv_features,
                    conv_dim=conv_dim,
                    kernel_size=conv_k_size,
                    dilation=conv_dilation,
                    bias=l_conv_bias,
                    nonlinear=nonlinear,
                )
                self.input_size = conv_features

            if rnn_features != 0 and rnn_type is not None:
                if rnn_type.upper() == "GRU":
                    rnn_type = rnn_cells.ConvGRUCell
                elif rnn_type.upper() == "MGU":
                    rnn_type = rnn_cells.ConvMGUCell
                elif rnn_type.upper() == "INDRNN":
                    rnn_type = rnn_cells.IndRNNCell
                else:
                    raise ValueError("Please specify a proper recurrent layer type.")

                rnn_layer = rnn_type(
                    self.input_size,
                    rnn_features,
                    conv_dim=conv_dim,
                    kernel_size=rnn_k_size,
                    dilation=rnn_dilation,
                    bias=rnn_bias,
                )

                self.input_size = rnn_features

                self.layers.append(conv_layers.ConvRNNStack(conv_layer, rnn_layer))

        self.final_layer = torch.nn.Sequential(conv_layer)

        self.recurrent_filters = recurrent_filters

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim

        self.no_dc = no_dc

        if not self.no_dc:
            self.dc_weight = torch.nn.Parameter(torch.ones(1))
            self.zero = torch.zeros(1, 1, 1, 1, 1)

        self.dimensionality = dimensionality
        self.consecutive_slices = consecutive_slices
        self.coil_combination_method = coil_combination_method

    def forward(
        self,
        y: torch.Tensor,
        masked_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        prediction: torch.Tensor = None,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
        keep_prediction: bool = False,
    ) -> Tuple[Any, Union[list, torch.Tensor, None]]:
        """Forward pass of :class:`RIMBlock`.

        Parameters
        ----------
        y : torch.Tensor
            Predicted k-space. Shape: ``[batch, coils, height, width, 2]``.
        masked_kspace : torch.Tensor
            Subsampled k-space. Shape: ``[batch, coils, height, width, 2]``.
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape: ``[batch, coils, height, width, 2]``.
        mask : torch.Tensor
            Subsampling mask. Shape: ``[batch, coils, height, width, 2]``.
        prediction : torch.Tensor, optional
            Initial (zero-filled) prediction. Shape: ``[batch, coils, height, width, 2]``.
        hx : torch.Tensor, optional
            Initial prediction for the hidden state. Shape: ``[batch, coils, height, width, 2]``.
        sigma : float, optional
            Noise level. Default is ``1.0``.
        keep_prediction : bool, optional
            Whether to keep the prediction. Default is ``False``.

        Returns
        -------
        Tuple[Any, Union[list, torch.Tensor, None]]
            Reconstructed image and hidden states.
        """
        batch = masked_kspace.shape[0]
        if self.dimensionality == 3 or self.consecutive_slices > 1:
            # 3D pred.shape = [batch, slices, coils, height, width, 2] -> [batch * slices, coils, height, width, 2]
            slices = masked_kspace.shape[1]
            y = y.reshape([batch * slices, *y.shape[2:]])
            masked_kspace = masked_kspace.reshape([batch * slices, *masked_kspace.shape[2:]])
            mask = mask.reshape([batch * slices, *mask.shape[2:]])
            sensitivity_maps = sensitivity_maps.reshape([batch * slices, *sensitivity_maps.shape[2:]])
        else:
            # 2D pred.shape = [batch, coils, height, width, 2]
            slices = 1

        if hx is None or (not isinstance(hx, list) and hx.dim() < 3):
            hx = [
                masked_kspace.new_zeros((masked_kspace.size(0), f, *masked_kspace.size()[2:-1]))
                for f in self.recurrent_filters
                if f != 0
            ]

        if prediction is None or prediction.ndim < 3:
            prediction = (
                y
                if keep_prediction
                else coil_combination_method(
                    ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                    sensitivity_maps,
                    method=self.coil_combination_method,
                    dim=self.coil_dim,
                )
            )

        if (self.consecutive_slices > 1 or self.dimensionality == 3) and prediction.dim() == 5:
            prediction = prediction.reshape([batch * slices, *prediction.shape[2:]])

        predictions = []
        for _ in range(self.time_steps):
            log_likelihood_gradient_prediction = rim_utils.log_likelihood_gradient(
                prediction,
                masked_kspace,
                sensitivity_maps,
                mask,
                sigma,
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
                self.coil_dim,
            ).contiguous()

            if self.consecutive_slices > 1 or self.dimensionality == 3:
                log_likelihood_gradient_prediction = log_likelihood_gradient_prediction.view(
                    [
                        batch * slices,
                        4,
                        log_likelihood_gradient_prediction.shape[2],
                        log_likelihood_gradient_prediction.shape[3],
                    ]
                ).permute(1, 0, 2, 3)

            for h, convrnn in enumerate(self.layers):
                hx[h] = convrnn(log_likelihood_gradient_prediction, hx[h])
                if self.consecutive_slices > 1 or self.dimensionality == 3:
                    hx[h] = hx[h].squeeze(0)
                log_likelihood_gradient_prediction = hx[h]

            log_likelihood_gradient_prediction = self.final_layer(log_likelihood_gradient_prediction)

            if self.dimensionality == 2:
                log_likelihood_gradient_prediction = log_likelihood_gradient_prediction.permute(0, 2, 3, 1)
            elif self.dimensionality == 3:
                log_likelihood_gradient_prediction = log_likelihood_gradient_prediction.permute(1, 2, 3, 0)
                for h in range(len(hx)):  # pylint: disable=consider-using-enumerate
                    hx[h] = hx[h].permute(1, 0, 2, 3)

            prediction = prediction + log_likelihood_gradient_prediction

            predictions.append(prediction)

        if self.consecutive_slices > 1 or self.dimensionality == 3:
            for i, pred in enumerate(predictions):
                predictions[i] = pred.reshape([batch, slices, *pred.shape[1:]])

        if self.no_dc:
            return predictions, hx

        soft_dc = torch.where(mask, y - masked_kspace, self.zero.to(masked_kspace)) * self.dc_weight
        current_kspace = [
            masked_kspace
            - soft_dc
            - fft2(
                complex_mul(e.unsqueeze(self.coil_dim), sensitivity_maps),
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            for e in predictions
        ]

        return current_kspace, hx
