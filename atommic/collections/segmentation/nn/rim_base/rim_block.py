# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Optional, Tuple

import torch

from atommic.collections.reconstruction.nn.rim_base import conv_layers, rnn_cells


class SegmentationRIMBlock(torch.nn.Module):
    """SegmentationRIMBlock is a block of Recurrent Inference Machines (RIMs) for segmentation tasks.

    References
    ----------
    .. empty:: TODO

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
    ):
        """Inits :class:`SegmentationRIMBlock`.

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
        """
        super().__init__()

        self.input_size = depth * conv_filters[-1]
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

    def forward(
        self, prediction: torch.Tensor, image: torch.Tensor, hx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`SegmentationRIMBlock`.

        Parameters
        ----------
        prediction : torch.Tensor
            Current prediction. Shape [batch_size, n_x, n_y].
        image : torch.Tensor
            Initial input image. Shape [batch_size, n_x, n_y].
        hx : torch.Tensor, optional
            Initial prediction for the hidden state. Shape: ``[batch, height, width]``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Prediction and hidden state.
        """
        if hx is None:
            hx = [
                prediction.new_zeros((prediction.size(0), f, *prediction.size()[2:]))
                for f in self.recurrent_filters
                if f != 0
            ]

        predictions = []
        for _ in range(self.time_steps):
            log_likelihood_gradient_prediction = torch.abs(prediction - image)
            log_likelihood_gradient_prediction = torch.cat([log_likelihood_gradient_prediction, prediction], dim=1)
            for h, convrnn in enumerate(self.layers):
                hx[h] = convrnn(log_likelihood_gradient_prediction, hx[h])
                log_likelihood_gradient_prediction = hx[h]
            prediction = prediction + self.final_layer(log_likelihood_gradient_prediction)
            predictions.append(prediction)

        return predictions, hx
