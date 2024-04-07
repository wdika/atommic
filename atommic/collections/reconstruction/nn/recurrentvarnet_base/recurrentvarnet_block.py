# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/nn/recurrentvarnet/recurrentvarnet.py

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import complex_conj, complex_mul


class Conv2dGRU(nn.Module):
    """2D Convolutional Gated Recurrent Unit."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
    ):
        """Inits :class:`Conv2dGRU`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        hidden_channels : int
            Number of hidden channels.
        out_channels : int, optional
            Number of output channels. If None, same as in_channels.
        num_layers : int, optional
            Number of layers. Default is ``2``.
        gru_kernel_size : int, optional
            Size of the GRU kernel. Default is ``1``.
        orthogonal_initialization : bool, optional
            Orthogonal initialization is used if set to True. Default is ``True``.
        instance_norm : bool, optional
            Instance norm is used if set to True. Default is ``False``.
        dense_connect : int, optional
            Number of dense connections. Default is ``0``.
        replication_padding : bool, optional
            If set to true replication padding is applied. Default is ``True``.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dense_connect = dense_connect

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks
        for idx in range(num_layers + 1):
            in_ch = in_channels if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < num_layers else out_channels
            padding = 0 if replication_padding else (2 if idx == 0 else 1)
            block = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5 if idx == 0 else 3,
                    dilation=(2 if idx == 1 else 1),
                    padding=padding,
                )
            )
            self.conv_blocks.append(nn.Sequential(*block))

        # Create GRU blocks
        for _ in range(num_layers):
            for gru_part in [self.reset_gates, self.update_gates, self.out_gates]:
                block = []
                if instance_norm:
                    block.append(nn.InstanceNorm2d(2 * hidden_channels))
                block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=gru_kernel_size,
                        padding=gru_kernel_size // 2,
                    )
                )
                gru_part.append(nn.Sequential(*block))

        if orthogonal_initialization:
            for reset_gate, update_gate, out_gate in zip(self.reset_gates, self.update_gates, self.out_gates):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`Conv2dGRU`.

        Parameters
        ----------
        cell_input : torch.Tensor
            Input tensor.
        previous_state : torch.Tensor
            Previous hidden state.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of output tensor and new hidden state.
        """
        new_states: List[torch.Tensor] = []
        conv_skip: List[torch.Tensor] = []

        if previous_state is None:
            batch_size, spatial_size = cell_input.size(0), (cell_input.size(2), cell_input.size(3))
            state_size = [batch_size, self.hidden_channels] + list(spatial_size) + [self.num_layers]
            previous_state = torch.zeros(*state_size, dtype=cell_input.dtype).to(cell_input.device)

        for idx in range(self.num_layers):
            if conv_skip:
                cell_input = F.relu(
                    self.conv_blocks[idx](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)

            stacked_inputs = torch.cat([cell_input, previous_state[:, :, :, :, idx]], dim=1)

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(
                self.out_gates[idx](torch.cat([cell_input, previous_state[:, :, :, :, idx] * reset], dim=1))
            )
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if conv_skip:
            out = self.conv_blocks[self.num_layers](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1))
        else:
            out = self.conv_blocks[self.num_layers](cell_input)

        return out, torch.stack(new_states, dim=-1)


class RecurrentInit(nn.Module):
    """Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [Yiasemis2021]_.

    References
    ----------
    .. [Yiasemis2021] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver
        Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...],
        dilations: Tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        """Inits :class:`RecurrentInit`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        out_channels : int
            Number of hidden channels of the recurrent unit of RecurrentVarNet Block.
        channels : Tuple[int, ...]
            Channels :math:`n_d` in the convolutional layers of initializer.
        dilations : Tuple[int, ...]
            Dilations :math:`p` of the convolutional layers of the initializer.
        depth : int, optional
            RecurrentVarNet Block number of layers :math:`n_l`. Default is ``2``.
        multiscale_depth : int, optional
            Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is disabled.
            Default is ``1``.
        """
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = in_channels
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`RecurrentInit`.

        Parameters
        ----------
        x : torch.Tensor
            Initialization for RecurrentInit.

        Returns
        -------
        torch.Tensor
            Initial recurrent hidden state from input `x`.
        """
        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        return torch.stack(output_list, dim=-1)


class RecurrentVarNetBlock(nn.Module):
    r"""Recurrent Variational Network Block :math:`'\'mathcal{H}_{'\'theta_{t}}` as presented in [Yiasemis2021]_.

    References
    ----------
    .. [Yiasemis2021] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver
        Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
    ):
        """Inits :class:`RecurrentVarNetBlock`.

        Parameters
        ----------
        in_channels : int
            Input channels. Default is ``2`` for complex data.
        hidden_channels : int
            Number of hidden channels of the recurrent unit of RecurrentVarNet Block. Default is ``64``.
        num_layers : int
            Number of layers of :math:`n_l` recurrent unit. Default is ``4``.
        fft_centered : bool
            Whether to center the FFT. Default is ``False``.
        fft_normalization : str
            Whether to normalize the FFT. Default is ``"backward"``.
        spatial_dims : Tuple[int, int], optional
            Spatial dimensions of the input. Default is ``None``.
        coil_dim : int
            Coil dimension of the input. Default is ``1``.
        """
        super().__init__()
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim

        self.learning_rate = nn.Parameter(torch.tensor([1.0]))  # :math:`\alpha_t`
        self.regularizer = Conv2dGRU(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            replication_padding=True,
        )  # Recurrent Unit of RecurrentVarNet Block :math:`\mathcal{H}_{\theta_t}`

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        hidden_state: Union[None, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`RecurrentVarNetBlock`.

        Parameters
        ----------
        current_kspace : torch.Tensor
            Current k-space prediction. Shape [batch_size, n_coil, height, width, 2].
        masked_kspace : torch.Tensor
            Subsampled k-space. Shape [batch_size, n_coil, height, width, 2].
        sampling_mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1, height, width, 1].
        sensitivity_map : torch.Tensor
            Coil sensitivities. Shape [batch_size, n_coil, height, width, 2].
        hidden_state : Union[None, torch.Tensor]
            ConvGRU hidden state. Shape [batch_size, n_l, height, width, hidden_channels].

        Returns
        -------
        new_kspace : torch.Tensor
            New k-space prediction. Shape [batch_size, n_coil, height, width, 2].
        new_hidden_state : list of torch.Tensor
            New ConvGRU hidden state. Shape [batch_size, n_l, height, width, hidden_channels].
        """
        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )

        recurrent_term = torch.cat(
            [
                complex_mul(
                    ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    complex_conj(sensitivity_map),
                ).sum(self.coil_dim)
                for kspace in torch.split(current_kspace, 2, -1)
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)

        recurrent_term, hidden_state = self.regularizer(recurrent_term, hidden_state)  # :math:`w_t`, :math:`h_{t+1}`
        recurrent_term = recurrent_term.permute(0, 2, 3, 1)

        recurrent_term = torch.cat(
            [
                fft2(
                    complex_mul(image.unsqueeze(self.coil_dim), sensitivity_map),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                for image in torch.split(recurrent_term, 2, -1)
            ],
            dim=-1,
        )

        new_kspace = current_kspace - self.learning_rate * kspace_error + recurrent_term

        return new_kspace, hidden_state
