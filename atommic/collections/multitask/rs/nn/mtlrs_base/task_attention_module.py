# coding=utf-8
__author__ = "Tim Paquaij"
import torch
import torch.nn as nn


class TaskAttentionalModule(nn.Module):
    """TaskAttentionalModule
    An attention model that utilises two tensor with identical number of feature channels to enhance the common features.

    Built based on the TAM module described in:
    https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenyu_Zhang_Joint_Task-Recursive_Learning_ECCV_2018_paper.pdf
    """

    def __init__(
        self,
        channels_in: int,
        kernel_size: int | tuple = (1, 1),
        padding: int | tuple = 0,
    ):
        """Inits :class:`TaskAttentionalModule`.

        Parameters
        ----------
        channels_in : int
            Number of feature channels.
        kernel_size : int | tuple, optional
            Size of the convolutional kernel. Default is 1
        padding : int | tuple, optional
            Padding around all four sizes of the input. Default is 0.
        """
        super().__init__()
        self.balance_conv1 = nn.Conv2d(
            int(channels_in * 2), int(channels_in), kernel_size=kernel_size, padding=padding
        )
        self.balance_conv2 = nn.Conv2d(int(channels_in), int(channels_in), kernel_size=kernel_size, padding=padding)
        self.residual_block = ResidualBlock(int(channels_in), int(channels_in))
        self.fc = nn.Conv2d(int(channels_in * 2), channels_in, kernel_size=kernel_size, padding=padding)

    def forward(self, rec_features: torch.Tensor, seg_features: torch.Tensor) -> torch.Tensor:
        """Forward :class:`TaskAttentionModule`.

        Parameters
        ----------
        rec_features : torch.Tensor
            Tensor of the reconstruction features with size [batch_size, feature_channels, height, width]
        seg_features : torch.Tensor
            Tensor of the segmentation features with size [batch_size, feature_channels, height, width]

        Returns
        -------
        new_rec_features : torch.Tensor
            Tensor of the optimised reconstruction features with size [batch_size, feature_channels, height, width]
        """
        # Balance unit
        concat_features = torch.cat((rec_features, seg_features), dim=1)
        balance_tensor = torch.sigmoid(self.balance_conv1(concat_features))
        balanced_output = self.balance_conv2(balance_tensor * rec_features + (1 - balance_tensor) * seg_features)
        # Conv-deconvolution layers for spatial attention
        res_block = torch.sigmoid(self.residual_block(balanced_output))
        # Generate gated features
        gated_rec_features = (1 + res_block) * rec_features
        gated_segmentation_features = (1 + res_block) * seg_features
        # Concatenate and apply convolutional layer
        concatenated_features = torch.cat((gated_rec_features, gated_segmentation_features), dim=1)
        output = self.fc(concatenated_features)
        return output


class ResidualBlock(nn.Module):
    """ResidualBlock
    A residual block with batch normalization and ReLU activation functions.
    Copied and adapted from:
    https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/blob/master/Residual-Attention-Network/model/basic_layers.py
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int | tuple = 1):
        """Inits :class:`ResidualBlock`.

        Parameters
        ----------
        input_channels : int
            Input number of feature channels
        output_channels : int
            Output number of feature channels
        stride : int | tuple, optional
            Stide of convolution. Default is 1
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, int(output_channels / 4), 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(output_channels / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(output_channels / 4), int(output_channels / 4), 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(output_channels / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(int(output_channels / 4), output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, input_features) -> torch.Tensor:
        """Forward :class:`ResidualBlock`.

        Parameters
        ----------
        input_features : torch.Tensor
            Tensor of the combined features with size [batch_size, feature_channels, height, width]

        Returns
        -------
        output_features : torch.Tensor
            Tensor of the optimised combined features with size [batch_size, feature_channels, height, width]
        """
        residual = input_features
        out = self.bn1(input_features)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        output_features += residual
        return output_features
