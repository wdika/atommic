# coding=utf-8
__author__ = "Tim Paquaij"
import torch
import torch.nn as nn


class SASG(nn.Module):
    """Spatial-Adapted Semantic Guidance
    An attention model based on segmentation probabilities to enhance reconstruction features
    Built based on the SASG module descibed in:
    https://www.sciencedirect.com/science/article/abs/pii/S0169260724000415?via%3Dihub
    """

    def __init__(
        self,
        channels_rec: int,
        channels_seg: int,
        kernel_size: int | tuple = (1, 1),
        padding: int | tuple = 0,
    ):
        """Inits :class:`SASG`.

        Parameters
        ----------
        channels_rec : int
            Number of reconstruction feature channels.
        channels_seg : int
            Number of segmentation classes.
        kernel_size : int | tuple, optional
            Size of the convolutional kernel. Default is 1
        padding : int | tuple, optional
            Padding around all four sizes of the input. Default is 0.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels_rec, channels_rec, kernel_size=kernel_size, padding=padding)
        self.spade = SPADE(channels_rec, channels_seg, kernel_size, padding)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, rec_features: torch.Tensor, seg_features: torch.Tensor) -> torch.Tensor:
        """Forward :class:`SASG`.

        Parameters
        ----------
        rec_features : torch.Tensor
            Tensor of the reconstruction features with size [batch_size, feature_channels, height, width]
        seg_features : torch.Tensor
            Tensor of the segmentation features with size [batch_size, nr_classes, height, width]

        Returns
        -------
        new_rec_features : torch.Tensor
            Tensor of the optimised reconstruction features with size [batch_size, feature_channels, height, width]
        """

        hidden_layers_features_s = self.spade(rec_features, seg_features)
        hidden_layers_features_s = self.conv(self.act(hidden_layers_features_s))
        hidden_layers_features_s = self.spade(hidden_layers_features_s, seg_features)
        hidden_layers_features_s = self.conv(self.act(hidden_layers_features_s))
        new_rec_features = hidden_layers_features_s + rec_features
        return new_rec_features


class SPADE(nn.Module):

    def __init__(
        self,
        channels_rec: int,
        channels_seg: int,
        kernel_size: int | tuple = (1, 1),
        padding: int | tuple = 0,
    ):
        """Inits :class:`SPADE`.

        Parameters
        ----------
        channels_rec : int
            Number of reconstruction feature channels.
        channels_seg : int
            Number of segmentation classes.
        kernel_size : int | tuple, optional
            Size of the convolutional kernel. Default is 1
        padding : int | tuple, optional
            Padding around all four sizes of the input. Default is 0.
        """
        super().__init__()
        self.conv_1 = nn.Conv2d(channels_seg, channels_seg, kernel_size=kernel_size, padding=padding)
        self.conv_2 = nn.Conv2d(channels_seg, channels_rec, kernel_size=kernel_size, padding=padding)
        self.instance = nn.InstanceNorm2d(channels_rec)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, rec_features, seg_features) -> torch.Tensor:
        """Forward :class:`SPADE`.

        Parameters
        ----------
        rec_features : torch.Tensor
            Tensor of the reconstruction features with size [batch_size, feature_channels, height, width]
        seg_features : torch.Tensor
            Tensor of the segmentation features with size [batch_size, nr_classes, height, width]

        Returns
        -------
        new_rec_features : torch.Tensor
            Tensor of the optimised reconstruction features with size [batch_size, feature_channels, height, width]
        """
        hidden_layers_features = self.instance(rec_features)
        segmentation_prob = self.act(self.conv_1(seg_features))
        new_rec_features = torch.mul(self.conv_2(segmentation_prob), hidden_layers_features) + self.conv_2(
            segmentation_prob
        )
        return new_rec_features
