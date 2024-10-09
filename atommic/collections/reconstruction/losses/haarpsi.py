# coding=utf-8
__author__ = "Tim Paquaij"

# Parts of the code have been taken and adapted from https://github.com/photosynthesis-team/piq/blob/master/piq/haarpsi.py
# Variables have been optimised for medical images based on :https://github.com/ideal-iqa/haarpsi-pytorch?tab=MIT-1-ov-file
from typing import Union, List, Tuple, Optional
import torch
import functools
import torch.nn.functional as F
from atommic.core.classes.loss import Loss


class HaarPSILoss(Loss):
    """Creates a criterion that measures  Haar Wavelet-Based Perceptual Similarity loss between
    each element in the input and target.

    References
    ----------
    Reisenhofer, R., Bosse, S., Kutyniok, G., & Wiegand, T. (2018). A Haar wavelet-based perceptual similarity index for image 
    quality assessment. Signal Processing: Image Communication, 61, 33-43.
    http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf

    Code from authors on MATLAB and Python:

    https://github.com/rgcda/haarpsi

    Examples
    ----------
    # >>> loss = HaarPSILoss()
    #>>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
    #>>> y = torch.rand(3, 3, 256, 256)
    #>>> output = loss(x, y)
    #>>> output.backward()
    """

    def __init__(
        self,
        reduction: str = 'mean',
        scales: int = 3,
        subsample: bool = True,
        c: float = 5,
        alpha: float = 4.2,
    ):
        super().__init__()
        self.reduction = reduction
        self.scales = scales
        self.subsample = subsample
        self.c = c
        self.alpha = alpha
        """Inits :class:`HaarPSILoss`.
        
        Parameters
        ----------
        reduction: str, optional
            ``none`` | ``mean`` | ``sum``. Default is ``mean``.
        scales: int, optional
            Number of Haar wavelets used for image decomposition. Default is 3.
        subsample: bool, optinal
            Flag to apply average pooling before HaarPSI computation. Default is True.
        c: float, optional
            Constant from the paper (30). Default for medical images is 5.
        alpha: float, optional
            Exponent used for similarity maps weightning (4.2). Default for medical images is 5.8.
        data_range: int, optional
            Maximum value range of images (usually 1.0 or 255). Default is 1.0.
        """
        self.haarpsi_func = functools.partial(
            haarpsi_func,
            scales=self.scales,
            subsample=self.subsample,
            c=self.c,
            alpha=self.alpha,
            reduction=self.reduction,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of :class:`HaarPSILoss`.

        Parameters
        ----------
        x : torch.Tensor
            First input tensor.
        y : torch.Tensor
            Second input tensor.
        data_range : torch.Tensor
            Data range of the input tensors. If ``None``, it is computed as the maximum value of both input tensors.
            Default is ``None``.

        Returns
        -------
        torch.Tensor
            HaarPSI loss tensor.
        """
        if data_range is None:
            data_range = torch.tensor([max(x.max(), y.max())]).to(y)
        if isinstance(data_range, int):
            data_range = torch.tensor([data_range]).to(y)

        return 1.0 - self.haarpsi_func(x=x, y=y, data_range=data_range)


def haarpsi_func(
    x: torch.Tensor,
    y: torch.Tensor,
    reduction: str = 'mean',
    data_range: Union[int, float] = 1.0,
    scales: int = 3,
    subsample: bool = True,
    c: float = 5,
    alpha: float = 4.2,
) -> torch.Tensor:
    """Compute Haar Wavelet-Based Perceptual Similarity

    References
    ----------

    Reisenhofer, R., Bosse, S., Kutyniok, G., & Wiegand, T. (2018). A Haar wavelet-based perceptual similarity index for image 
    quality assessment. Signal Processing: Image Communication, 61, 33-43.
    http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf

    Code from authors on MATLAB and Python

    https://github.com/rgcda/haarpsi


    Parameters
    ----------
    reduction: str, optional
        ``none`` | ``mean`` | ``sum``. Default is ``mean``.
    scales: int, optional
        Number of Haar wavelets used for image decomposition. Default is 3.
    subsample: bool, optinal
        Flag to apply average pooling before HaarPSI computation. Default is True.
    c: float, optional
        Constant from the paper (30). Default for medical images is 5.
    alpha: float, optional
        Exponent used for similarity maps weightning (4.2). Default for medical images is 5.8.
    data_range: int, optional
        Maximum value range of images (usually 1.0 or 255). Default is 1.0.

    Returns
    -------
    torch.Tensor
        HaarPSI metric tensor.
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if y.dim() == 3:
        y = y.unsqueeze(1)

    _validate_input([x, y], dim_range=(4, 4), data_range=[0, data_range])

    # Assert minimal image size
    kernel_size = 2 ** (scales + 1)
    if x.size(-1) < kernel_size or x.size(-2) < kernel_size:
        raise ValueError(
            f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
            f'Kernel size: {kernel_size}'
        )

    # Rescale images
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255

    num_channels = x.size(1)
    # Convert RGB to YIQ color space https://en.wikipedia.org/wiki/YIQ
    if num_channels == 3:
        x_yiq = rgb2yiq(x)
        y_yiq = rgb2yiq(y)
    else:
        x_yiq = x
        y_yiq = y

    # Downscale input to simulates the typical distance between an image and its viewer.
    if subsample:
        up_pad = 0
        down_pad = max(x.shape[2] % 2, x.shape[3] % 2)
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        x_yiq = F.pad(x_yiq, pad=pad_to_use)
        y_yiq = F.pad(y_yiq, pad=pad_to_use)

        x_yiq = F.avg_pool2d(x_yiq, kernel_size=2, stride=2, padding=0)
        y_yiq = F.avg_pool2d(y_yiq, kernel_size=2, stride=2, padding=0)

    # Haar wavelet decomposition
    coefficients_x, coefficients_y = [], []
    for scale in range(scales):
        kernel_size = 2 ** (scale + 1)
        h_filter = haar_filter(kernel_size, dtype=x.dtype, device=x.device)
        kernels = torch.stack([h_filter, h_filter.transpose(-1, -2)])

        # Asymmetrical padding due to even kernel size. Matches MATLAB conv2(A, B, 'same')
        upper_pad = kernel_size // 2 - 1
        bottom_pad = kernel_size // 2
        pad_to_use = [upper_pad, bottom_pad, upper_pad, bottom_pad]
        coeff_x = torch.nn.functional.conv2d(F.pad(x_yiq[:, :1], pad=pad_to_use, mode='constant'), kernels)
        coeff_y = torch.nn.functional.conv2d(F.pad(y_yiq[:, :1], pad=pad_to_use, mode='constant'), kernels)

        coefficients_x.append(coeff_x)
        coefficients_y.append(coeff_y)

    # Shape (N, {scales * 2}, H, W)
    coefficients_x = torch.cat(coefficients_x, dim=1)
    coefficients_y = torch.cat(coefficients_y, dim=1)

    # Low-frequency coefficients used as weights
    # Shape (N, 2, H, W)
    weights = torch.max(torch.abs(coefficients_x[:, 4:]), torch.abs(coefficients_y[:, 4:]))

    # High-frequency coefficients used for similarity computation in 2 orientations (horizontal and vertical)
    sim_map = []
    for orientation in range(2):
        magnitude_x = torch.abs(coefficients_x[:, (orientation, orientation + 2)])
        magnitude_y = torch.abs(coefficients_y[:, (orientation, orientation + 2)])
        sim_map.append(similarity_map(magnitude_x, magnitude_y, constant=c).sum(dim=1, keepdims=True) / 2)

    if num_channels == 3:
        pad_to_use = [0, 1, 0, 1]
        x_yiq = F.pad(x_yiq, pad=pad_to_use)
        y_yiq = F.pad(y_yiq, pad=pad_to_use)
        coefficients_x_iq = torch.abs(F.avg_pool2d(x_yiq[:, 1:], kernel_size=2, stride=1, padding=0))
        coefficients_y_iq = torch.abs(F.avg_pool2d(y_yiq[:, 1:], kernel_size=2, stride=1, padding=0))

        # Compute weights and simmilarity
        weights = torch.cat([weights, weights.mean(dim=1, keepdims=True)], dim=1)
        sim_map.append(similarity_map(coefficients_x_iq, coefficients_y_iq, constant=c).sum(dim=1, keepdims=True) / 2)

    sim_map = torch.cat(sim_map, dim=1)

    # Calculate the final score
    eps = torch.finfo(sim_map.dtype).eps
    score = (((sim_map * alpha).sigmoid() * weights).sum(dim=[1, 2, 3]) + eps) / (
        torch.sum(weights, dim=[1, 2, 3]) + eps
    )
    # Logit of score
    score = (torch.log(score / (1 - score)) / alpha) ** 2

    return _reduce(score, reduction)


def haar_filter(kernel_size: int, device: Optional[str] = None, dtype: Optional[type] = None) -> torch.Tensor:
    """Creates Haar kernel

    Parameters
    ----------
    kernel_size: int
        Size of the kernel.
    device: str, optional
        Target device for kernel generation. Default is ``None``.
    dtype: type, optional
        type of tensor to be used

    Returns
    -------
    torch.Tensor
        kernel Tensor with shape (1, kernel_size, kernel_size)
    """
    kernel = torch.ones((kernel_size, kernel_size), device=device, dtype=dtype) / kernel_size
    kernel[kernel_size // 2 :, :] = -kernel[kernel_size // 2 :, :]
    return kernel.unsqueeze(0)


def similarity_map(map_x: torch.Tensor, map_y: torch.Tensor, constant: float, alpha: float = 0.0) -> torch.Tensor:
    """Compute similarity_map between two tensors using Dice-like equation.

    Parameters
     ----------
     map_x: torch.Tensor
        Tensor with map to be compared.
     map_y: torch.Tensor
        Tensor with map to be compared.
     c: float
        Constant from the paper.
     alpha: float, optional
        Exponent used for similarity map weightning. Default is 0.

    Returns
    -------
    torch.Tensor
        Simmilarity map.
    """
    return (2.0 * map_x * map_y - alpha * map_x * map_y + constant) / (
        map_x**2 + map_y**2 - alpha * map_x * map_y + constant
    )


def rgb2yiq(x: torch.Tensor) -> torch.Tensor:
    """Convert a batch of RGB images to a batch of YIQ images

    Parameters
    ----------
    x: torch.Tensor
        Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns
    -------
    torch.Tensor
        Batch of images with shape (N, 3, H, W). YIQ colour space.
    """
    yiq_weights = torch.tensor(
        [[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]], dtype=x.dtype, device=x.device
    ).t()
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq


def _reduce(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Reduce input in batch dimension if needed.

    Parameters
    ----------
    x: torh.Tensor
        Tensor with shape (N, *).
    reduction: str, optional
        ``none`` | ``mean`` | ``sum``. Default is ``mean``.

    Returns
    -------
    torch.Tensor with shape (*)
    """
    if reduction == 'none':
        return x
    elif reduction == 'mean':
        return x.mean(dim=0)
    elif reduction == 'sum':
        return x.sum(dim=0)
    else:
        raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")


def _validate_input(
    tensors: List[torch.Tensor],
    dim_range: Tuple[int, int] = (0, -1),
    data_range: Tuple[float, float] = (0.0, -1.0),
    size_range: Optional[Tuple[int, int]] = None,
    check_for_channels_first: bool = False,
) -> None:
    """Check that input(-s)  satisfies the requirements

    Parameters
    ------
    tensors: List[torch.Tensor],
        List with tensors to check for valite input.
    dim_range: Tuple[int, int], optional
        Allowed number of dimensions. Default is [0., -1.].
    data_range: Tuple[int, int], optional
        Allowed number of dimensions. Default is [0., -1.].
    size_range: Optional[Tuple[int, int]], optional
        Allowed size of the tensors. Default is None.
    check_for_channels: bool, optional
        Check if channels are in first dimension. Default is False.

    Returns
    -------
    None
    """

    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert torch.is_tensor(t), f'Expected torch.Tensor, got {type(t)}'
        assert t.device == x.device, f'Expected tensors to be on {x.device}, got {t.device}'

        if size_range is None:
            assert t.size() == x.size(), f'Expected tensors with same size, got {t.size()} and {x.size()}'
        else:
            assert (
                t.size()[size_range[0] : size_range[1]] == x.size()[size_range[0] : size_range[1]]
            ), f'Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}'

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f'Expected number of dimensions to be {dim_range[0]}, got {t.dim()}'
        elif dim_range[0] < dim_range[1]:
            assert (
                dim_range[0] <= t.dim() <= dim_range[1]
            ), f'Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}'

        if data_range[0] < data_range[1]:
            assert (
                data_range[0] <= t.min()
            ), f'Expected values to be greater or equal to {data_range[0]}, got {t.min()}'
            assert t.max() <= data_range[1], f'Expected values to be lower or equal to {data_range[1]}, got {t.max()}'

        if check_for_channels_first:
            channels_last = t.shape[-1] in {1, 2, 3}
            assert (
                not channels_last
            ), "Expected tensor to have channels first format, but got channels last. \
                Please permute channels (e.g. t.permute(0, 3, 1, 2) for 4D tensors) and rerun."
