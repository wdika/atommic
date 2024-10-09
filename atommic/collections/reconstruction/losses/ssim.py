# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import torch
import torch.nn.functional as F

from atommic.core.classes.loss import Loss


class SSIMLoss(Loss):
    """Computes the (1-) SSIM loss between two tensors.

    Examples
    --------
    >>> from atommic.collections.reconstruction.losses.ssim import SSIMLoss
    >>> import torch
    >>> loss = SSIMLoss(win_size=7, k1=0.01, k2=0.03)
    >>> loss(X=torch.rand(1, 1, 256, 256), Y=torch.rand(1, 1, 256, 256), data_range=torch.tensor([1.]))
    tensor(0.9872)
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """Inits :class:`SSIMLoss`.

        Parameters
        ----------
        win_size : int, optional
            Window size for SSIM calculation.
        k1 : float, optional
            k1 parameter for SSIM calculation.
        k2 : float, optional
            k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor = None):
        """Forward pass of :class:`SSIMLoss`.

        Parameters
        ----------
        X : torch.Tensor
            First input tensor.
        Y : torch.Tensor
            Second input tensor.
        data_range : torch.Tensor
            Data range of the input tensors. If ``None``, it is computed as the maximum range of the input tensors.
            Default is ``None``.
        """
        if not isinstance(self.w, torch.Tensor):  # type: ignore  # pylint: disable=access-member-before-definition
            raise AssertionError

        if X.dim() == 3:
            X = X.unsqueeze(1)
        if Y.dim() == 3:
            Y = Y.unsqueeze(1)

        # This is necessary to first assign self.w to CUDA and then in case of fp32 to avoid RuntimeError: Inference
        # tensors cannot be saved for backward.
        self.w = self.w.to(Y).clone()  # type: ignore

        if data_range is None:
            data_range = torch.tensor([max(X.max() - X.min(), Y.max() - Y.min())]).to(Y)
        if isinstance(data_range, int):
            data_range = torch.tensor([data_range]).to(Y)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (2 * ux * uy + C1, 2 * vxy + C2, ux**2 + uy**2 + C1, vx + vy + C2)
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()
