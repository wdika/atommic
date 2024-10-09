# coding=utf-8
__author__ = "Tim Paquaij"

import pytest
import torch
from atommic.collections.reconstruction.losses import SSIMLoss, HaarPSILoss


class TestIQLosses:

    def test_ssim_loss(self):
        y = (torch.rand(10, 256, 256, requires_grad=False) * 2) - 1
        y_norm = torch.abs(y / torch.max(torch.abs(y)))
        x = (torch.rand(10, 256, 256, requires_grad=True) * 2) - 1
        x_norm = torch.abs(x / torch.max(torch.abs(x)))
        SSIM_loss = SSIMLoss()
        result = SSIM_loss(x_norm, y_norm)
        result.backward()

    def test_haarpsi_loss(self):
        y = (torch.rand(10, 256, 256, requires_grad=False) * 2) - 1
        y_norm = torch.abs(y / torch.max(torch.abs(y)))
        x = (torch.rand(10, 256, 256, requires_grad=True) * 2) - 1
        x_norm = torch.abs(x / torch.max(torch.abs(x)))
        # Only supports image with pixel values above 0
        HaarPSI_loss = HaarPSILoss()
        result = HaarPSI_loss(x_norm, y_norm)
        result.backward()
