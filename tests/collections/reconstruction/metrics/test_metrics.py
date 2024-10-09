# coding=utf-8
__author__ = "Tim Paquaij"
import numpy as np
from atommic.collections.reconstruction.metrics import mse, nmse, psnr, ssim, haarpsi


class TestRecMetrics:
    np.random.seed(1)

    def test_mse_score(self):
        y = (np.random.rand(10, 256, 256) * 2) - 1
        y_norm = np.abs(y / np.max(np.abs(y)))
        x = (np.random.rand(10, 256, 256) * 2) - 1
        x_norm = np.abs(x / np.max(np.abs(x)))
        mse(x_norm, y_norm)

    def test_nmse_score(self):
        y = (np.random.rand(10, 256, 256) * 2) - 1
        y_norm = np.abs(y / np.max(np.abs(y)))
        x = (np.random.rand(10, 256, 256) * 2) - 1
        x_norm = np.abs(x / np.max(np.abs(x)))
        nmse(x_norm, y_norm)

    def test_psnr_score(self):
        y = (np.random.rand(10, 256, 256) * 2) - 1
        y_norm = np.abs(y / np.max(np.abs(y)))
        x = (np.random.rand(10, 256, 256) * 2) - 1
        x_norm = np.abs(x / np.max(np.abs(x)))
        psnr(x_norm, y_norm)

    def test_ssim_score(self):
        y = (np.random.rand(10, 256, 256) * 2) - 1
        y_norm = np.abs(y / np.max(np.abs(y)))
        x = (np.random.rand(10, 256, 256) * 2) - 1
        x_norm = np.abs(x / np.max(np.abs(x)))
        ssim(x_norm, y_norm)

    def test_haarpsi_score(self):
        np.random.seed(1)
        y = (np.random.rand(10, 256, 256) * 2) - 1
        y_norm = np.abs(y / np.max(np.abs(y)))
        x = (np.random.rand(10, 256, 256) * 2) - 1
        x_norm = np.abs(x / np.max(np.abs(x)))
        haarpsi(x_norm, y_norm)
