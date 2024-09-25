# coding=utf-8

import torch

from atommic.collections.common.parts.transforms import RandomFlipper


class TestRandomFlipper:
    # Tests that flipping is applied correctly on a tensor with default parameters.
    def test_flipping_default_parameters(self):
        data = torch.randn(1, 15, 320, 320)
        flipping = RandomFlipper(axes=(2, 3), flip_probability=1.0)
        flipped_data = flipping(data)
        assert flipped_data.shape == (1, 15, 320, 320)

    # Tests that flipping is applied correctly on a real tensor without coils or singlecoil with default parameters.
    def test_flipping_no_coils_default_parameters(self):
        data = torch.randn(1, 320, 320)
        flipping = RandomFlipper(axes=(0), flip_probability=1.0)
        flipped_data = flipping(data)
        assert flipped_data.shape == (1, 320, 320)

    # Tests that flipping is applied correctly on a tensor with apply_backward_transform=True.
    def test_flipping_apply_backward_transform(self):
        data = torch.randn(1, 15, 320, 320, 2)
        flipping = RandomFlipper(axes=(2, 3), flip_probability=1.0, apply_ifft=True)
        flipped_data = flipping(data, apply_backward_transform=True)
        assert flipped_data.shape == (1, 15, 320, 320, 2)

    # Tests that flipping is applied correctly on a tensor with apply_forward_transform=True.
    def test_flipping_apply_forward_transform(self):
        data = torch.randn(1, 15, 320, 320, 2)
        flipping = RandomFlipper(axes=(2, 3), flip_probability=0.5, apply_ifft=True)
        flipped_data = flipping(data, apply_forward_transform=True)
        assert flipped_data.shape == (1, 15, 320, 320, 2)
