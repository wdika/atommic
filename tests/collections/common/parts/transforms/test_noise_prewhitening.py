# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch

from atommic.collections.common.parts.transforms import NoisePreWhitening


class TestNoisePreWhitening:
    # Tests that noise pre-whitening is applied with default parameters.
    def test_apply_prewhitening_default_parameters(self):
        # Create an instance of NoisePreWhitening with default parameters
        prewhitening = NoisePreWhitening()

        # Create dummy data
        data = torch.randn([30, 100, 100], dtype=torch.complex64)
        data = torch.view_as_real(data)

        # Apply noise pre-whitening
        result = prewhitening(data)

        # Assert that the result is not equal to the input data
        assert not torch.allclose(result, data)

        # Assert that the shape of the result is the same as the input data
        assert result.shape == data.shape

    # Tests that noise pre-whitening is applied with find_patch_size=False and patch_size defined.
    def test_apply_prewhitening_find_patch_size_false_patch_size_defined(self):
        # Create an instance of NoisePreWhitening with find_patch_size=False and patch_size defined
        prewhitening = NoisePreWhitening(find_patch_size=False, patch_size=[10, 20, 30, 40])

        # Create dummy data
        data = torch.randn([30, 100, 100], dtype=torch.complex64)
        data = torch.view_as_real(data)

        # Apply noise pre-whitening
        result = prewhitening(data)

        # Assert that the result is not equal to the input data
        assert not torch.allclose(result, data)

        # Assert that the shape of the result is the same as the input data
        assert result.shape == data.shape

    # Tests that noise pre-whitening is applied with apply_backward_transform=True.
    def test_apply_prewhitening_apply_backward_transform_true(self):
        # Create an instance of NoisePreWhitening with apply_backward_transform=True
        prewhitening = NoisePreWhitening()

        # Create dummy data
        data = torch.randn([30, 100, 100], dtype=torch.complex64)
        data = torch.view_as_real(data)

        # Apply noise pre-whitening
        result = prewhitening(data, apply_backward_transform=True)

        # Assert that the result is not equal to the input data
        assert not torch.allclose(result, data)

        # Assert that the shape of the result is the same as the input data
        assert result.shape == data.shape

    # Tests that noise pre-whitening is applied with apply_forward_transform=True.
    def test_apply_prewhitening_apply_forward_transform_true(self):
        # Create an instance of NoisePreWhitening with apply_forward_transform=True
        prewhitening = NoisePreWhitening()

        # Create dummy data
        data = torch.randn([30, 100, 100], dtype=torch.complex64)
        data = torch.view_as_real(data)

        # Apply noise pre-whitening
        result = prewhitening(data, apply_forward_transform=True)

        # Assert that the result is not equal to the input data
        assert not torch.allclose(result, data)

        # Assert that the shape of the result is the same as the input data
        assert result.shape == data.shape
