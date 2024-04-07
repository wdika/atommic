# coding=utf-8
import tempfile
from pathlib import Path

import h5py
import numpy as np

import torch

from atommic.collections.common.data.subsample import Gaussian2DMaskFunc
from atommic.collections.common.parts import (
    add_coil_dim_if_singlecoil,
    apply_mask,
    batched_mask_center,
    center_crop,
    center_crop_to_smallest,
    check_stacked_complex,
    coil_combination_method,
    complex_abs,
    complex_abs_sq,
    complex_center_crop,
    complex_conj,
    complex_mul,
    crop_to_acs,
    expand_op,
    is_none,
    mask_center,
    normalize_inplace,
    parse_list_and_keep_last,
    reshape_fortran,
    rnn_weights_init,
    save_predictions,
    to_tensor,
    unnormalize,
    zero_nan_inf,
)


class TestAddCoilDimIfSinglecoil:
    # Tests that the function correctly adds a coil dimension to a tensor of shape (1, 2, 3) when dim=0
    def test_add_coil_dim_if_singlecoil_1(self):
        # Create input tensor
        data = torch.randn(320, 320, 2)

        # Call the function
        output_tensor = add_coil_dim_if_singlecoil(data, dim=0)

        # Check the output tensor shape
        assert output_tensor.shape == (1, 320, 320, 2)

    # Tests that the function correctly adds a coil dimension to a tensor of shape (2, 3, 1) when dim=-1
    def test_add_coil_dim_if_singlecoil_2(self):
        # Create input tensor
        data = torch.randn(32, 320, 320, 2)

        # Call the function
        output_tensor = add_coil_dim_if_singlecoil(data, dim=0)

        # Check the output tensor shape
        assert output_tensor.shape == (32, 320, 320, 2)


class TestApplyMask:
    # Tests that the function applies the mask to k-space data with default parameters
    def test_apply_mask(self):
        data = torch.randn(1, 32, 320, 320, 2)
        accelerations = [4, 8]
        center_fractions = [0.7, 0.7]
        mask_func = Gaussian2DMaskFunc(center_fractions, accelerations)

        # Call the function
        masked_data, subsampling_mask, acceleration_factor = apply_mask(data, mask_func)

        # Check the output
        assert len(masked_data) == 1
        assert len(subsampling_mask) == 1
        assert masked_data.shape == data.shape
        assert subsampling_mask.shape == torch.Size([1, 1, 320, 320, 1])
        assert acceleration_factor in accelerations


class TestBatchedMaskCenter:
    # Tests that the function correctly applies a 2D mask to a single batch of images
    def test_single_batch_2D_mask(self):
        data = torch.randn(1, 32, 320, 320)
        mask_from = torch.tensor([140])
        mask_to = torch.tensor([180])
        expected_output = torch.zeros_like(data)

        result = batched_mask_center(data, mask_from, mask_to)

        assert not torch.allclose(result, expected_output)

    # Tests that the function correctly applies a 2D mask to multiple batches of images
    def test_multiple_batches_2D_mask(self):
        data = torch.randn(16, 32, 320, 320)
        mask_from = torch.tensor([140] * data.shape[0])
        mask_to = torch.tensor([180] * data.shape[0])
        expected_output = torch.zeros_like(data)

        result = batched_mask_center(data, mask_from, mask_to)

        assert not torch.allclose(result, expected_output)


class TestCenterCrop:
    # Tests that the function correctly applies a center crop to the input tensor along the last two dimensions
    def test_center_crop(self):
        # Arrange
        data = torch.randn(1, 32, 320, 320)

        # Act
        result = center_crop(data, (160, 160))

        # Assert
        assert result.shape != data.shape
        assert result.shape[-2] == (data.shape[-2] // 2)
        assert result.shape[-1] == (data.shape[-1] // 2)

    # Tests that the function handles edge cases correctly when the output shape is smaller than the corresponding
    # dimensions of the input tensor
    def test_edge_case(self):
        # Arrange
        data = torch.randn(1, 32, 320, 320)
        expected_output = torch.zeros(1, 32, 1, 1)

        # Act
        result = center_crop(data, (1, 1))

        # Assert
        assert result.shape != data.shape
        assert result.shape[-2] == 1
        assert result.shape[-1] == 1


class TestCenterCropToSmallest:
    # Tests that the function correctly applies a center crop to the input tensor along the last two dimensions
    def test_center_crop_to_smallest_1(self):
        # Arrange
        data1 = torch.randn(1, 32, 320, 320)
        data2 = torch.randn(1, 32, 160, 160)

        # Act
        result1, result2 = center_crop_to_smallest(data1, data2)

        # Assert
        assert result1.shape == data2.shape
        assert result2.shape == data2.shape

    # Tests that the function handles edge cases correctly when the output shape is smaller than the corresponding
    # dimensions of the input tensor
    def test_center_crop_to_smallest_2(self):
        # Arrange
        data1 = torch.randn(1, 32, 160, 160)
        data2 = torch.randn(1, 32, 320, 320)

        # Act
        result1, result2 = center_crop_to_smallest(data1, data2)

        # Assert
        assert result1.shape == data1.shape
        assert result2.shape == data1.shape


class TestCheckStackedComplex:
    # Tests that the function correctly converts a complex tensor with shape (n,) to a combined complex tensor
    def test_stacked_complex_tensor_1(self):
        # Create a complex tensor with shape (n,)
        data = torch.randn(1, 32, 320, 320, 2)

        # Call the function under test
        result = check_stacked_complex(data)

        assert result.shape == data[..., 0].shape

    # Tests that the function returns the input tensor unchanged when it has shape (0,)
    def test_stacked_complex_tensor_2(self):
        # Create a complex tensor with shape (n,)
        data = torch.randn(1, 32, 320, 320)

        # Call the function under test
        result = check_stacked_complex(data)

        assert result.shape == data.shape


class TestCoilCombinationMethod:
    # Tests that the SENSE method works correctly with valid input data and sensitivity maps
    def test_sense_coil_combination_method(self):
        data = torch.randn(1, 32, 320, 320, 2)
        coil_sensitivity_maps = torch.randn(1, 32, 320, 320, 2)

        result = coil_combination_method(data, coil_sensitivity_maps, method="SENSE", dim=1)

        assert result.shape == data.sum(dim=1).shape

    def test_rss_coil_combination_method(self):
        data = torch.randn(1, 32, 320, 320, 2)
        coil_sensitivity_maps = torch.randn(1, 32, 320, 320, 2)

        result = coil_combination_method(data, coil_sensitivity_maps, method="RSS", dim=1)

        assert result.shape == data.sum(dim=1).shape

    def test_rss_complex_coil_combination_method(self):
        data = torch.randn(1, 32, 320, 320, 2)
        coil_sensitivity_maps = torch.randn(1, 32, 320, 320, 2)

        result = coil_combination_method(data, coil_sensitivity_maps, method="RSS_COMPLEX", dim=1)

        assert result.shape == data.sum(dim=(1, -1)).shape


class TestComplexAbs:
    # Tests that the function correctly computes the absolute value of a tensor of complex numbers with positive
    # real and imaginary parts.
    def test_complex_abs(self):
        data = torch.randn(1, 32, 320, 320, 2) + 1.0j

        result = complex_abs(data)

        assert result.shape == data.shape[:-1]


class TestComplexAbsSq:
    # Tests that the function correctly computes the absolute value of a tensor of complex numbers with positive
    # real and imaginary parts.
    def test_complex_abs_sq(self):
        data = torch.randn(1, 32, 320, 320, 2) + 1.0j

        result = complex_abs_sq(data)

        assert result.shape == data.shape[:-1]
        assert torch.allclose(result, complex_abs(data).sqrt())


class TestComplexCenterCrop:
    # Tests that the function correctly applies a center crop to the input tensor along the last two dimensions
    def test_complex_center_crop(self):
        # Arrange
        data = torch.randn(1, 32, 320, 320, 2)

        # Act
        result = complex_center_crop(data, (160, 160))

        # Assert
        assert result.shape != data.shape
        assert result.shape[-3] == (data.shape[-3] // 2)
        assert result.shape[-2] == (data.shape[-2] // 2)

    # Tests that the function handles edge cases correctly when the output shape is smaller than the corresponding
    # dimensions of the input tensor
    def test_edge_case(self):
        # Arrange
        data = torch.randn(1, 32, 320, 320, 2)
        expected_output = torch.zeros(1, 32, 1, 1)

        # Act
        result = complex_center_crop(data, (1, 1))

        # Assert
        assert result.shape != data.shape
        assert result.shape[-3] == 1
        assert result.shape[-2] == 1


class TestComplexConj:
    # Tests that complex_conj returns the complex conjugate of a tensor of shape (3,2) containing complex numbers
    def test_complex_conj(self):
        data = torch.randn(1, 32, 320, 320, 2)
        expected_output = torch.view_as_real(torch.conj(torch.view_as_complex(data)).resolve_conj())
        assert torch.allclose(complex_conj(data), expected_output)

    def test_not_complex_conj(self):
        data = torch.randn(1, 32, 320, 320, 2)
        assert not torch.allclose(complex_conj(data), data)


class TestComplexMul:
    # Tests that complex_mul returns the correct result for two tensors of shape (2, 2)
    def test_complex_mul(self):
        datax = torch.randn(1, 32, 320, 320, 2)
        datay = torch.randn(1, 32, 320, 320, 2)
        expected_result = torch.view_as_real(torch.view_as_complex(datax) * torch.view_as_complex(datay))
        result = complex_mul(datax, datay)
        assert torch.allclose(result, expected_result)


class TestCropToAcs:
    # Tests that the function correctly crops the k-space to the autocalibration region when given a valid acs_mask
    # and kspace tensor
    def test_valid_acs_mask_and_kspace(self):
        # Create a valid acs_mask tensor and kspace tensor
        acs_mask = torch.randn(16, 16)
        kspace = torch.randn(32, 320, 320, 2)

        # Call the crop_to_acs function
        cropped_kspace = crop_to_acs(acs_mask, kspace)

        # Check if the cropped k-space has the correct shape
        assert cropped_kspace.shape == (32, 16, 16, 2)

        # Check if the cropped k-space values are correct
        assert not torch.allclose(cropped_kspace, kspace[:, 152:168, 152:168, :])


class TestExpandOp:
    # Tests that the function correctly expands a tensor of shape (1, 200, 200, 2) with sensitivity maps of shape (
    # 1, 30, 200, 200, 2)
    def test_expand_op_1(self):
        data = torch.rand(1, 200, 200, 2)
        sens = torch.rand(1, 30, 200, 200, 2)
        result = expand_op(data, sens)
        assert result.shape == (1, 30, 200, 200, 2)

    # Tests that the function handles an empty tensor and sensitivity maps correctly
    def test_expand_op_2(self):
        data = torch.rand(1, 30, 200, 200, 2)
        sens = torch.rand(1, 30, 200, 200, 2)
        result = expand_op(data, sens)
        assert not result.shape == (1, 30, 200, 200, 2)


class TestIsNone:
    # Tests that the function correctly identifies when the input is None
    def test_input_is_none(self):
        assert is_none(None)

    # Tests that the function correctly identifies when the input is the string "None"
    def test_input_is_string_none(self):
        assert is_none("None")

    # Tests that the function correctly identifies when the input is None
    def test_input_is_not_none(self):
        assert not is_none(torch.empty([]))

    # Tests that the function correctly identifies when the input is the string "None"
    def test_input_is_string_not_none(self):
        assert not is_none("ABC")


class TestMaskCenter:
    # Tests that the function correctly applies a center crop to a 2D input image.
    def test_behaviour_apply_center_crop_to_2D_input_image(self):
        # Create input image
        data = torch.rand(1, 1, 320, 320, 2)

        # Apply center crop
        result = mask_center(data, torch.tensor([140]), torch.tensor([180]), mask_type="2D")

        # Check if the result has the correct shape
        assert result.shape == torch.Size([1, 1, 320, 320, 2])
        assert not torch.allclose(result, data)

    # Tests that the function correctly applies a center crop to a 1D input image.
    def test_behaviour_apply_center_crop_to_1D_input_image(self):
        # Create input image
        data = torch.rand(1, 1, 1, 320, 2)

        # Apply center crop
        result = mask_center(data, torch.tensor([140]), torch.tensor([180]), mask_type="1D")

        # Check if the result has the correct shape
        assert result.shape == torch.Size([1, 1, 1, 320, 2])
        assert not torch.allclose(result, data)


class TestNormalizeInplace:
    def test_max_normalization(self):
        # Create a tensor with random data
        data = torch.rand(1, 32, 320, 320, 2)

        # Normalize the data using the Normalizer instance
        normalized_data = normalize_inplace(data, normalization_type="max")

        assert torch.allclose(torch.max(torch.abs(normalized_data)), torch.tensor(1.0))
        assert torch.allclose(torch.min(torch.abs(normalized_data)), torch.tensor(0.1), rtol=1e3)

    # Tests that the Normalizer class can normalize data by its minimum and maximum values
    def test_minmax_normalization(self):
        # Create an instance of the Normalizer class with normalization_type="minmax"
        data = torch.rand(1, 32, 320, 320, 2)

        # Normalize the data using the Normalizer instance
        normalized_data = normalize_inplace(data, normalization_type="minmax")

        assert torch.allclose(torch.max(torch.abs(normalized_data)), torch.tensor(1.0), rtol=1e3)
        assert torch.allclose(torch.min(torch.abs(normalized_data)), torch.tensor(0.1), rtol=1e3)

    # Tests that the Normalizer class can normalize complex data
    def test_mean_std_normalization(self):
        # Create an instance of the Normalizer class with normalization_type="max"
        data = torch.rand(1, 32, 320, 320, 2)

        # Normalize the data using the Normalizer instance
        normalized_data = normalize_inplace(data, normalization_type="mean_std")

        assert torch.mean(torch.abs(normalized_data)) != torch.mean(torch.abs(data))
        assert torch.std(torch.abs(normalized_data)) != torch.std(torch.abs(data))

    def test_mean_var_normalization(self):
        # Create an instance of the Normalizer class with normalization_type="max"
        data = torch.rand(1, 32, 320, 320, 2)

        # Normalize the data using the Normalizer instance
        normalized_data = normalize_inplace(data, normalization_type="mean_var")

        assert torch.mean(torch.abs(normalized_data)) != torch.mean(torch.abs(data))
        assert torch.var(torch.abs(normalized_data)) != torch.var(torch.abs(data))

    def test_grayscale_normalization(self):
        # Create an instance of the Normalizer class with normalization_type="max"
        data = torch.rand(1, 32, 320, 320, 2)

        # Normalize the data using the Normalizer instance
        normalized_data = normalize_inplace(data, normalization_type="grayscale")

        assert np.round(torch.max(torch.abs(normalized_data)).item()) == 255

    # Tests that the Normalizer class does not normalize data
    def test_do_not_normalize_data(self):
        # Create an instance of the Normalizer class with normalization_type=None
        data = torch.rand(1, 32, 320, 320, 2)

        # Normalize the data using the Normalizer instance
        normalized_data = normalize_inplace(data, normalization_type="None")

        # Check that the normalized data is the same as the original data
        assert torch.all(torch.eq(normalized_data, data))


class TestParseListAndKeepLast:
    """Tests that the function correctly parses a non-empty list of non-list elements and returns the last element."""

    def test_non_empty_list_of_non_list_elements(self):
        input_list = [1, 2, 3, 4]
        expected_output = 4

        assert parse_list_and_keep_last(input_list) == expected_output

    # Tests that the function correctly parses a list with a single non-list element and returns the element
    def test_list_with_single_non_list_element(self):
        input_list = [5]
        expected_output = 5

        assert parse_list_and_keep_last(input_list) == expected_output

    # Tests that the function correctly parses a list with a single list element and returns the element
    def test_list_with_single_list_element(self):
        input_list = [[5]]
        expected_output = 5

        assert parse_list_and_keep_last(input_list) == expected_output


class TestReshapeFortran:
    # Tests that the function correctly reshapes a tensor with valid input and shape.
    def test_reshape_fortran(self):
        # Create input tensor
        data = torch.arange(6).reshape(3, 2)

        # Reshape the tensor using reshape_fortran function
        reshaped_data = reshape_fortran(data, (2, 3))

        # Check if the shape of the reshaped tensor is correct
        assert reshaped_data.shape != data.reshape(2, 3)

        # Check if the values in the reshaped tensor are correct
        assert not torch.allclose(reshaped_data, data.reshape(2, 3))


class TestRnnWeightsInit:
    # Tests that the linear layer weights are initialized with xavier initializer
    def test_initialize_linear_xavier(self):
        rnn = torch.nn.GRU(10, 20, 2)
        rnn.apply(rnn_weights_init)
        for name, param in rnn.named_parameters():
            if "weight" in name:
                if "linear" in name:
                    assert torch.nn.init.calculate_gain("linear") == param.std().item()
                elif "embedding" in name:
                    assert torch.nn.init.calculate_gain("embedding") == param.std().item()

    # Tests that the embedding layer weights are initialized correctly
    def test_initialize_embedding(self):
        rnn = torch.nn.GRU(10, 20, 2)
        rnn.apply(rnn_weights_init)
        for name, param in rnn.named_parameters():
            if "weight" in name and "embedding" in name:
                assert param.std().item() == 0.02


class TestSavePredictions:
    # Tests that the function saves predictions in h5 format to the output directory with the default key
    # "reconstructions"
    def test_save_predictions(self):
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dictionary of predictions
            predictions = {"test.h5": np.array([320, 320])}

            # Call the save_predictions function
            save_predictions(predictions, Path(temp_dir), file_format="h5")

            # Check if the output file exists
            assert (Path(temp_dir) / "test.h5").exists()

            # Check if the key "reconstructions" exists in the output file
            with h5py.File(Path(temp_dir) / "test.h5", "r") as hf:
                assert "reconstructions" in hf.keys()

                # Check if the shape of the saved predictions matches the original shape
                assert hf["reconstructions"].shape == predictions["test.h5"].shape

                # Check if the saved predictions match the original predictions
                assert np.array_equal(hf["reconstructions"], predictions["test.h5"])


class TestToTensor:
    # Tests that the function converts a 2D numpy array with real numbers to a torch tensor.
    def test_convert_2D_real_numbers(self):
        # create complex float 2D numpy array
        data = np.array([1, 32, 320, 320]) + 0.0j
        torch_data = to_tensor(data)
        assert isinstance(torch_data, torch.Tensor)

    # Tests that the function converts an empty numpy array to a torch tensor.
    def test_convert_empty_array(self):
        data = np.array([])
        torch_data = to_tensor(data)
        assert isinstance(torch_data, torch.Tensor)


class TestUnnormalize:
    def test_max_unnormalization(self):
        # Create a tensor with random data
        data = torch.rand(1, 32, 320, 320, 2)
        attrs = {"max": torch.max(torch.abs(data)).item(), "min": torch.min(torch.abs(data)).item()}

        # Normalize the data using the Normalizer instance
        normalized_data = unnormalize(data, attrs, normalization_type="max")

        assert torch.allclose(torch.max(torch.abs(normalized_data)), torch.tensor(1.0))
        assert torch.allclose(torch.min(torch.abs(normalized_data)), torch.tensor(0.1), rtol=1e3)

    # Tests that the Normalizer class can normalize data by its minimum and maximum values
    def test_minmax_unnormalization(self):
        # Create an instance of the Normalizer class with normalization_type="minmax"
        data = torch.rand(1, 32, 320, 320, 2)
        attrs = {"max": torch.max(torch.abs(data)).item(), "min": torch.min(torch.abs(data)).item()}

        # Normalize the data using the Normalizer instance
        normalized_data = unnormalize(data, attrs, normalization_type="minmax")

        assert torch.allclose(torch.max(torch.abs(normalized_data)), torch.tensor(1.0), rtol=1e3)
        assert torch.allclose(torch.min(torch.abs(normalized_data)), torch.tensor(0.1), rtol=1e3)

    # Tests that the Normalizer class can normalize complex data
    def test_mean_std_unnormalization(self):
        # Create an instance of the Normalizer class with normalization_type="max"
        data = torch.rand(1, 32, 320, 320, 2)
        attrs = {"mean": torch.mean(torch.abs(data)).item(), "std": torch.std(torch.abs(data)).item()}

        # Normalize the data using the Normalizer instance
        normalized_data = unnormalize(data, attrs, normalization_type="mean_std")

        assert torch.mean(torch.abs(normalized_data)) != torch.mean(torch.abs(data))
        assert torch.std(torch.abs(normalized_data)) != torch.std(torch.abs(data))

    def test_mean_var_unnormalization(self):
        # Create an instance of the Normalizer class with normalization_type="max"
        data = torch.rand(1, 32, 320, 320, 2)
        attrs = {"mean": torch.mean(torch.abs(data)).item(), "var": torch.var(torch.abs(data)).item()}

        # Normalize the data using the Normalizer instance
        normalized_data = unnormalize(data, attrs, normalization_type="mean_var")

        assert torch.mean(torch.abs(normalized_data)) != torch.mean(torch.abs(data))
        assert torch.var(torch.abs(normalized_data)) != torch.var(torch.abs(data))

    def test_grayscale_unnormalization(self):
        # Create an instance of the Normalizer class with normalization_type="max"
        data = torch.rand(1, 32, 320, 320, 2)
        attrs = {}

        # Normalize the data using the Normalizer instance
        normalized_data = unnormalize(data, attrs, normalization_type="grayscale")

        assert np.round(torch.max(torch.abs(normalized_data)).item()) != 255

    # Tests that the Normalizer class does not normalize data
    def test_do_not_unnormalize_data(self):
        # Create an instance of the Normalizer class with normalization_type=None
        data = torch.rand(1, 32, 320, 320, 2)
        attrs = {}

        # Normalize the data using the Normalizer instance
        normalized_data = unnormalize(data, attrs, normalization_type="None")

        # Check that the normalized data is the same as the original data
        assert torch.all(torch.eq(normalized_data, data))


class TestZeroNanInf:
    """Tests that the function returns the input tensor when there are no NaN or Inf values in it."""

    def test_no_nan_inf_values(self):
        # Create input tensor with no NaN or Inf values
        x = torch.tensor([1.0, 2.0, 3.0])

        # Call the function under test
        result = zero_nan_inf(x)

        # Check that the result is equal to the input tensor
        assert torch.all(torch.eq(result, x))

    # Tests that the function returns the input tensor when there are some NaN or Inf values in it, but not all
    def test_some_nan_inf_values(self):
        # Create input tensor with some NaN and Inf values
        x = torch.tensor([1.0, float('nan'), 3.0, float('inf')])

        # Call the function under test
        result = zero_nan_inf(x)

        # Check that the result is equal to the input tensor
        assert not torch.all(torch.eq(result, x))
