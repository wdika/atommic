# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from atommic.collections.common.parts.coil_sensitivity_maps import MaximumEigenvaluePowerMethod


@pytest.mark.parametrize("size", [20, 30])
def test_power_method(size):
    mat = torch.rand((size, size)) + torch.rand((size, size)) * 1j
    x0 = torch.ones(size) + 0 * 1j

    def A(x):
        return mat @ x

    algo = MaximumEigenvaluePowerMethod(A)
    algo.fit(x0)

    all_eigenvalues = torch.linalg.eig(mat).eigenvalues
    max_eig_torch = all_eigenvalues[all_eigenvalues.abs().argmax()]

    assert torch.allclose(algo.max_eig, max_eig_torch, 0.001)
